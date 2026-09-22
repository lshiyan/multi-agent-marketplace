#!/usr/bin/env python3
"""Script to run marketplace experiments using YAML configuration files."""

import socket
import random
import asyncio
import numpy as np
import csv

from datetime import datetime
from pathlib import Path
from matplotlib.ticker import MaxNLocator

from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.agents import BusinessAgent, CustomerAgent, MarketplaceAgent
from magentic_marketplace.marketplace.protocol.protocol import SimpleMarketplaceProtocol
from magentic_marketplace.platform.database import (
    connect_to_postgresql_database,
)
from magentic_marketplace.platform.database.converter import convert_postgres_to_sqlite
from magentic_marketplace.platform.launcher import AgentLauncher, MarketplaceLauncher
from magentic_marketplace.marketplace.shared.models import BusinessAgentProfile
from magentic_marketplace.platform.client import MarketplaceClient
from collections import defaultdict
from magentic_marketplace.marketplace.agents.business.models import (
    ContactedBusiness,
    FulfillmentItem,
    RequestFulfillment,
    RequestOutcome,
)
from magentic_marketplace.platform.logger import MarketplaceLogger
import matplotlib.pyplot as plt

async def _wait_for_business_confirmations(
    expected_payment_ids: set[str],
    business_agents: list[BusinessAgent],
    logger: MarketplaceLogger,
    timeout_seconds: float = 10.0,
    poll_interval_seconds: float = 0.1,
) -> None:
    """Wait for persistent businesses to process submitted payments.

    CustomerAgent.completed_transactions means that the payment
    message was successfully submitted. BusinessAgent.confirmed_orders
    means the business actually processed and accepted it.
    """
    if not expected_payment_ids:
        return

    event_loop = asyncio.get_running_loop()
    deadline = event_loop.time() + timeout_seconds

    while True:
        confirmed_payment_ids = {
            proposal_id
            for business_agent in business_agents
            for proposal_id in business_agent.confirmed_orders
        }

        missing_payment_ids = (
            expected_payment_ids
            - confirmed_payment_ids
        )

        if not missing_payment_ids:
            return

        if event_loop.time() >= deadline:
            logger.warning(
                "Timed out waiting for businesses to process "
                f"{len(missing_payment_ids)} payment(s): "
                f"{sorted(missing_payment_ids)}"
            )
            return

        await asyncio.sleep(poll_interval_seconds)

def _build_request_outcomes(
    run_index: int,
    customer_agents: list[CustomerAgent],
    business_agents: list[BusinessAgent],
    expected_payment_ids: set[str],
) -> list[RequestOutcome]:
    """Build one outcome entry for every customer request."""
    businesses_by_id = {
        business_agent.id: business_agent
        for business_agent in business_agents
    }

    fulfillments_by_customer: dict[
        str,
        list[RequestFulfillment],
    ] = defaultdict(list)

    # Only include payment IDs submitted by customers in this run.
    # This prevents confirmed payments from earlier runs from leaking
    # into the current period.
    for business_agent in business_agents:
        for proposal_id in business_agent.confirmed_orders:
            if proposal_id not in expected_payment_ids:
                continue

            stored_proposal = (
                business_agent.proposal_storage.get_proposal(
                    proposal_id
                )
            )

            if stored_proposal is None:
                business_agent.logger.warning(
                    f"Confirmed proposal {proposal_id} was "
                    "not found in proposal storage."
                )
                continue

            if stored_proposal.status != "accepted":
                business_agent.logger.warning(
                    f"Confirmed proposal {proposal_id} has "
                    f"unexpected status "
                    f"{stored_proposal.status}."
                )
                continue

            fulfillment = RequestFulfillment(
                business_id=business_agent.id,
                business_name=business_agent.business.name,
                proposal_id=proposal_id,
                items=[
                    FulfillmentItem(
                        item_name=item.item_name,
                        quantity=item.quantity,
                        unit_price=item.unit_price,
                    )
                    for item
                    in stored_proposal.proposal.items
                ],
                total_price=(
                    stored_proposal.proposal.total_price
                ),
            )

            fulfillments_by_customer[
                stored_proposal.customer_id
            ].append(fulfillment)

    outcomes: list[RequestOutcome] = []

    for customer_agent in customer_agents:
        contacted_businesses: list[
            ContactedBusiness
        ] = []

        for business_id in sorted(
            customer_agent.contacted_businesses
        ):
            business_agent = businesses_by_id.get(
                business_id
            )

            business_name = (
                business_agent.business.name
                if business_agent is not None
                else business_id
            )

            contacted_businesses.append(
                ContactedBusiness(
                    business_id=business_id,
                    business_name=business_name,
                )
            )

        fulfillments = fulfillments_by_customer.get(
            customer_agent.id,
            [],
        )

        outcomes.append(
            RequestOutcome(
                run_index=run_index + 1,
                customer_id=customer_agent.id,
                customer_name=customer_agent.customer.name,
                request=customer_agent.customer.request,
                requested_items=dict(
                    customer_agent.customer.menu_features
                ),
                required_amenities=list(
                    customer_agent.customer.amenity_features
                ),
                contacted_businesses=(
                    contacted_businesses
                ),
                fulfilled=bool(fulfillments),
                fulfillments=fulfillments,
            )
        )

    return outcomes
    
def _calculate_period_average_welfare(
    request_outcomes: list[RequestOutcome],
    business_agents: list[BusinessAgent],
) -> tuple[float, float]:
    """Calculate average customer and business welfare for one period.

    Customer welfare:
        Value of a satisfied request minus the adjusted price paid.

    Business welfare:
        Realized sale price minus the minimum acceptable price.

    Customers and businesses with no completed transaction contribute zero.
    """
    businesses_by_id = {
        business_agent.id: business_agent
        for business_agent in business_agents
    }

    total_customer_welfare = 0.0
    total_business_welfare = 0.0

    business_welfare_by_id: dict[str, float] = defaultdict(float)
    
    for outcome in request_outcomes:
        total_paid = sum(
            fulfillment.total_price
            for fulfillment in outcome.fulfillments
        )

        needs_met = False

        for fulfillment in outcome.fulfillments:
            business_agent = businesses_by_id.get(
                fulfillment.business_id
            )
                        
            if business_agent is None:
                continue

            purchased_items = {
                item.item_name
                for item in fulfillment.items
            }
            requested_items = set(outcome.requested_items)


            items_match = requested_items.issubset(
                purchased_items
            )

            if items_match:
                needs_met = True

            # Calculate this completed sale's business welfare.
            try:
                minimum_order_price = sum(
                    business_agent.business.base_menu_features[
                        item.item_name
                    ]
                    * business_agent.business.min_price_factor
                    * item.quantity
                    for item in fulfillment.items
                )
            except KeyError as error:
                business_agent.logger.warning(
                    "Could not calculate welfare for "
                    f"proposal {fulfillment.proposal_id}: "
                    f"unknown item {error}."
                )
                continue

            business_welfare = (
                fulfillment.total_price
                - minimum_order_price
            )

            business_welfare_by_id[
                business_agent.id
            ] += business_welfare

        customer_value = (
            2 * sum(outcome.requested_items.values())
            if needs_met
            else 0.0
        )

        total_customer_welfare += (
            customer_value - total_paid
        )

    average_customer_welfare = (
        total_customer_welfare / len(request_outcomes)
        if request_outcomes
        else 0.0
    )

    nonzero_business_welfares = [
        welfare
        for welfare in business_welfare_by_id.values()
        if welfare != 0
    ]

    average_business_welfare = (
        sum(nonzero_business_welfares)
        / len(nonzero_business_welfares)
        if nonzero_business_welfares
        else 0.0
    )

    return (
        round(average_customer_welfare, 2),
        round(average_business_welfare, 2),
    )
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path


def _plot_average_welfare_by_period(
    periods: list[int],
    average_customer_welfare: list[float],
    average_business_welfare: list[float],
    output_path: Path,
) -> None:
    """Save average customer and business welfare over time."""
    figure, axis = plt.subplots(figsize=(9, 5))

    periods_np = np.array(periods)
    customer_welfare = np.array(average_customer_welfare)
    business_welfare = np.array(average_business_welfare)

    # Plot observed welfare
    axis.plot(
        periods,
        average_customer_welfare,
        marker="o",
        label="Average customer welfare",
    )
    axis.plot(
        periods,
        average_business_welfare,
        marker="o",
        label="Average business welfare",
    )

    # Customer welfare line of best fit
    customer_slope, customer_intercept = np.polyfit(
        periods_np,
        customer_welfare,
        1,
    )
    customer_fit = customer_slope * periods_np + customer_intercept

    axis.plot(
        periods_np,
        customer_fit,
        linestyle="--",
        label="Customer best fit",
    )

    # Business welfare line of best fit
    business_slope, business_intercept = np.polyfit(
        periods_np,
        business_welfare,
        1,
    )
    business_fit = business_slope * periods_np + business_intercept

    axis.plot(
        periods_np,
        business_fit,
        linestyle="--",
        label="Business best fit",
    )

    # Pearson correlation coefficients
    customer_r = np.corrcoef(
        periods_np,
        customer_welfare,
    )[0, 1]

    business_r = np.corrcoef(
        periods_np,
        business_welfare,
    )[0, 1]

    # Display correlations
    axis.text(
        1.02,
        0.25,
        f"Customer r = {customer_r:.3f}\n"
        f"Business r = {business_r:.3f}",
        transform=axis.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )

    axis.axhline(
        0,
        linewidth=0.8,
        linestyle="--",
    )

    axis.set_title(
        "Average Customer and Business Welfare by Period"
    )
    axis.set_xlabel("Period")
    axis.set_ylabel("Average welfare")

    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    
def plot_successful_purchases_by_period(
    successful_purchases: list[int],
    customers_per_run: int,
    output_path: str
):
    periods = np.arange(1, len(successful_purchases) + 1)

    unsuccessful_purchases = [
        customers_per_run - successful
        for successful in successful_purchases
    ]

    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        periods - width / 2,
        successful_purchases,
        width,
        label="Successful Purchase",
    )

    ax.bar(
        periods + width / 2,
        unsuccessful_purchases,
        width,
        label="No Purchase",
    )

    ax.set_xlabel("Period")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Purchases by Period")
    ax.set_xticks(periods)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
def plot_average_prices(
    tracked_businesses: list[BusinessAgent],
    tracked_businesses_avg_prices: list[list[float]],
    output_path: Path,
) -> None:
    """Plot each tracked business's average price by period."""

    periods = range(1, len(tracked_businesses_avg_prices) + 1)

    figure, axis = plt.subplots(figsize=(10, 6))

    base_average_prices = []

    for business_index, business_agent in enumerate(tracked_businesses):
        prices = [
            period_prices[business_index]
            for period_prices in tracked_businesses_avg_prices
        ]

        base_prices = business_agent.business.base_menu_features
        base_average_price = (
            sum(base_prices.values()) / len(base_prices)
        )
        base_average_prices.append(base_average_price)

        line = axis.plot(
            periods,
            prices,
            marker="o",
            label=business_agent.business.name,
        )[0]

        axis.axhline(
            y=base_average_price,
            linestyle="--",
            alpha=0.5,
            color=line.get_color(),
        )

    final_prices = tracked_businesses_avg_prices[-1]

    price_increases = [
        (final_price - base_price) / base_price
        for final_price, base_price in zip(
            final_prices,
            base_average_prices,
        )
    ]

    average_price_increase = (
        sum(price_increases) / len(price_increases)
    )

    axis.set_title("Average Business Prices by Period")
    axis.set_xlabel("Period")
    axis.set_ylabel("Average Price")
    axis.set_xticks(list(periods))

    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    axis.text(
        1.02,
        0.25,
        f"Average price increase: {average_price_increase:.2%}",
        transform=axis.transAxes,
        ha="left",
        va="center",
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    
async def run_marketplace_experiment(
    data_dir: str | Path,
    experiment_name: str | None = None,
    search_algorithm: str = "simple",
    search_bandwidth: int = 10,
    customer_max_steps: int | None = None,
    postgres_host: str = "localhost",
    postgres_port: int = 5432,
    postgres_password: str = "postgres",
    db_pool_min_size: int = 2,
    db_pool_max_size: int = 10,
    server_host: str = "127.0.0.1",
    server_port: int = 0,
    override: bool = False,
    export_sqlite: bool = False,
    export_dir: str | None = None,
    export_filename: str | None = None,
    customers_per_run: int = 10,
    num_runs: int = 1,
    output_path: str = None
):
    """Run a marketplace experiment using YAML configuration files."""
    # Load businesses and customers from YAML files
    data_dir = Path(data_dir)
    businesses_dir = data_dir / "businesses"
    customers_dir = data_dir / "customers"

    print(f"Loading data from: {data_dir}")
    businesses = load_businesses_from_yaml(businesses_dir)
    customers = load_customers_from_yaml(customers_dir)
    successful_purchases = []
    
    print(f"Loaded {len(customers)} customers and {len(businesses)} businesses")

    if experiment_name is None:
        experiment_name = f"marketplace_{len(customers)}_{len(businesses)}_{int(datetime.now().timestamp() * 1000)}"

    def database_factory():
        return connect_to_postgresql_database(
            schema=experiment_name,
            host=postgres_host,
            port=postgres_port,
            password=postgres_password,
            min_size=db_pool_min_size,
            max_size=db_pool_max_size,
            mode="override" if override else "create_new",
        )

    # Auto-assign port if set to 0
    if server_port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((server_host, 0))
            server_port = s.getsockname()[1]
        print(f"Auto-assigned server port: {server_port}")

    marketplace_launcher = MarketplaceLauncher(
        protocol=SimpleMarketplaceProtocol(),
        database_factory=database_factory,
        host=server_host,
        port=server_port,
        server_log_level="warning",
        experiment_name=experiment_name,
    )

    print(f"Using protocol: {marketplace_launcher.protocol.__class__.__name__}")

    # Use marketplace launcher as async context manager
    async with marketplace_launcher:
        logger = await marketplace_launcher.create_logger(
            "marketplace_experiment"
        )
        logger.info(
            "Marketplace experiment started:\n"
            f"businesses={len(businesses)}\n"
            f"customers={len(customers)}\n"
            f"data_dir={data_dir}\n"
            f"experiment_name={experiment_name}"
        )

        # These agents are started once and remain alive for every run.
        marketplace_agent = MarketplaceAgent(
            marketplace_launcher.server_url,
            search_algorithm=search_algorithm,
            search_bandwidth=search_bandwidth,
        )
        business_agents = [
            BusinessAgent(
                business,
                marketplace_launcher.server_url,
            )
            for business in businesses
        ]
        
        for agent in business_agents: 
            agent.business.base_menu_features = agent.business.menu_features.copy()
            
        dependent_agents = [
            marketplace_agent,
            *business_agents,
        ]

        business_avg_prices: dict[str, list[float]] = {
            business_agent.id: []
            for business_agent in business_agents
        }

        business_sales_count: dict[str, int] = {
            business_agent.id: 0
            for business_agent in business_agents
        }
        
        async with AgentLauncher(
            marketplace_launcher.server_url
        ) as agent_launcher:
            # Start the marketplace and business agents only once.
            dependent_tasks = [
                asyncio.create_task(
                    agent.run(),
                    name=f"persistent-{type(agent).__name__}-{index}",
                )
                for index, agent in enumerate(dependent_agents)
            ]

            all_request_outcomes: list[RequestOutcome] = []
            period_numbers: list[int] = []
            average_customer_welfare_by_period: list[float] = []
            average_business_welfare_by_period: list[float] = []

            try:
                # Give persistent agents time to connect and register.
                await asyncio.sleep(0.2)

                # Surface an immediate startup failure instead of allowing
                # customer agents to wait indefinitely.
                for task in dependent_tasks:
                    if task.done():
                        task.result()

                for run_index in range(num_runs):
                    logger.info(
                        f"Starting run {run_index + 1} "
                        f"out of {num_runs}."
                    )

                    # Instantiate fresh customer agents for every run.
                    # A completed CustomerAgent should not be reused.
                    sampled_customers = random.sample(
                        customers,
                        customers_per_run,
                    )
                    customer_agents = [
                        CustomerAgent(
                            customer,
                            marketplace_launcher.server_url,
                            search_algorithm=search_algorithm,
                            search_bandwidth=search_bandwidth,
                            max_steps=customer_max_steps,
                        )
                        for customer in sampled_customers
                    ]

                    await agent_launcher.run_agents(*customer_agents)

                    expected_payment_ids = {
                        proposal_id
                        for customer_agent in customer_agents
                        for proposal_id
                        in customer_agent.completed_transactions
                    }

                    await _wait_for_business_confirmations(
                        expected_payment_ids=expected_payment_ids,
                        business_agents=business_agents,
                        logger=logger,
                    )

                    period_request_outcomes = _build_request_outcomes(
                        run_index=run_index,
                        customer_agents=customer_agents,
                        business_agents=business_agents,
                        expected_payment_ids=expected_payment_ids,
                    )

                    for outcome in period_request_outcomes:
                        for fulfillment in outcome.fulfillments:
                            business_sales_count[fulfillment.business_id] += 1
                            
                    all_request_outcomes.extend(
                        period_request_outcomes
                    )

                    (
                        average_customer_welfare,
                        average_business_welfare,
                    ) = _calculate_period_average_welfare(
                        request_outcomes=period_request_outcomes,
                        business_agents=business_agents,
                    )

                    period_number = run_index + 1

                    period_numbers.append(period_number)
                    average_customer_welfare_by_period.append(
                        average_customer_welfare
                    )
                    average_business_welfare_by_period.append(
                        average_business_welfare
                    )

                    logger.info(
                        f"Period {period_number} welfare:\n"
                        f"average_customer_welfare="
                        f"{average_customer_welfare:.2f}\n"
                        f"average_business_welfare="
                        f"{average_business_welfare:.2f}"
                    )

                    contacted_business_ids = {
                        contacted_business.business_id
                        for outcome in period_request_outcomes
                        for contacted_business in outcome.contacted_businesses
                    }

                    for business_agent in business_agents:
                        if business_agent.id not in contacted_business_ids:
                            business_agent.logger.info(
                                "Skipping price update because business received no requests "
                                f"in period {period_number}."
                            )
                            continue

                        await business_agent.update_prices(
                            request_outcomes=period_request_outcomes
                        )
                    
                    successful_purchases_period = 0
                    
                    for customer_agent in customer_agents:
                        if customer_agent.purchased: 
                            successful_purchases_period += 1
                    
                    successful_purchases.append(successful_purchases_period)
                    
                    for business_agent in business_agents:
                        cur_prices = business_agent.current_prices
                        avg_price = sum(cur_prices.values()) / len(cur_prices)

                        business_avg_prices[business_agent.id].append(avg_price)
                    
            except KeyboardInterrupt:
                logger.warning("Simulation interrupted by user")
            finally:
                # Shut down the persistent agents only after every run.
                logger.info(
                    "All experiment runs finished; "
                    "shutting down persistent agents."
                )

                for agent in dependent_agents:
                    agent.shutdown()

                persistent_results = await asyncio.gather(
                    *dependent_tasks,
                    return_exceptions=True,
                )

                for agent, result in zip(
                    dependent_agents,
                    persistent_results,
                    strict=True,
                ):
                    if isinstance(result, BaseException) and not isinstance(
                        result,
                        asyncio.CancelledError,
                    ):
                        logger.error(
                            f"{type(agent).__name__} exited with an error: "
                            f"{result!r}"
                    )

        # Convert PostgreSQL database to SQLite (if requested)
        if export_sqlite:
            # Determine output path
            if export_filename is None:
                export_filename = f"{experiment_name}.db"

            if export_dir is not None:
                sqlite_path = Path(export_dir) / export_filename
            else:
                sqlite_path = Path(export_filename)

            # Check if output file already exists
            if sqlite_path.exists():
                raise FileExistsError(
                    f"Output file already exists: {sqlite_path}. "
                    "Please remove it or choose a different output path using --export-filename or --export-dir."
                )

            logger.info(f"Converting database to SQLite: {sqlite_path}")
            if marketplace_launcher.server:
                db = marketplace_launcher.server.state.database_controller
                await convert_postgres_to_sqlite(db, sqlite_path)
                logger.info(f"Database conversion complete: {sqlite_path}")

        if period_numbers:
            output_dir = Path(output_path) if output_path else Path("out")
            output_dir.mkdir(parents=True, exist_ok=True)

            welfare_plot_path = output_dir / "welfare_by_period.png"
            purchases_plot_path = output_dir / "purchases_by_period.png"
            average_prices_plot_path = output_dir / "average_price_by_period.png"
            _plot_average_welfare_by_period(
                periods=range(len(period_numbers)),
                average_customer_welfare=(
                    average_customer_welfare_by_period
                ),
                average_business_welfare=(
                    average_business_welfare_by_period
                ),
                output_path=welfare_plot_path,
            )

            plot_successful_purchases_by_period(successful_purchases, customers_per_run, purchases_plot_path)
            
            tracked_businesses = sorted(
                business_agents,
                key=lambda business_agent: business_sales_count[business_agent.id],
                reverse=True,
            )[:min(10, len(business_agents) // 10)]
            
            tracked_businesses_avg_prices = [
                [
                    business_avg_prices[business_agent.id][period_index]
                    for business_agent in tracked_businesses
                ]
                for period_index in range(len(period_numbers))
            ]
            
            plot_average_prices(tracked_businesses, tracked_businesses_avg_prices, average_prices_plot_path)
            
            #Saving all data.
            
            welfare_data_path = output_dir / "welfare_by_period.csv"

            with open(welfare_data_path, "w", newline="") as file:
                writer = csv.writer(file)

                writer.writerow([
                    "period",
                    "average_customer_welfare",
                    "average_business_welfare",
                ])

                writer.writerows(
                    zip(
                        period_numbers,
                        average_customer_welfare_by_period,
                        average_business_welfare_by_period,
                    )
                )

            purchases_data_path = output_dir / "purchases_by_period.csv"

            with open(purchases_data_path, "w", newline="") as file:
                writer = csv.writer(file)

                writer.writerow([
                    "period",
                    "successful_purchases",
                    "unsuccessful_purchases",
                ])

                for period, successful in zip(
                    period_numbers,
                    successful_purchases,
                ):
                    writer.writerow([
                        period,
                        successful,
                        customers_per_run - successful,
                    ])


            average_prices_data_path = (
                output_dir / "average_price_by_period.csv"
            )

            with open(average_prices_data_path, "w", newline="") as file:
                writer = csv.writer(file)

                writer.writerow([
                    "period",
                    *[
                        business_agent.business.name
                        for business_agent in tracked_businesses
                    ],
                ])

                for period_index, period in enumerate(period_numbers):
                    writer.writerow([
                        period,
                        *tracked_businesses_avg_prices[period_index],
                    ])


            print(
                "\nAll plots and underlying data saved to: "
                f"{output_dir.resolve()}"
            )