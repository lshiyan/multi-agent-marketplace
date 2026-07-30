#!/usr/bin/env python3
"""Script to run marketplace experiments using YAML configuration files."""

import socket
import random
import asyncio
from datetime import datetime
from pathlib import Path

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
    num_runs: int = 1
):
    """Run a marketplace experiment using YAML configuration files."""
    # Load businesses and customers from YAML files
    data_dir = Path(data_dir)
    businesses_dir = data_dir / "businesses"
    customers_dir = data_dir / "customers"

    print(f"Loading data from: {data_dir}")
    businesses = load_businesses_from_yaml(businesses_dir)
    customers = load_customers_from_yaml(customers_dir)

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

                    all_request_outcomes.extend(
                        period_request_outcomes
                    )

                    for business_agent in business_agents:
                        await business_agent.update_prices(
                            request_outcomes=period_request_outcomes
                        )
                        
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

        print(f"\nRun analytics with: magentic-marketplace analyze {experiment_name}")
