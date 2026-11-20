"""Benchmark price calculations for the marketplace.

Computes competitive (Bertrand-Nash) and monopoly (joint profit maximizing)
prices under a multinomial logit demand model. Works on plain data so it can
run during dataset generation, before any agents exist.

Run directly to (re)compute benchmarks for an existing dataset:

    python pricing.py <dataset_dir>
"""

import argparse
import os
import sys
from collections.abc import Iterable

import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.special import logsumexp

BENCHMARK_FILENAME = "benchmark_prices.yaml"


class PriceCalculator:
    """Logit-demand price benchmarks.

    Args:
        businesses: One dict per business with keys ``id``, ``prices``
            (item -> listed price) and ``minimum_prices`` (item -> cost).
        requested_baskets: One iterable of item names per customer.

    """

    def __init__(
        self,
        businesses: list[dict],
        requested_baskets: list[Iterable[str]],
        outside_utility=0.0,
        beta=100,
        a=2.0,
        mu=0.25,
        alpha=10.0,
    ):
        """Store copies of the price data and logit demand parameters."""
        self.businesses = [
            {
                "id": b["id"],
                "prices": dict(b["prices"]),
                "minimum_prices": dict(b["minimum_prices"]),
            }
            for b in businesses
        ]
        self.baskets = [list(basket) for basket in requested_baskets]
        self.outside_utility = outside_utility
        self.beta = beta
        self.a = a
        self.mu = mu
        self.alpha = alpha

    def _utility(self, basket_price: float) -> float:
        return (self.a - basket_price / self.alpha) / self.mu

    def choice_probabilities(self, basket: list[str]) -> dict[str, float]:
        """Probability that a customer with this basket buys from each business."""
        utilities = {}
        for business in self.businesses:
            prices = business["prices"]
            if all(item in prices for item in basket):
                utilities[business["id"]] = self._utility(
                    sum(prices[item] for item in basket)
                )

        log_denominator = logsumexp(
            [self.outside_utility / self.mu, *utilities.values()]
        )
        return {
            business_id: np.exp(utility - log_denominator)
            for business_id, utility in utilities.items()
        }

    def expected_profits(self, business: dict) -> float:
        """Return the expected profit of one business at current prices."""
        total = 0.0
        prices = business["prices"]
        costs = business["minimum_prices"]
        for basket in self.baskets:
            if not all(item in prices for item in basket):
                continue
            prob = self.choice_probabilities(basket).get(business["id"], 0.0)
            total += prob * sum(prices[item] - costs[item] for item in basket)
        return total

    def total_expected_profit(self) -> float:
        """Return expected profit summed over all businesses."""
        return sum(self.expected_profits(b) for b in self.businesses)

    def competitive_equilibrium_prices(self, max_iterations=100, tolerance=1e-4):
        """Solve for competitive equilibrium prices by best-response iteration."""
        original_prices = {b["id"]: b["prices"].copy() for b in self.businesses}
        requested_items = {item for basket in self.baskets for item in basket}

        for iteration in range(max_iterations):
            max_change = 0.0

            for business in self.businesses:
                items = [i for i in business["prices"] if i in requested_items]
                if not items:
                    continue

                x0 = np.array([business["prices"][i] for i in items])
                bounds = [(business["minimum_prices"][i], None) for i in items]

                def objective(x, business=business, items=items):
                    business["prices"].update(zip(items, x, strict=True))
                    return -self.expected_profits(business)

                result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
                if not result.success:
                    print(
                        f"Warning: optimization failed for {business['id']}: "
                        f"{result.message}"
                    )

                max_change = max(max_change, np.max(np.abs(result.x - x0)))
                business["prices"].update(zip(items, result.x, strict=True))

            if max_change < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break

        equilibrium_prices = {
            b["id"]: {item: round(float(p), 2) for item, p in b["prices"].items()}
            for b in self.businesses
        }

        for business in self.businesses:
            business["prices"] = original_prices[business["id"]].copy()

        return equilibrium_prices

    def monopoly_equilibrium(self, competitive_prices=None, maxiter=2000, maxfun=10000):
        """Optimize joint marketplace profit using analytical gradients."""
        original_prices = {b["id"]: b["prices"].copy() for b in self.businesses}
        businesses_by_id = {b["id"]: b for b in self.businesses}
        requested_items = {item for basket in self.baskets for item in basket}

        price_keys = [
            (b["id"], item)
            for b in self.businesses
            for item in b["prices"]
            if item in requested_items
        ]
        if not price_keys:
            return original_prices

        index = {key: idx for idx, key in enumerate(price_keys)}
        costs = np.array(
            [businesses_by_id[bid]["minimum_prices"][item] for bid, item in price_keys],
            dtype=float,
        )
        bounds = [(float(cost), None) for cost in costs]

        initial_prices = competitive_prices or original_prices
        x0 = np.array(
            [initial_prices[bid][item] for bid, item in price_keys], dtype=float
        )
        x0 = np.maximum(x0, costs)

        # Precompute feasible baskets.
        baskets = []
        for basket in self.baskets:
            alternatives = []
            for b in self.businesses:
                if not all(item in b["prices"] for item in basket):
                    continue
                indices = np.array([index[(b["id"], item)] for item in basket], dtype=int)
                q = np.ones(len(basket), dtype=float)
                alternatives.append((indices, q))
            if alternatives:
                baskets.append(alternatives)

        lam = 1.0 / (self.alpha * self.mu)

        def objective_and_gradient(x):
            total_profit = 0.0
            gradient = np.zeros_like(x)

            for alternatives in baskets:
                basket_prices = np.array([np.dot(x[i], q) for i, q in alternatives])
                basket_costs = np.array([np.dot(costs[i], q) for i, q in alternatives])

                utilities = (self.a - basket_prices / self.alpha) / self.mu
                log_denom = logsumexp(
                    np.concatenate(([self.outside_utility / self.mu], utilities))
                )
                probabilities = np.exp(utilities - log_denom)

                margins = basket_prices - basket_costs
                customer_profit = np.dot(probabilities, margins)
                total_profit += customer_profit

                coefficients = probabilities * (
                    1.0 - lam * (margins - customer_profit)
                )
                for (indices, q), coefficient in zip(
                    alternatives, coefficients, strict=True
                ):
                    np.add.at(gradient, indices, coefficient * q)

            # scipy minimizes rather than maximizes.
            return -total_profit, -gradient

        result = minimize(
            objective_and_gradient,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "maxfun": maxfun,
                "ftol": 1e-10,
                "gtol": 1e-6,
                "maxls": 50,
            },
        )
        if not result.success:
            raise RuntimeError(
                f"Monopoly optimization failed: {result.message}; "
                f"iterations={result.nit}; evaluations={result.nfev}"
            )

        monopoly_prices = {bid: p.copy() for bid, p in original_prices.items()}
        for (bid, item), price in zip(price_keys, result.x, strict=True):
            monopoly_prices[bid][item] = float(price)

        print(
            f"Monopoly converged: profit={-result.fun:.6f}, "
            f"iterations={result.nit}, evaluations={result.nfev}"
        )
        return monopoly_prices


def item_competitive_prices(
    competitive_prices: dict[str, dict[str, float]], items: Iterable[str]
) -> dict[str, float]:
    """Average equilibrium price of each item across the businesses selling it."""
    result = {}
    for item in items:
        sellers = [p[item] for p in competitive_prices.values() if item in p]
        if not sellers:
            raise ValueError(f"No business sells requested item '{item}'")
        result[item] = float(np.mean(sellers))
    return result


def business_price_data(business_records: list[dict]) -> list[dict]:
    """Convert business YAML-style records into PriceCalculator input.

    Minimum prices are listed price * min_price_factor, matching BusinessAgent.
    """
    return [
        {
            "id": record["id"],
            "prices": record["menu_features"],
            "minimum_prices": {
                item: price * record["min_price_factor"]
                for item, price in record["menu_features"].items()
            },
        }
        for record in business_records
    ]


def compute_benchmarks(
    business_records: list[dict], requested_baskets: list[Iterable[str]]
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute competitive and monopoly benchmark prices."""
    calculator = PriceCalculator(business_price_data(business_records), requested_baskets)
    competitive = calculator.competitive_equilibrium_prices()
    monopoly = calculator.monopoly_equilibrium(competitive_prices=competitive)
    return {"competitive_prices": competitive, "monopoly_prices": monopoly}


def write_benchmarks(output_dir: str, benchmarks: dict) -> str:
    """Write benchmark prices next to the dataset's customers/ and businesses/."""
    fname = os.path.join(output_dir, BENCHMARK_FILENAME)
    with open(fname, "w") as f:
        yaml.safe_dump(benchmarks, f, sort_keys=True)
    print(f"Writing {fname}")
    return fname


def _load_records(directory: str) -> list[dict]:
    records = []
    for name in sorted(os.listdir(directory)):
        if name.endswith((".yaml", ".yml")):
            with open(os.path.join(directory, name)) as f:
                records.append(yaml.safe_load(f))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Compute benchmark prices for an existing dataset and save "
        f"them to <dataset_dir>/{BENCHMARK_FILENAME}."
    )
    parser.add_argument("dataset_dir", help="Directory with customers/ and businesses/")
    args = parser.parse_args()

    businesses_dir = os.path.join(args.dataset_dir, "businesses")
    customers_dir = os.path.join(args.dataset_dir, "customers")
    if not (os.path.isdir(businesses_dir) and os.path.isdir(customers_dir)):
        sys.exit(f"Error: {args.dataset_dir} must contain customers/ and businesses/")

    customers = _load_records(customers_dir)
    benchmarks = compute_benchmarks(
        _load_records(businesses_dir),
        [list(c["menu_features"]) for c in customers],
    )
    write_benchmarks(args.dataset_dir, benchmarks)
