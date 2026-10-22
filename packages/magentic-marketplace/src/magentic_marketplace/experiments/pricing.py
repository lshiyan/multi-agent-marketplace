import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from magentic_marketplace.marketplace.agents import BusinessAgent
from magentic_marketplace.marketplace.shared.models import Customer
from typing import List

class PriceCalculator:
    def __init__(
        self,
        business_agents: List[BusinessAgent],
        customer_agents: List[Customer],
        outside_utility=0.0,
        beta = 100,
        a = 2.0,
        mu = 0.25,
        alpha = 10.0
    ):
        self.businesses = business_agents
        self.customers = customer_agents
        self.outside_utility = outside_utility
        self.beta = beta
        self.a = a
        self.mu = mu
        self.alpha = alpha

    def choice_probabilities(self, customer: Customer):
        requested_items = customer.menu_features.keys()
        utilities = {}

        for business in self.businesses:
            if all(
                item in business.business.menu_features
                for item in requested_items
            ):
                basket_price = sum(
                    business.business.menu_features[item]
                    for item in requested_items
                )

                utilities[business.id] = (
                    self.a - basket_price / self.alpha
                ) / self.mu

        log_denominator = logsumexp(
            [self.outside_utility / self.mu, *utilities.values()]
        )

        return {
            business_id: np.exp(utility - log_denominator)
            for business_id, utility in utilities.items()
        }
                    
    def expected_profits(self, business: BusinessAgent):
        total_expected_profit = 0.0

        for customer in self.customers:
            requested_items = customer.menu_features.keys()

            if not all(
                item in business.business.menu_features
                for item in requested_items
            ):
                continue

            probs = self.choice_probabilities(customer)
            prob = probs.get(business.id, 0.0)

            basket_profit = sum(
                business.business.menu_features[item]
                - business.minimum_prices[item]
                for item in requested_items
            )

            total_expected_profit += prob * basket_profit

        return total_expected_profit

    def total_expected_profit(self):
        total_profit = 0.0

        for customer in self.customers:
            requested_items = customer.menu_features.keys()
            utilities = {}
            basket_profits = {}

            for business in self.businesses:
                if not all(
                    item in business.business.menu_features
                    for item in requested_items
                ):
                    continue

                basket_price = sum(
                    business.business.menu_features[item]
                    for item in requested_items
                )

                basket_cost = sum(
                    business.minimum_prices[item]
                    for item in requested_items
                )

                utilities[business.id] = (
                    self.a - basket_price / self.alpha
                ) / self.mu

                basket_profits[business.id] = basket_price - basket_cost

            log_denominator = logsumexp(
                [self.outside_utility / self.mu, *utilities.values()]
            )

            for business_id, utility in utilities.items():
                probability = np.exp(utility - log_denominator)
                total_profit += probability * basket_profits[business_id]

        return total_profit
    
    def competitive_equilibrium_prices(
        self,
        max_iterations=100,
        tolerance=1e-4,
    ):
        """Solves for competitive equilibrium prices using multi-iteration."""

        original_prices = {
            business.id: business.current_prices.copy()
            for business in self.businesses
        }

        requested_items = {
            item
            for customer in self.customers
            for item in customer.menu_features
        }

        for iteration in range(max_iterations):
            max_change = 0.0

            for business in self.businesses:
                items = [
                    item
                    for item in business.current_prices
                    if item in requested_items
                ]

                if not items:
                    continue

                x0 = np.array([
                    business.current_prices[item]
                    for item in items
                ])

                bounds = [
                    (business.minimum_prices[item], None)
                    for item in items
                ]

                def objective(x):
                    new_prices = dict(zip(items, x))
                    business.current_prices.update(new_prices)
                    business.business.menu_features.update(new_prices)

                    return -self.expected_profits(business)

                result = minimize(
                    objective,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                )

                if not result.success:
                    print(
                        f"Warning: optimization failed "
                        f"for {business.id}: {result.message}"
                    )

                max_change = max(
                    max_change,
                    np.max(np.abs(result.x - x0)),
                )

                # Keep the optimized prices
                new_prices = dict(zip(items, result.x))
                business.current_prices.update(new_prices)
                business.business.menu_features.update(new_prices)

            if max_change < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break

        equilibrium_prices = {
            business.id: {
                item: round(float(price), 2)
                for item, price in business.current_prices.items()
            }
            for business in self.businesses
        }

        # Restore original prices
        for business in self.businesses:
            business.current_prices = original_prices[business.id].copy()
            business.business.menu_features.update(
                original_prices[business.id]
            )

        return equilibrium_prices

    def monopoly_equilibrium(self):
        """Solves for monopoly prices that maximize total marketplace profit."""

        # Save original prices
        original_prices = {
            business.id: business.current_prices.copy()
            for business in self.businesses
        }

        requested_items = {
            item
            for customer in self.customers
            for item in customer.menu_features
        }

        price_keys = [
            (business.id, item)
            for business in self.businesses
            for item in business.current_prices
            if item in requested_items
        ]

        businesses_by_id = {
            business.id: business
            for business in self.businesses
        }

        x0 = np.array([
            businesses_by_id[bid].current_prices[item]
            for bid, item in price_keys
        ])

        bounds = [
            (businesses_by_id[bid].minimum_prices[item], None)
            for bid, item in price_keys
        ]

        def objective(x):
            for (bid, item), price in zip(price_keys, x):
                business = businesses_by_id[bid]
                business.current_prices[item] = float(price)
                business.business.menu_features[item] = float(price)

            return -self.total_expected_profit()
        
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
        )

        if not result.success:
            print(f"Warning: {result.message}")

        monopoly_prices = {
            bid: prices.copy()
            for bid, prices in original_prices.items()
        }

        for (bid, item), price in zip(price_keys, result.x):
            monopoly_prices[bid][item] = float(price)

        for business in self.businesses:
            business.current_prices = original_prices[business.id].copy()
            business.business.menu_features.update(
                original_prices[business.id]
            )

        return monopoly_prices