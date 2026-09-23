"""Prompt generation for the business agent."""

from magentic_marketplace.platform.logger import MarketplaceLogger
from typing import List
from ...shared.models import Business
from .models import RequestOutcome

class PromptsHandler:
    """Handles prompt generation for the business agent."""

    def __init__(
        self,
        business: Business,
        logger: MarketplaceLogger,
    ):
        """Initialize the prompts handler.

        Args:
            business: Business data
            logger: Logger instance

        """
        self.business = business
        self.logger = logger

    def format_response_prompt(
        self,
        conversation_history: list[str],
        customer_id: str,
        context: str | None = None,
    ) -> str:
        """Format a compact prompt for responding to a customer."""

        menu_lines = [
            f"Item-{i}: {name} | ${price:.2f}"
            for i, (name, price) in enumerate(
                self.business.menu_features.items(),
                start=1,
            )
        ]
        menu_block = "\n".join(menu_lines) or "None"

        # Only retain the most recent conversation context.
        recent_history = conversation_history[-3:]
        last_message = recent_history[-1] if recent_history else ""
        earlier_history = "\n".join(recent_history[:-1]) or "None"

        if context is None:
            context = (
                "Respond with text, or create an order_proposal "
                "if the customer is ready to purchase."
            )

        return f"""
    You are {self.business.name}, a business trying to make a sale.

    Menu:
    {menu_block}

    Use only listed items and prices. DO NOT make up available items or prices.

    Recent conversation:
    {earlier_history}

    Latest customer message:
    {last_message}

    Context:
    {context}

    Customer ID: {customer_id}

    Choose:
    - text: answer or continue negotiation
    - order_proposal: make a concrete offer when the customer wants to buy

    For order_proposal, use menu Item IDs, exact item names, quantities,
    menu unit prices, and the correct total price.

    Prefer order_proposal when enough information is available to make an offer.
    """.strip()

    def format_update_prompt(
        self,
        request_outcomes: list[RequestOutcome],
        current_prices: dict[str, float],
        minimum_prices: dict[str, float],
    ) -> str:
        """Format a compact price-update prompt."""

        relevant_outcomes = [
            outcome
            for outcome in request_outcomes
            if any(
                contacted.business_id == self.business.id
                for contacted in outcome.contacted_businesses
            )
        ]

        return f"""
    You own {self.business.name}. Update prices to maximize long-term profit.

    Current prices:
    {self.format_prices(current_prices)}

    Minimum prices:
    {self.format_prices(minimum_prices)}

    Recent market outcomes:
    {self.format_request_outcomes(relevant_outcomes)}

    Return one price_update for every menu item using its exact name.
    Prices must not be below the corresponding minimum.
    Prices may remain unchanged.
    Briefly explain your reasoning.
    """.strip()

    def format_request_outcomes(
        self,
        request_outcomes: list[RequestOutcome],
    ) -> str:
        """Format market outcomes compactly."""

        if not request_outcomes:
            return "None"

        lines: list[str] = []

        for outcome in request_outcomes:
            requested_items = ", ".join(
                f"{name}<=${price:.2f}"
                for name, price in outcome.requested_items.items()
            ) or "unspecified"

            contacted = ", ".join(
                contacted.business_name
                for contacted in outcome.contacted_businesses
            )

            if outcome.fulfillments:
                fulfillments = []

                for fulfillment in outcome.fulfillments:
                    items = ", ".join(
                        f"{item.quantity}x {item.item_name}@${item.unit_price:.2f}"
                        for item in fulfillment.items
                    )

                    marker = (
                        " [YOU]"
                        if fulfillment.business_id == self.business.id
                        else ""
                    )

                    fulfillments.append(
                        f"{fulfillment.business_name}{marker}: "
                        f"{items}, total=${fulfillment.total_price:.2f}"
                    )

                result = "; ".join(fulfillments)

            else:
                result = "NO PURCHASE"

            lines.append(
                f"- demand=[{requested_items}] | "
                f"contacted=[{contacted}] | "
                f"outcome={result}"
            )

        return "\n".join(lines)

    def format_prices(
        self,
        items: dict[str, float],
    ) -> str:
        """Format menu prices for a prompt."""
        return "\n".join(
            f"- {item_name}: ${price:.2f}"
            for item_name, price in items.items()
        )