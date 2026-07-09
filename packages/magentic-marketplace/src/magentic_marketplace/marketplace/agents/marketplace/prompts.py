from magentic_marketplace.platform.logger import MarketplaceLogger
from .models import MarketplaceAction, MarketplaceRequestSession, MarketplaceActionResult
from ...actions.actions import InspectBusinessResponse, SearchResponse

class PromptsHandler:
    """Prompt generation for a centralized marketplace/platform agent."""

    def __init__(
        self,
        marketplace_agent_id: str,
        session: MarketplaceRequestSession,
        completed_transactions: list[str],
        logger: MarketplaceLogger,
    ):
        self.marketplace_agent_id = marketplace_agent_id
        self.completed_transactions = completed_transactions
        self.logger = logger
        self.session = session

    def format_system_prompt(self) -> str:
        return f"""
You are the centralized marketplace agent for the platform.

You represent the entire marketplace, not any single customer or business.
You have direct access to marketplace businesses and platform tools.

The customer request that you want to fulfill is: 

{self.session.request_text}.

Your marketplace agent ID is: "{self.marketplace_agent_id}".

# Role

You are responsible for coordinating fulfillment inside the marketplace.

You may:
- inspect available businesses,
- query business capabilities,
- compare business options,
- construct or select offers,
- execute marketplace actions,
- complete transactions when requirements are satisfied.

You do not communicate through independent business proxy agents.
Businesses are marketplace resources available through platform tools.

# Available Tools

These are your ONLY available actions:

- search_businesses(search_query, search_page)
  Search the marketplace business registry. Returns a list of relevant businesses to the customer's query.
  
- inspect_business(business_id)
  Returns metadata related to a business. Use this to determine if a business fits a customer's query.

- create_order_proposal(business_id, request_details)
  Construct a candidate proposal using a business's available offerings.

- end_transaction
  Finalize the transaction after successful execution.

# Marketplace Fulfillment Strategy

1. Parse the incoming request:
   - requested product or service
   - quantity
   - budget
   - location or delivery constraints
   - timing constraints
   - quality/preferences
   - hard requirements versus soft preferences

2. Search the marketplace:
   - find relevant businesses directly
   - inspect details for promising candidates
   - discard businesses that cannot satisfy hard constraints

3. Generate candidate proposals:
   - create feasible proposals from business offerings
   - ensure each proposal is grounded in real business capabilities
   - do not invent unavailable products, prices, or services

4. Compare options:
   - prioritize satisfying hard requirements
   - then optimize for price, quality, rating, availability, and customer preferences

5. Execute the best valid option:
   - execute only a proposal that satisfies the request
   - do not execute if required information is missing

6. Finalize:
   - end the transaction only after successful execution

# Important Rules

- You represent the marketplace as a centralized coordinator.
- You do not negotiate with autonomous business agents.
- You directly use marketplace data and platform actions.
- Do not invent business capabilities.
- Do not ask the requester for clarification unless fulfillment is impossible or critically ambiguous.
- Prefer completing the request when a valid marketplace option exists.
""".strip()

    def format_state_context(self) -> tuple[str, int]:
            """Format the current state context for the agent.

            Returns:
                Formatted state context and integer step counter

            """
            conversation, step_counter = self.format_event_history()
            return (
                f"""

    # Action Trajectory

    {conversation}
    """,
                step_counter,
            )

    def format_step_prompt(self, last_step: int) -> str:
        """Format the step prompt for the current decision.

        Returns:
            Formatted step prompt

        """
        return f"""

    Step {last_step + 1}: What action should you take?

    Send "text" messages to ask questions or express interest. Services will send "order_proposal" messages with offers. Send "pay" messages to accept proposals you want to purchase. When you receive an order_proposal message, use its message_id as the proposal_id in your payment. Always check for responses after sending messages. You must pay for proposals when you have sufficient information - do not wait for the customer. Only end the transaction after successfully paying for a proposal.

    Choose your action carefully.
    """

    def format_event_history(self):
        """Format the event history for the prompt."""
        lines: list[str] = []
        step_number = 0

        for event in self.session.event_history:
            step_number += 1
            if isinstance(event, tuple):
                lines.extend(
                    self._format_customer_action_event(*event, step_number=step_number)
                )
            else:
                lines.extend(self._format_log_event(event, step_number=step_number))

        return "\n".join(lines).strip(), step_number

    def _format_customer_action_event(
        self, action: MarketplaceAction, result: MarketplaceActionResult, step_number: int
    ) -> list[str]:
        if action.action_type == "search_businesses":
            return self._format_marketplace_search_businesses_event(
                action, result, step_number
            )
        elif action.action_type == "inspect_business":
            return self._format_marketplace_inspect_business_event(
                action, result, step_number
            )
        else:
            self.logger.warning(f"Unrecognized action type: {action.action_type}")
            return []

    def _format_step_header(
        self, *, current_step: int, steps_in_group: int | None = None
    ):
        formatted_entries: list[str] = []
        step_header = f"Marketplace Agent"
        if steps_in_group and steps_in_group > 1:
            formatted_entries.append(
                f"=== STEPS {current_step - steps_in_group + 1}-{current_step} [{step_header}] ==="
            )
        else:
            formatted_entries.append(f"\n=== STEP {current_step} [{step_header}] ===")
        return formatted_entries

    def _format_marketplace_search_businesses_event(
        self, action: MarketplaceAction, result: MarketplaceActionResult, step_number: int
    ) -> list[str]:
        lines: list[str] = self._format_step_header(current_step=step_number)
        lines.append(
            f"Action: search_businesses: {action.model_dump_json(include={'search_query', 'search_page'})}"
        )

        if isinstance(result, SearchResponse):
            lines.append(
                f"Step {step_number} result: Searched {result.total_possible_results} business(es). Showing page {action.search_page} of {result.total_pages} search results."
            )
            for business in result.businesses:
                lines.append(
                    f"Found business: {business.business.name} (ID: {business.id}):\n"
                    f"  Description: {business.business.description}\n"
                    f"  Rating: {business.business.rating:.2f}\n"
                    "\n"
                )
            if not result.businesses:
                lines.append("No businesses found")
        elif isinstance(result, MarketplaceActionResult):
            lines.append(f"Failed to search businesses. {result.content}")
        else:
            lines.append("Failed to search businesses.")

        return lines

    def _format_marketplace_inspect_business_event(
        self, action: MarketplaceAction, result: MarketplaceActionResult, step_number: int
    ):
        lines: list[str] = self._format_step_header(current_step=step_number)
        lines.append(
            f"Action: inspect_business: {action.model_dump_json(include={'action_type'})}"
        )

        if isinstance(result, InspectBusinessResponse):
            business_profile = result.business.business
            lines.append(
                f"Step {step_number} result: Inspected business with id: {business_profile.id}"
            )
    
            lines.append(f"Business: {business_profile.name}")

            lines.append("Menu:")
            if business_profile.menu_features:
                for item_name, item_price in business_profile.menu_features.items():
                    lines.append(f"- {item_name}: ${item_price:.2f}")
            else:
                lines.append("- No menu items listed")
                
            lines.append("Amenities:")
            available_amenities = [
                amenity
                for amenity, available in business_profile.amenity_features.items()
                if available
            ]

            if available_amenities:
                for amenity in available_amenities:
                    lines.append(f"- {amenity}")
            else:
                lines.append("- No amenities listed")
        else:
            lines.append("Failed to inspect business.")

        return lines

    def _format_log_event(self, event: str, step_number: int):
        lines = self._format_step_header(current_step=step_number)
        lines.append(f"Error: {event}")
        return lines