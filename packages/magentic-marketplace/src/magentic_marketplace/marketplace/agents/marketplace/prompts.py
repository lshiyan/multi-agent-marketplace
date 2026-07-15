from magentic_marketplace.platform.logger import MarketplaceLogger
from .models import MarketplaceAction, MarketplaceRequestSession, MarketplaceActionResult
from ...actions.actions import InspectBusinessResponse, SearchResponse
from .models import OrderProposal

class PromptsHandler:
    """Prompt generation for a centralized marketplace/platform agent."""

    def __init__(
        self,
        marketplace_agent_id: str,
        session: MarketplaceRequestSession,
        logger: MarketplaceLogger,
    ):
        self.marketplace_agent_id = marketplace_agent_id
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
   - required menu items
   - required amenities

2. Search the marketplace:
   - find relevant businesses directly
   - inspect details for promising candidates
   - discard businesses that cannot satisfy requirements

3. Generate candidate proposals:
   - create feasible proposals from business information
   - ensure each proposal is grounded in real business capabilities
   - do not invent unavailable products, prices, or services

6. Finalize:
   - end the transaction only after successful execution

# Important Rules

- You represent the marketplace as a centralized coordinator.
- You directly use marketplace data and platform actions.
- Do not invent business capabilities, only use information that you have obtained through searching or inspecting businesses.
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

    Use 'search_businesses' when you want to search for relevant businesses related to a customer's request.
    
    Use 'inspect_business' when you want to find more detailed information regarding a business.

    Use 'create_order_proposal' when you've decided a suitable business to fulfill the customer's request.

    Continue taking actions until the customer's request has been successfully fulfilled or cannot be completed. Only end the transaction after payment has succeeded and the purchase has been confirmed. If fulfillment is impossible, explain the reason clearly.

    Choose the next action carefully.
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
        elif action.action_type == "create_order_proposal":
            return self._format_marketplace_order_proposal_event(
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
                
            lines.append("Available Amenities:")
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

    def _format_marketplace_order_proposal_event(
        self,
        action: MarketplaceAction,
        result: MarketplaceActionResult,
        step_number: int,
    ) -> list[str]:
        """Format a create_order_proposal action and its result."""

        lines: list[str] = self._format_step_header(current_step=step_number)

        proposal = result
         
        lines.append(
            "Action: create_order_proposal: "
            f"{action.model_dump_json(include={'business_id', 'request_details'})}"
        )
        lines.append(
            f"Step {step_number} result: Successfully created order proposal."
        )
        lines.append(f"Proposal ID: {proposal.id}")

        business_id = getattr(action, "business_id", None)
        if business_id:
            lines.append(f"Business ID: {business_id}")

        lines.append("Order items:")

        for item in proposal.items:
            lines.append(
                f"- {item.item_name}: "
                f"{item.quantity} × ${item.unit_price:.2f} "
                f"= ${item.quantity * item.unit_price:.2f}"
            )

        lines.append(f"Total price: ${proposal.total_price:.2f}")

        return lines
        
    def _format_log_event(self, event: str, step_number: int):
        lines = self._format_step_header(current_step=step_number)
        lines.append(f"Error: {event}")
        return lines