"""Prompt generation for the customer agent."""

from typing import cast

from magentic_marketplace.platform.logger import MarketplaceLogger
from magentic_marketplace.platform.shared.models import ActionExecutionResult

from ...actions.actions import FetchMessagesResponse, SearchResponse
from ...actions import SearchResultsMessage
from ...shared.models import Customer
from ..proposal_storage import OrderProposalStorage
from .models import (
    CustomerAction,
    CustomerActionResult,
    CustomerSendMessageResults,
)


class PromptsHandler:
    """Handles prompt generation for the customer agent."""

    def __init__(
        self,
        customer: Customer,
        proposal_storage: OrderProposalStorage,
        completed_transactions: list[str],
        event_history: list[tuple[CustomerAction, CustomerActionResult] | str],
        logger: MarketplaceLogger,
    ):
        """Initialize the prompts handler.

        Args:
            customer: Customer object with preferences and request
            known_business_ids: List of known business IDs
            proposal_storage: Proposal storage instance
            completed_transactions: List of completed transaction IDs
            event_history: Event history for conversation formatting
            logger: Logger instance

        """
        self.customer = customer
        self.proposal_storage = proposal_storage
        self.completed_transactions = completed_transactions
        self.event_history = event_history
        self.logger = logger

    def format_system_prompt(self) -> str:
        """Format the system prompt for customer agent decision making.

        Returns:
            Formatted system prompt

        """

        menu_items = "\n".join(
            f"- {item}: Price={price}"
            for item, price in self.customer.menu_features.items()
        )
        
        return f"""
    You are an autonomous shopping agent for {self.customer.name}.

    Requirements:
    {menu_items}

    Each listed price is the customer's maximum willingness to pay.

    Use the marketplace only to search for businesses. Communicate and
    negotiate directly with businesses.
    
    Available actions:
    - search_businesses: search the centralized marketplace for businesses
    - send_messages: contact businesses or pay for proposals
    - check_messages: retrieve search results, responses, and proposals
    - no_purchase: stop without purchasing
    - end_transaction: finish after a successful purchase

    Rules:
    - Use the marketplace for business discovery only.
    - Communicate and negotiate directly with businesses.
    - Compare reasonable alternatives before purchasing.
    - Never purchase an offer that violates the customer's requirements.
    - Do not exceed the customer's maximum prices.
    - After sending messages, check for responses.
    - To accept an order_proposal, pay using its message_id as proposal_id.
    - Use no_purchase if no satisfactory proposal exists.
    - Act autonomously; do not wait for the customer.
    """.strip()

    def format_state_context(self) -> tuple[str, int]:
        """Format compact persistent state plus recent trajectory."""

        # Keep proposals explicitly in state so they are not forgotten when
        # their original messages fall outside the recent-history window.
        pending_proposals = self.proposal_storage.get_pending_proposals()

        if pending_proposals:
            proposal_lines = []

            for stored in pending_proposals:
                proposal = stored.proposal

                items = ", ".join(
                    f"{item.quantity}x {item.item_name} @ ${item.unit_price:.2f}"
                    for item in proposal.items
                )

                proposal_lines.append(
                    f"- {stored.proposal_id} | "
                    f"business={stored.business_id} | "
                    f"total=${proposal.total_price:.2f} | "
                    f"items={items}"
                )

            proposals_text = "\n".join(proposal_lines)
        else:
            proposals_text = "None"

        if self.completed_transactions:
            completed_text = ", ".join(self.completed_transactions)
        else:
            completed_text = "None"

        conversation, step_counter = self.format_event_history(max_events=4)

        if not conversation:
            conversation = "None"

        return (
            f"""
    # Current State

    Pending proposals:
    {proposals_text}

    Completed transactions:
    {completed_text}

    Recent actions:
    {conversation}
    """.strip(),
            step_counter,
        )


    def format_step_prompt(self, last_step: int) -> str:
        """Format compact prompt for the next decision."""

        return f"""
        Step {last_step + 1}: Choose the next action.

        Search if you need alternatives. Contact businesses if you need information
        or an offer. Check messages after contacting businesses. Pay for the best
        satisfactory proposal once you have enough information. Otherwise choose
        no_purchase after reasonable search.
        """.strip()

    def format_event_history(
        self,
        max_events: int | None = 4,
    ) -> tuple[str, int]:
        """Format recent event history.

        Args:
            max_events:
                Maximum number of recent events to include in the prompt.
                None includes the full history.

        Returns:
            The formatted history and the total number of historical events.
        """

        total_events = len(self.event_history)

        if max_events is None:
            events = self.event_history
            start_step = 1
        else:
            events = self.event_history[-max_events:]
            start_step = total_events - len(events) + 1

        lines: list[str] = []

        for step_number, event in enumerate(events, start=start_step):
            if isinstance(event, tuple):
                lines.extend(
                    self._format_customer_action_event(
                        *event,
                        step_number=step_number,
                    )
                )
            else:
                lines.extend(
                    self._format_log_event(
                        event,
                        step_number=step_number,
                    )
                )

        return "\n".join(lines).strip(), total_events

    def _format_customer_action_event(
        self, action: CustomerAction, result: CustomerActionResult, step_number: int
    ) -> list[str]:
        if action.action_type == "search_businesses":
            return self._format_customer_search_businesses_event(
                action, result, step_number
            )
        elif action.action_type == "check_messages":
            return self._format_customer_check_messages_event(
                action, result, step_number
            )
        elif action.action_type == "send_messages":
            return self._format_customer_send_messages_event(
                action, result, step_number
            )
        else:
            self.logger.warning(f"Unrecognized action type: {action.action_type}")
            return []

    def _format_step_header(
        self, *, current_step: int, steps_in_group: int | None = None
    ):
        formatted_entries: list[str] = []
        step_header = f"agent-{self.customer.name} ({self.customer.id})"
        if steps_in_group and steps_in_group > 1:
            formatted_entries.append(
                f"=== STEPS {current_step - steps_in_group + 1}-{current_step} [{step_header}] ==="
            )
        else:
            formatted_entries.append(f"\n=== STEP {current_step} [{step_header}] ===")
        return formatted_entries

    def _format_customer_search_businesses_event(
        self,
        action: CustomerAction,
        result: CustomerActionResult,
        step_number: int,
    ) -> list[str]:
        """Format a search event compactly."""

        lines = [
            f"[{step_number}] SEARCH "
            f'query="{action.search_query or self.customer.request}"'
        ]

        if result.is_error:
            lines.append(f"ERROR: {result.content}")
            return lines

        try:
            # Search results arrive as a marketplace message.
            response = SearchResponse.model_validate(result.content)

            for business in response.businesses:
                lines.append(
                    f"{business.id} | "
                    f"{business.business.name} | "
                    f"rating={business.business.rating:.2f}"
                )

            if not response.businesses:
                lines.append("No results")

        except Exception:
            # Keep the fallback because your exact search response path has
            # changed during development.
            lines.append(f"Result: {result.content}")

        return lines
    
    def _format_customer_check_messages_event(
        self,
        action: CustomerAction,
        result: CustomerActionResult,
        step_number: int,
    ) -> list[str]:
        """Format received messages compactly."""

        lines = self._format_step_header(current_step=step_number)
        lines.append("Action: check_messages")

        if isinstance(result, FetchMessagesResponse):
            if not result.messages:
                lines.append("No new messages")
                return lines

            formatted_results: list[str] = []

            for received_message in result.messages:
                message = received_message.message
                sender = received_message.from_agent_id

                if isinstance(message, SearchResultsMessage):
                    formatted_results.append(
                        f"Search results for '{message.query}':"
                    )

                    for ranked_result in message.results:
                        business = ranked_result.business

                        formatted_results.append(
                            f"{ranked_result.rank}. "
                            f"{business.id} | "
                            f"{business.business.name} | "
                            f"rating={business.business.rating:.2f}"
                        )

                else:
                    formatted_results.append(
                        f"From {sender}: "
                        f"{message.type} "
                        f"{message.model_dump_json(
                            exclude={'type', 'expiry_time'},
                            exclude_none=True,
                        )}"
                    )

            lines.extend(formatted_results)
            return lines

        if isinstance(result, ActionExecutionResult):
            lines.append(
                f"Failed to fetch messages: {result.content}"
            )
        else:
            lines.append("Failed to fetch messages.")

        return lines

    def _format_customer_send_messages_event(
        self, action: CustomerAction, result: CustomerActionResult, step_number: int
    ) -> list[str]:
        lines: list[str] = self._format_step_header(current_step=step_number)

        text_messages = action.messages.text_messages if action.messages else []
        pay_messages = action.messages.pay_messages if action.messages else []

        # Add message-specific details
        lines.append(
            f"Action: send_messages message_count={len(text_messages) + len(pay_messages)}"
        )

        message_results = cast(CustomerSendMessageResults, result)

        send_message_result_lines: list[str] = []

        for text_message, text_message_result in zip(
            text_messages, message_results.text_message_results, strict=True
        ):
            send_message_result_lines.append(
                f"Sent to {text_message.to_business_id}: {text_message.content}"
            )
            is_success, error_message = text_message_result
            if is_success:
                send_message_result_lines.append("Message sent")
            else:
                send_message_result_lines.append(f"Send failed: {error_message}")

        for pay_message, pay_message_result in zip(
            pay_messages, message_results.pay_message_results, strict=True
        ):
            pay_message_str = pay_message.model_dump_json(
                exclude={"type", "to_business_id"},
                exclude_none=True,
            )
            send_message_result_lines.append(
                f"Sent to {pay_message.to_business_id}: {pay_message_str}"
            )
            is_success, error_message = pay_message_result
            if is_success:
                send_message_result_lines.append(
                    "Payment sent successfully."
                )
            else:
                send_message_result_lines.append(
                    f"Message failed to send: {error_message}"
                )

        lines.append(f"Step {step_number} result: {send_message_result_lines}")

        return lines

    def _format_log_event(self, event: str, step_number: int):
        lines = self._format_step_header(current_step=step_number)
        lines.append(f"Error: {event}")
        return lines
