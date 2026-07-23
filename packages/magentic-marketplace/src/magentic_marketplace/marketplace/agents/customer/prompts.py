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
You are an autonomous agent working for customer {self.customer.name} ({self.customer.id}). They have the following request:

{menu_items}. Note that the price is the **maximum** price that they are willing to pay for that item. Obviously, they would be willing to pay any lower price.

Your agent ID is: "{self.customer.id}" and your name is "agent-{self.customer.name} ({self.customer.id})".

IMPORTANT: You do NOT have access to the customer directly. You must autonomously fulfill their request by interacting with the centralized marketplace agent for business discovery and with individual business agents for negotiation and purchasing.

# Available Tools

These are your ONLY available actions:

* **search_businesses**: Send a search request to the centralized marketplace agent to obtain ranked businesses relevant to the customer's request.
* **send_messages**: Send text messages directly to business agents to ask questions, negotiate, or request additional information. Send payment messages to accept proposals.
* **check_messages**: Check for search results from the marketplace agent and responses, proposals, or confirmations from business agents.
* **no_purchase**: End the shopping process without making a purchase.
* **end_transaction**: Finish the transaction.

# Shopping Strategy

### 1. Understand

Carefully analyze the customer's request, including:

* requested products or services,
* quantities,
* budget,
* preferences,
* constraints,
* timing or delivery requirements.

### 2. Search

Send a search request to the centralized marketplace agent.

The marketplace agent will:

* retrieve relevant businesses,
* filter unsuitable businesses,
* rank businesses according to its search policy.

The marketplace does **not** negotiate or sell products.

### 3. Evaluate Search Results

Review the ranked businesses returned by the marketplace.

Use the rankings as recommendations rather than guarantees.

Select one or more promising businesses to contact directly.

### 4. Contact Businesses

Send messages directly to businesses to:

* verify availability,
* clarify missing information,
* negotiate when appropriate,
* request offers.

You may contact multiple businesses before making a decision.

### 5. Evaluate Proposals

When businesses send order proposals, compare them using:

* satisfaction of hard constraints,
* price,
* quality,
* quantity,
* availability,
* customer preferences,
* overall expected utility for the customer.

Do not automatically accept the first proposal received.

### 6. Decide

If a proposal clearly satisfies the customer's requirements and is preferable to not purchasing anything, send a payment message using the proposal's `message_id` as the `proposal_id`.

If, after reasonable search and communication, no available proposal matches the customer's request, choose 'no_purchase'.

### 7. Finish

Only call `end_transaction` after:

* a payment has succeeded, or
* the outside option has been selected.

# Important Notes

* 
* The marketplace agent performs **search and ranking only**.
* Individual business agents are responsible for answering questions, negotiating, creating proposals, and completing sales.
* Communicate directly with businesses after receiving search results.
* You may contact multiple businesses before deciding.
* Always check for responses after sending messages.
* Do not wait for the customer to make decisions—you are acting autonomously on their behalf.
* Do not purchase simply to complete the task. If and only if there are no proposals that match the customer's request, choose 'no_purchase'.""".strip()


    def format_state_context(self) -> tuple[str, int]:
        """Format the current state context for the agent.

        Returns:
            Formatted state context and integer step counter

        """
        # Format available proposals with IDs
        #         pending_proposals = self.proposal_storage.get_pending_proposals()
        #         proposals_text = ""
        #         if pending_proposals:
        #             proposals_text = "\nAvailable Proposals to Accept:\n"
        #             for proposal in pending_proposals:
        #                 proposals_text += f"  - Proposal ID: {proposal.proposal_id} from {proposal.business_id} (${proposal.proposal.total_price})\n"

        #         return f"""
        # Known Businesses: {len(self.known_business_ids)} businesses found
        # Received Proposals: {len(self.proposal_storage.proposals)} proposals
        # Completed Transactions: {len(self.completed_transactions)} transactions{proposals_text}
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

Send "text" messages to submit requirements to the market. The market will send "order_proposal" messages with offers. Send "pay" messages to accept proposals you want to purchase. When you receive an order_proposal message, use its message_id as the proposal_id in your payment. Always check for responses after sending messages. You must pay for proposals when you have sufficient information - do not wait for the customer. Only end the transaction after successfully paying for a proposal.

Choose your action carefully.
"""

    def format_event_history(self):
        """Format the event history for the prompt."""
        lines: list[str] = []
        step_number = 0

        for event in self.event_history:
            step_number += 1
            if isinstance(event, tuple):
                lines.extend(
                    self._format_customer_action_event(*event, step_number=step_number)
                )
            else:
                lines.extend(self._format_log_event(event, step_number=step_number))

        return "\n".join(lines).strip(), step_number

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
        self, action: CustomerAction, result: CustomerActionResult, step_number: int
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
        elif isinstance(result, ActionExecutionResult):
            lines.append(f"Failed to search businesses. {result.content}")
        else:
            lines.append("Failed to search businesses.")

        return lines

    def _format_customer_check_messages_event(
        self, action: CustomerAction, result: CustomerActionResult, step_number: int
    ) -> list[str]:
        lines = self._format_step_header(current_step=step_number)
        lines.append("Action: check_messages (checking for responses)")

        if isinstance(result, FetchMessagesResponse):
            message_count = len(result.messages)
            if message_count == 0:
                lines.append(f"Step {step_number} result: 📭 No new messages")
            else:
                formatted_results: list[str] = []
                # Add received messages to conversation
                for received_message in result.messages:
                    message_content = received_message.message
                    formatted_results.append(
                        f"📨 Received {message_content.type} from {received_message.from_agent_id}: "
                        f"{message_content.model_dump_json(exclude={'type', 'expiry_time'}, exclude_none=True)}"
                    )
                lines.append(f"Step {step_number} result: {formatted_results}")
        
        elif isinstance(result, SearchResultsMessage):
            lines.append(
                f"Step {step_number} result: 🔍 Search results "
                f"for '{result.query}' using {result.algorithm}"
            )
            lines.append(
                f"Returned {len(result.results)} of "
                f"{result.total_possible_results} matching businesses."
            )

            for ranked in result.results:
                business = ranked.business

                entry = (
                    f"  {ranked.rank}. {business.id} "
                    f"({business.business.name})"
                )

                if ranked.score is not None:
                    entry += f" score={ranked.score:.3f}"

                if ranked.rationale:
                    entry += f" — {ranked.rationale}"

                lines.append(entry)
            
        elif isinstance(result, ActionExecutionResult):
            lines.append(
                f"Step {step_number} result: Failed to fetch messages. {result.content}"
            )
        else:
            lines.append(f"Step {step_number} result: Failed to fetch messages.")

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
                send_message_result_lines.append("✅ Message sent successfully")
            else:
                send_message_result_lines.append(f"❌ Send failed: {error_message}")

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
                    "🎉 PAYMENT COMPLETED SUCCESSFULLY! Transaction accepted by platform. The purchase has been finalized."
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
