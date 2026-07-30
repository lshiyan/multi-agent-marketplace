"""Main business agent implementation."""

import asyncio
import math
from collections import defaultdict
from typing import Literal

from ...actions import (
    Message,
    OrderProposal,
    Payment,
    ReceivedMessage,
    TextMessage,
)
from ...llm.config import BaseLLMConfig
from ...shared.models import Business, BusinessAgentProfile
from ..base import BaseSimpleMarketplaceAgent
from ..proposal_storage import OrderProposalStorage
from .models import (
    BusinessPriceUpdate,
    BusinessSummary,
    RequestOutcome,
    MenuItemPriceUpdate
)
from .responses import ResponseHandler
from ..proposal_storage import StoredOrderProposal


class BusinessAgent(BaseSimpleMarketplaceAgent[BusinessAgentProfile]):
    """Business agent that responds to customer inquiries and creates proposals."""

    def __init__(
        self,
        business: Business,
        base_url: str,
        llm_config: BaseLLMConfig | None = None,
        polling_interval: float = 2,
    ):
        """Initialize the business agent.

        Args:
            business: Business object with menu and capabilities
            base_url: The marketplace server URL
            llm_config: LLM configuration for the agent
            polling_interval: Number of seconds to wait after fetching 0 messages.

        """
        profile = BusinessAgentProfile.from_business(business)
        super().__init__(profile, base_url, llm_config)

        # Initialize state from BaseBusinessAgent
        self.customer_histories: dict[str, list[str]] = defaultdict(list)
        self.proposal_storage = OrderProposalStorage()
        
        self.confirmed_orders: list[str] = []

        self._completed_payment_cursor = 0
        self.confirmed_orders: list[str] = []
        self._polling_interval = polling_interval

        # Initialize handler resources
        self._responses = ResponseHandler(
            business=self.business,
            agent_id=self.id,
            proposal_storage=self.proposal_storage,
            logger=self.logger,
            generate_struct_fn=self.generate_struct,
        )
        
        self.current_prices: dict[str, float] = dict(
            business.menu_features
        )

        self.minimum_prices: dict[str, float] = {
            item_name: original_price * business.min_price_factor
            for item_name, original_price
            in business.menu_features.items()
        }

    @property
    def business(self) -> Business:
        """Access business data from profile with full type safety."""
        return self.profile.business

    async def _handle_new_customer_messages(
        self, customer_id: str, new_messages: list[ReceivedMessage]
    ):
        messages_to_send: list[tuple[str, Message]] = []

        # First, handle all payments to update proposal statuses
        last_text_message: TextMessage | None = None

        for received_message in new_messages:
            if isinstance(received_message.message, Payment):
                response = await self._handle_payment(
                    customer_id, received_message.message
                )
                messages_to_send.append((customer_id, response))

            elif isinstance(received_message.message, TextMessage):
                last_text_message = received_message.message

            self.add_to_history(
                customer_id,
                received_message.message,
                "customer",
            )

        # Generate a text response if there are any text messages
        if last_text_message is not None:
            response_message = await self._responses.generate_response_to_inquiry(
                customer_id, self.customer_histories[customer_id]
            )
            messages_to_send.append((customer_id, response_message))

        # Send all generated responses
        for customer_id, message in messages_to_send:
            # If this is a proposal, store it in proposal storage
            if isinstance(message, OrderProposal):
                self.proposal_storage.add_proposal(message, self.id, customer_id)

            # FUTURE -- add retries on message fail.
            try:
                result = await self.send_message(customer_id, message)
                if result.is_error:
                    error_msg = f"Error: Failed to send message to {customer_id}: {result.content}"
                    self.logger.error(error_msg)
                    self.add_to_history(customer_id, error_msg, "business")

                else:
                    self.add_to_history(customer_id, message, "business")
            except Exception as e:
                error_msg = f"Error: Failed to send message to {customer_id}: {e}"
                self.logger.exception(error_msg)
                self.add_to_history(customer_id, error_msg, "business")

    def add_to_history(
        self,
        customer_id: str,
        message: Message | str,
        customer_or_agent: Literal["customer", "business"],
    ):
        """Add a message to the customer's history.

        Args:
            customer_id: ID of the customer
            message: The message to add
            customer_or_agent: Whether the message is from the customer or the agent

        """
        prefix = "Customer" if customer_or_agent == "customer" else "You"
        formatted_message = None

        if isinstance(message, str):
            formatted_message = f"{prefix}: {message}"
        elif isinstance(message, TextMessage):
            formatted_message = f"{prefix}: {message.content}"
        elif isinstance(message, Payment):
            formatted_message = f"{prefix}: {message.model_dump(exclude_none=True)}"
        elif isinstance(message, OrderProposal):
            formatted_message = f"{prefix}: {message.model_dump(exclude_none=True)}"
        else:
            self.logger.warning(
                "Ignoring message in Business add_to_history: ", message
            )

        if formatted_message is not None:
            self.customer_histories[customer_id].append(formatted_message)

    async def step(self):
        """One step of business agent logic - check for and handle customer messages."""
        # Check for new messages
        messages = await self.fetch_messages()

        # Group new messages by customer
        new_messages_by_customer: dict[str, list[ReceivedMessage]] = defaultdict(list)

        for received_message in messages.messages:
            new_messages_by_customer[received_message.from_agent_id].append(
                received_message
            )

        if new_messages_by_customer:
            await asyncio.gather(
                *[
                    self._handle_new_customer_messages(customer_id, new_messages)
                    for customer_id, new_messages in new_messages_by_customer.items()
                ]
            )

        if len(new_messages_by_customer) == 0:
            # Wait before next check
            await asyncio.sleep(self._polling_interval)
        else:
            await asyncio.sleep(0)

    async def on_started(self):
        """Handle when the business agent starts."""
        self.logger.info("Ready for customers")

    async def _handle_payment(self, customer_id: str, payment: Payment) -> TextMessage:
        """Handle a payment from a customer.

        Args:
            customer_id: ID of the customer
            payment: The payment message

        Returns:
            Message to send back to customer

        """
        proposal_id = payment.proposal_message_id
        self.logger.info(
            f"Processing payment for proposal {proposal_id} from customer {customer_id}"
        )

        stored_proposal = self.proposal_storage.get_proposal(proposal_id)
        if stored_proposal and stored_proposal.status == "pending":
            # Accept the payment
            self.proposal_storage.update_proposal_status(proposal_id, "accepted")
            self.confirmed_orders.append(proposal_id)

            # Generate confirmation using ResponseHandler
            confirmation = self._responses.generate_payment_confirmation(
                proposal_id, stored_proposal.proposal.total_price
            )
            self.logger.info(
                f"Confirmed payment for proposal {proposal_id} from customer {customer_id}"
            )
            return confirmation
        else:
            if stored_proposal:
                self.logger.error(
                    f"Failed to process payment for proposal {proposal_id} from customer {customer_id}. Proposal status is not pending: {stored_proposal.status}."
                )
            else:
                self.logger.error(
                    f"Failed to process payment for proposal {proposal_id} from customer {customer_id}. No proposals match that id."
                )

            # Generate error message using ResponseHandler
            error_message = self._responses.generate_proposal_not_found_error(
                proposal_id
            )
            return error_message

    def get_business_summary(self) -> BusinessSummary:
        """Get a summary of business operations.

        Returns:
            Summary of business state and transactions

        """
        return BusinessSummary(
            business_id=self.business.id,
            business_name=self.business.name,
            description=self.business.description,
            rating=self.business.rating,
            menu_items=len(self.business.menu_features),
            amenities=len(self.business.amenity_features),
            pending_proposals=self.proposal_storage.count_pending_proposals(),
            confirmed_orders=len(self.confirmed_orders),
            delivery_available=self.business.amenity_features.get("delivery", False),
        )

    async def update_prices(
    self,
    request_outcomes: list[RequestOutcome],
) -> BusinessPriceUpdate | None:
        """Update prices after one experiment run.

        Only requests for which this business was contacted are used.
        """
        relevant_outcomes = [
            outcome
            for outcome in request_outcomes
            if any(
                contacted.business_id == self.id
                for contacted in outcome.contacted_businesses
            )
        ]

        prompt = self._responses.prompts.format_update_prompt(
            request_outcomes=relevant_outcomes,
            current_prices=self.current_prices,
            minimum_prices=self.minimum_prices,
        )

        update, _ = await self.generate_struct(
            prompt=prompt,
            response_format=BusinessPriceUpdate,
        )

        proposed_prices: dict[str, float] = {}

        for item_update in update.price_updates:
            if item_update.item_name in proposed_prices:
                self.logger.warning(
                    "Duplicate price update returned for "
                    f"{item_update.item_name}; using the final value."
                )

            proposed_prices[item_update.item_name] = item_update.price

        unknown_items = (
            set(proposed_prices)
            - set(self.current_prices)
        )

        if unknown_items:
            self.logger.warning(
                "Ignoring prices for unknown menu items: "
                f"{sorted(unknown_items)}"
            )

        validated_prices: dict[str, float] = {}

        for item_name, current_price in self.current_prices.items():
            proposed_price = proposed_prices.get(
                item_name,
                current_price,
            )

            if not math.isfinite(proposed_price):
                self.logger.warning(
                    f"Invalid price returned for {item_name}: "
                    f"{proposed_price}. Keeping current price "
                    f"${current_price:.2f}."
                )
                proposed_price = current_price

            validated_prices[item_name] = max(
                self.minimum_prices[item_name],
                proposed_price,
            )

        self.current_prices = validated_prices
        self.business.menu_features.update(validated_prices)

        validated_update = BusinessPriceUpdate(
            price_updates=[
                MenuItemPriceUpdate(
                    item_name=item_name,
                    price=price,
                )
                for item_name, price in validated_prices.items()
            ],
            reasoning=update.reasoning,
        )
        
        self.logger.info(
            f"Updated prices: {validated_prices}. "
            f"Reasoning: {update.reasoning}"
        )

        return validated_update
    
    def consume_completed_payments(self) -> list[StoredOrderProposal]:
        """Return completed payments since the previous business update.

        Calling this method marks those payments as consumed by the update process.
        """
        new_proposal_ids = self.confirmed_orders[
            self._completed_payment_cursor:
        ]

        completed_payments: list[StoredOrderProposal] = []

        for proposal_id in new_proposal_ids:
            stored_proposal = self.proposal_storage.get_proposal(proposal_id)

            if stored_proposal is None:
                self.logger.warning(
                    f"Completed proposal {proposal_id} was not found in storage."
                )
                continue

            if stored_proposal.status != "accepted":
                self.logger.warning(
                    f"Completed proposal {proposal_id} has unexpected "
                    f"status {stored_proposal.status}."
                )
                continue

            completed_payments.append(stored_proposal)

        self._completed_payment_cursor = len(self.confirmed_orders)

        return completed_payments
