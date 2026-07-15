from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from collections import defaultdict

from ...actions import (
    OrderProposal,
    Payment,
    ReceivedMessage,
    Search,
    SearchAlgorithm,
    SearchResponse,
    TextMessage,
    InspectBusiness
)
from ..proposal_storage import OrderProposalStorage
from ...actions import OrderProposal, OrderItem

from ...llm.config import BaseLLMConfig
from ..base import BaseSimpleMarketplaceAgent
from magentic_marketplace.platform.shared.models import AgentProfile
from .models import MarketplaceAction, MarketplaceRequestSession, MarketplaceAgentProfile
from magentic_marketplace.platform.shared.models import (
    BaseAction,
)
from .prompts import PromptsHandler

import asyncio
import uuid
import traceback


MAX_ITERS = 6

class MarketplaceAgent(BaseSimpleMarketplaceAgent[AgentProfile]):
    """Centralized marketplace agent.

    This agent represents the marketplace itself. It does not act on behalf of
    one customer agent and does not communicate with business proxy agents.
    It directly searches and reasons over marketplace business resources.
    """

    def __init__(
        self,
        base_url: str,
        llm_config: BaseLLMConfig | None = None,
        search_algorithm: str = "simple",
        search_bandwidth: int = 10,
        polling_interval: float = 2,
        max_steps: int | None = None,
    ):
        profile = MarketplaceAgentProfile(id = "marketplace")
        super().__init__(profile, base_url, llm_config)

        self.conversation_step: int = 0
        self.completed_transactions: list[str] = []

        self.sessions: dict[str, MarketplaceRequestSession] = {}
        self.prompt_handlers: dict[str, PromptsHandler] = {}

        self._search_algorithm = SearchAlgorithm(search_algorithm)
        self._search_bandwidth = search_bandwidth
        self._polling_interval = polling_interval
        self._max_steps = max_steps

        self._active_request: str | None = None

    async def step(self):
        """One step of marketplace agent logic - check messages, then advance sessions."""
        messages = await self.fetch_messages()

        new_messages_by_customer: dict[str, list[ReceivedMessage]] = defaultdict(list)
        for received_message in messages.messages:
            new_messages_by_customer[received_message.from_agent_id].append(
                received_message
            )

        if new_messages_by_customer:
            await asyncio.gather(
                *[
                    self._handle_new_customer_messages(customer_id, msgs)
                    for customer_id, msgs in new_messages_by_customer.items()
                ]
            )

        # 2. Advance any active sessions by one LLM-decided action
        active_sessions = [s for s in self.sessions.values() if s.status == "active"]
        if active_sessions:
            await asyncio.gather(
                *[self._step_session(s.request_id) for s in active_sessions]
            )

        # 3. Backoff only if nothing happened at all
        if not new_messages_by_customer and not active_sessions:
            await asyncio.sleep(self._polling_interval)
        else:
            await asyncio.sleep(0)

    async def _handle_new_customer_messages(
        self, customer_id: str, new_messages: list[ReceivedMessage]
    ):
        """Routes incoming messages to the right session, creating one if needed."""
        for received_message in new_messages:
            message = received_message.message

            if isinstance(message, Payment):
                session = self._find_session_for_payment(customer_id, message)
                if session is None:
                    self.logger.error(
                        f"Received payment from {customer_id} with no matching session"
                    )
                    continue
                response = await self.handle_customer_payment(
                    session.request_id, message
                )
                await self.send_message(customer_id, response)

            elif isinstance(message, TextMessage):
                session = self._find_active_session_for_customer(customer_id)
                if session is None:
                    request_id = uuid.uuid4().hex
                    session = MarketplaceRequestSession(
                        request_id=request_id,
                        customer_id=customer_id,
                        request_text=message.content
                    )
                    self.sessions[request_id] = session
                else:
                    session.event_history.append(f"Customer: {message.content}")

            else:
                self.logger.warning(
                    f"Ignoring unsupported message type from {customer_id}: {type(message)}"
                )

    def _find_active_session_for_customer(
        self, customer_id: str
    ) -> MarketplaceRequestSession | None:
        for session in self.sessions.values():
            if session.customer_id == customer_id and session.status == "active":
                return session
        return None

    def _find_session_for_payment(
        self, customer_id: str, payment: Payment
    ) -> MarketplaceRequestSession | None:
        for session in self.sessions.values():
            if (
                session.customer_id == customer_id
                and payment.proposal_message_id in session.pending_proposal_ids
            ):
                return session
        return None

    async def _step_session(self, session_id: str):
        """Advances a single session by one LLM-decided action."""
        session = self.sessions[session_id]
        session.step += 1

        action = await self._generate_marketplace_action(session_id)
        if action is None:
            return

        await self._execute_marketplace_action(session, action)

        if self._max_steps is not None and session.step >= self._max_steps:
            session.status = "failed"
            
    async def execute_action(self, action: BaseAction):
        """Execute an action through the marketplace platform."""
        return await super().execute_action(action)

    async def _run_session(self, session_id: str):
        """Runs the LLM concurrently on a given session_id."""
        session = self.sessions[session_id]
        while session.status == "active":
            session.step += 1

            action = await self._generate_marketplace_action(session_id)
            if action is None:
                await asyncio.sleep(self._polling_interval)
                continue

            await self._execute_marketplace_action(session, action)

            if self._max_steps is not None and session.step >= self._max_steps:
                session.status = "failed"
                break

            await asyncio.sleep(self._polling_interval)

    async def on_started(self):
        self.logger.info("Starting centralized marketplace agent.")

    def _get_prompts_handler(self, session_id: str) -> PromptsHandler:
        session = self.sessions[session_id]
        
        if session_id not in self.prompt_handlers:
            self.prompt_handlers[session_id] = PromptsHandler(
            marketplace_agent_id=self.id,
            session=session,
            logger=self.logger,
        )
        
        return self.prompt_handlers[session_id]

    async def _generate_marketplace_action(self, session_id: str) -> MarketplaceAction | None:
        session = self.sessions[session_id]
        prompts = self._get_prompts_handler(session_id)

        system_prompt = prompts.format_system_prompt().strip()
        state_context, step_counter = prompts.format_state_context()
        step_prompt = prompts.format_step_prompt(step_counter).strip()

        full_prompt = f"{system_prompt}\n\n\n\n{state_context.strip()}\n\n{step_prompt}"

        try:
            action, _ = await self.generate_struct(
                prompt=full_prompt,
                response_format=MarketplaceAction,
            )

            self.logger.info(
                f"[Step {session.step}/{self._max_steps or 'inf'}] "
                f"Action: {action.action_type}. Reason: {action.reason}"
            )

            return action

        except Exception:
            self.logger.exception(
                f"[Step {self.conversation_step}/{self._max_steps or 'inf'}] "
                "LLM decision failed."
            )
            session.event_history.append(
                f"LLM decision failed: {traceback.format_exc()}"
            )
            return None

    async def _execute_marketplace_action(
        self,
        session: MarketplaceRequestSession,
        action: MarketplaceAction,
    ) -> bool:
        """Execute the LLM-selected marketplace action."""

        if action.action_type == "search_businesses":
            search_action = Search(
                query=action.search_query or self._active_request or "",
                search_algorithm=self._search_algorithm,
                limit=self._search_bandwidth,
                page=action.search_page,
            )

            search_result = await self.execute_action(search_action)

            if not search_result.is_error:
                search_response = SearchResponse.model_validate(search_result.content)

                business_names = [
                    business.business.name
                    for business in search_response.businesses
                ]

                self.logger.info(
                    f'Search: "{search_action.query}", '
                    f"{search_action.search_algorithm}, "
                    f"found {len(search_response.businesses)} business(es) "
                    f"out of {search_response.total_possible_results}. "
                    f"Showing page {action.search_page} of "
                    f"{search_response.total_pages}."
                )

                self.logger.info(f"Search result: {', '.join(business_names)}")

                session.add_event(action, search_result)

            else:
                session.add_event(action, search_result)

        elif action.action_type == "inspect_business":
            inspect_action = InspectBusiness(business = action.business_id)
            result = await self.execute_action(inspect_action)
            session.add_event(action, result)

        elif action.action_type == "create_order_proposal":
            proposal = await self._create_order_proposal(session, action)
            session.add_event(action, proposal)

        elif action.action_type == "end_transaction":
            session.status="finished"

        else:
            self.logger.warning(
                f"Unknown marketplace action type: {action.action_type}"
            )
            session.event_history.append(
                f"Unknown marketplace action type: {action.action_type}"
            )

        return False
    
    async def _create_order_proposal(
        self,
        session: MarketplaceRequestSession,
        action: MarketplaceAction,
    ) -> OrderProposal:
        """Creates and then sends an order proposal to the customer."""
        proposal = OrderProposal(
            id=uuid.uuid4().hex,
            items=action.order_proposal_message.items,
            total_price=action.order_proposal_message.total_price,
        )

        session.proposal_storage.add_proposal(
            proposal=proposal,
            business_id=action.business_id,
            customer_id=session.customer_id,
        )

        session.pending_proposal_ids.append(proposal.id)

        await self.send_message(
            session.customer_id,
            proposal,
        )

        return proposal

    async def handle_customer_payment(
        self,
        session_id: str,
        payment: Payment,
    ) -> TextMessage:
        session = self.sessions[session_id]
        proposal_id = payment.proposal_message_id

        stored = session.proposal_storage.get_proposal(proposal_id)

        if stored is None:
            return TextMessage(
                content=f"No proposal found with id {proposal_id}."
            )

        if stored.status != "pending":
            return TextMessage(
                content=f"Proposal {proposal_id} is already {stored.status}."
            )

        session.proposal_storage.update_proposal_status(
            proposal_id,
            "accepted",
        )

        session.completed_transactions.append(proposal_id)
        self.completed_transactions.append(proposal_id)
        session.status = "finished"

        return TextMessage(
            content=f"Payment confirmed for proposal {proposal_id}."
        )