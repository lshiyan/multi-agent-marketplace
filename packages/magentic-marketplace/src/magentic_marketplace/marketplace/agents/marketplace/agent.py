from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from ...actions import (
    OrderProposal,
    Payment,
    ReceivedMessage,
    Search,
    SearchAlgorithm,
    SearchResponse,
    TextMessage,
)

from ...llm.config import BaseLLMConfig
from ..base import BaseSimpleMarketplaceAgent
from magentic_marketplace.platform.shared.models import AgentProfile
from .models import MarketplaceAction, MarketplaceRequestSession
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
        profile: AgentProfile,
        base_url: str,
        llm_config: BaseLLMConfig | None = None,
        search_algorithm: str = "simple",
        search_bandwidth: int = 10,
        polling_interval: float = 2,
        max_steps: int | None = None,
    ):
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

    async def handle_customer_requests(self, customer_id: str, text: str):
        """Creates an async task for an incoming customer request."""
        request_id = uuid.uuid4().hex

        session = MarketplaceRequestSession(
            request_id=request_id,
            customer_id=customer_id,
            request_text=text,
            event_history=[],
            completed_transactions=[],
        )

        self.sessions[request_id] = session

        asyncio.create_task(self._run_session(request_id))

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
        if self._active_request is None:
            raise ValueError("Marketplace agent has no active request.")

        session = self.sessions[session_id]
        
        if session_id not in self.prompt_handlers:
            self.prompt_handlers = PromptsHandler(
            marketplace_agent_id=self.id,
            session=session,
            completed_transactions=self.completed_transactions,
            event_history=self._event_history,
            logger=self.logger,
        )
        
        return self.prompt_handlers[session_id]

    async def _generate_marketplace_action(self, session_id: str) -> MarketplaceAction | None:
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
                f"[Step {self.conversation_step}/{self._max_steps or 'inf'}] "
                f"Action: {action.action_type}. Reason: {action.reason}"
            )

            return action

        except Exception:
            self.logger.exception(
                f"[Step {self.conversation_step}/{self._max_steps or 'inf'}] "
                "LLM decision failed."
            )
            self._event_history.append(
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
            result = await self._inspect_business(action.business_id)
            session.add_event(action, result)

        elif action.action_type == "create_order_proposal":
            result = await self._create_order_proposal(action)
            session.add_event(action, result)

        elif action.action_type == "end_transaction":
            session.status="finished"

        else:
            self.logger.warning(
                f"Unknown marketplace action type: {action.action_type}"
            )
            self._event_history.append(
                f"Unknown marketplace action type: {action.action_type}"
            )

        return False

    async def _inspect_business(self, business_id: str):
        """Directly inspect a business.

        Replace this with the actual platform/database action once available.
        """
        

    async def _create_order_proposal(self, action: MarketplaceAction):
        """Create a proposal directly from marketplace business data.

        Replace this with your centralized proposal-generation logic.
        """
        raise NotImplementedError("create_order_proposal is not wired yet.")

    async def _reply_to_requester(self, action: MarketplaceAction):
        """Return a final response to whoever submitted the marketplace request.

        This replaces customer/business messaging.
        """
        raise NotImplementedError("reply_to_requester is not wired yet.")