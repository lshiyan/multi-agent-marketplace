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
from .models import MarketplaceAgentProfile
from magentic_marketplace.platform.shared.models import (
    BaseAction,
)

import asyncio

from ...actions import Search, SearchResponse
from ...actions.messaging import (
    RankedBusiness,
    SearchRequestMessage,
    SearchResultsMessage,
)


class MarketplaceAgent(BaseSimpleMarketplaceAgent[AgentProfile]):
    """Centralized search and ranking agent."""

    def __init__(
        self,
        base_url: str,
        llm_config: BaseLLMConfig | None = None,
        search_algorithm: str = "simple",
        search_bandwidth: int = 10,
        candidate_bandwidth: int = 30,
        polling_interval: float = 2,
    ):
        profile = MarketplaceAgentProfile(id="marketplace")
        super().__init__(profile, base_url, llm_config)

        self._search_algorithm = SearchAlgorithm(search_algorithm)
        self._search_bandwidth = search_bandwidth
        self._candidate_bandwidth = candidate_bandwidth
        self._polling_interval = polling_interval

    async def on_started(self):
        """Handle when the marketplace agent starts."""
        self.logger.info("Starting marketplace agent.")
        
    async def step(self) -> None:
        fetch_response = await self.fetch_messages()

        requests = [
            message
            for message in fetch_response.messages
            if isinstance(message.message, SearchRequestMessage)
        ]

        if not requests:
            await asyncio.sleep(self._polling_interval)
            return

        await asyncio.gather(
            *[
                self._handle_search_request(
                    customer_id=received.from_agent_id,
                    request=received.message,
                )
                for received in requests
            ]
        )

    async def _handle_search_request(
        self,
        customer_id: str,
        request: SearchRequestMessage,
    ) -> None:
        search = Search(
            query=request.query,
            search_algorithm=self._search_algorithm,
            limit=self._candidate_bandwidth,
            page=1,
        )

        result = await self.execute_action(search)

        if result.is_error:
            await self.send_message(
                customer_id,
                TextMessage(
                    content=f"Search failed: {result.content}"
                ),
            )
            return

        response = SearchResponse.model_validate(result.content)

        selected = response.businesses[: request.limit]

        ranked_results = [
            RankedBusiness(
                business=business,
                rank=rank,
            )
            for rank, business in enumerate(selected, start=1)
        ]

        await self.send_message(
            customer_id,
            SearchResultsMessage(
                request_id=request.request_id,
                query=request.query,
                algorithm=self._search_algorithm.value,
                results=ranked_results,
                total_possible_results=response.total_possible_results,
                total_pages=response.total_pages,
            ),
        )