from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from magentic_marketplace.platform.shared.models import ActionExecutionResult
from magentic_marketplace.platform.shared.models import AgentProfile
from ...actions import SearchResponse, InspectBusinessResponse, OrderProposal
from ..proposal_storage import OrderProposalStorage

class MarketplaceAction(BaseModel):
    """Actions the Assistant can take.

    Use:
        - search_businesses to search for businesses.
        - send_messages to send messages to some businesses.
        - check_messages to check for new responses from businesses.
        - end_transaction if you have paid for an order or received confirmation.

    Do not end if you haven't completed a purchase transaction.
    """

    action_type: Literal[
        "search_businesses", "inspect_business", "create_order_proposal", "end_transaction"
    ] = Field(description="Type of action to take")
    reason: str = Field(description="Reason for taking this action")

    business_id: str = Field(description="Id of business to inspect.")
    # Search-specific fields
    search_query: str | None = Field(
        default=None,
        description="Search query for businesses.",
    )
    
    search_page: int = Field(
        default=1,
        description="Page number to retrieve for the search results (default: 1)",
    )

    @model_validator(mode="after")
    def validate_model(self):
        """Validate the BaseModel structure."""
        if self.action_type == "search_businesses":
            if not self.search_query:
                raise ValueError(
                    "search_query is required when action_type is search_businesses"
                )
        if self.action_type == "inspect_business":
            if not self.business_id: 
                raise ValueError(
                    "business_id is required when action_type is inspect_business"
                )

        return self
    
class MarketplaceAgentProfile(AgentProfile):
    """Profile for the centralized marketplace agent.

    The centralized marketplace agent represents the marketplace platform
    and handles customer requests by directly accessing businesses and
    marketplace resources.
    """

    name: str = "Centralized Marketplace Agent"
    description: str = (
        "Centralized marketplace agent that receives customer requests "
        "and directly searches, evaluates, and coordinates marketplace businesses."
    )


MarketplaceActionResult = (
    ActionExecutionResult
    | SearchResponse
    | InspectBusinessResponse
    | OrderProposal
)

class MarketplaceRequestSession:
    def __init__(
        self,
        request_id: str,
        customer_id: str,
        request_text: str,
    ):
        self.request_id = request_id
        self.customer_id = customer_id
        self.request_text = request_text
        self.event_history: list[
            tuple[MarketplaceAction, MarketplaceActionResult] | str
        ] = []
        self.completed_transactions: list[str] = []
        self.step = 0
        self.status: Literal["active", "completed", "failed"] = "active"
        self.proposal_storage = OrderProposalStorage()
        self.pending_proposal_ids: list[str] = []

    def add_event(
        self,
        action: MarketplaceAction,
        result: MarketplaceActionResult,
    ):
        self.event_history.append((action, result))