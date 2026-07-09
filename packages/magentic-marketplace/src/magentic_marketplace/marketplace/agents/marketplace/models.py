from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from magentic_marketplace.platform.shared.models import ActionExecutionResult
from ...actions.actions import SearchResponse, InspectBusinessResponse
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

        return self
    
MarketplaceActionResult = (
    ActionExecutionResult
    | SearchResponse
    | InspectBusinessResponse
)

class MarketplaceRequestSession:
    request_id: str
    customer_id: str
    request_text: str
    event_history: list[tuple[MarketplaceAction, MarketplaceActionResult] | str]
    completed_transactions: list[str]
    step: int = 0
    status: Literal["active", "completed", "failed"] = "active"
    
    def add_event(self, action: MarketplaceAction, result: MarketplaceActionResult):
        self.event_history.append(action, result)