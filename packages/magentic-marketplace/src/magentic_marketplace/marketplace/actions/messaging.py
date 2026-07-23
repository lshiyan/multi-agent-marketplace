"""Messaging actions for the simple marketplace."""

from typing import Annotated, Literal
from uuid import uuid4
from pydantic import BaseModel, Field
from pydantic.type_adapter import TypeAdapter

from ..shared.models import BusinessAgentProfile, SearchConstraints

class OrderItem(BaseModel):
    """An item in an order with quantity and pricing."""

    id: str = Field(description="Menu item ID from the business")
    item_name: str = Field(description="Name of the item")
    quantity: int = Field(description="Quantity ordered", ge=1)
    unit_price: float = Field(description="Price per unit", ge=0)


class TextMessage(BaseModel):
    """A text message."""

    type: Literal["text"] = "text"
    content: str = Field(description="Text content of the message")


class OrderProposal(BaseModel):
    """Order proposal details sent by service agents to customers."""

    type: Literal["order_proposal"] = "order_proposal"
    id: str = Field(description="The unique id of this proposal", min_length=1)
    items: list[OrderItem] = Field(
        min_length=1,
        description="Required; the list of OrderItem objects with item_name, quantity, and unit_price",
    )
    total_price: float = Field(description="Required; total price for the entire order")


class SearchRequestMessage(BaseModel):
    type: Literal["search_request"] = "search_request"

    request_id: str = Field(
        default_factory=lambda: uuid4().hex
    )
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1)
    page: int = Field(default=1, ge=1)


class RankedBusiness(BaseModel):
    business: BusinessAgentProfile
    rank: int = Field(ge=1)

    # Optional when using deterministic ranking.
    score: float | None = None
    rationale: str | None = None


class SearchResultsMessage(BaseModel):
    type: Literal["search_results"] = "search_results"

    request_id: str
    query: str
    algorithm: str
    results: list[RankedBusiness]
    total_possible_results: int
    total_pages: int
    
class Payment(BaseModel):
    """A payment message to accept an order proposal."""

    type: Literal["payment"] = "payment"
    proposal_message_id: str = Field(
        description="ID of the message containing the order proposal to accept"
    )
    payment_method: str | None = Field(
        default=None,
        description="Payment method to use (e.g., 'credit_card', 'cash', 'digital_wallet')",
    )
    delivery_address: str | None = Field(
        default=None, description="Delivery address if different from customer profile"
    )
    payment_message: str | None = Field(
        default=None, description="Additional message to include with the payment"
    )


# Message is a union type of the message types
Message = Annotated[TextMessage | OrderProposal | Payment | SearchRequestMessage | SearchResultsMessage , Field(discriminator="type")]

# Type adapter for Message for serialization/deserialization
MessageAdapter: TypeAdapter[Message] = TypeAdapter(Message)
