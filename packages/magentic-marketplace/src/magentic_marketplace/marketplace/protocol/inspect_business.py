
import logging
import math

from magentic_marketplace.platform.database.base import BaseDatabaseController
from magentic_marketplace.platform.database.queries.agents import query as agent_query
from magentic_marketplace.platform.database.queries.base import RangeQueryParams

from ..actions import InspectBusiness, InspectBusinessResponse
from .search.utils import convert_agent_rows_to_businesses

async def execute_inspect(
    action: InspectBusiness,
    database: BaseDatabaseController,
) -> InspectBusinessResponse:
    """Inspects a singular business based off of id."""
    # Get all business agents
    business_filter = agent_query(path="$.business", value=None, operator="!=")
    all_agent_rows = await database.agents.find(business_filter, RangeQueryParams())

    # Convert to BusinessAgentProfile objects
    businesses = await convert_agent_rows_to_businesses(all_agent_rows)

    # Rating rank before lexical rank to help with:
    # 1. Tie breaking in lexical search
    # 2. Handle sort in no-query case
    businesses = filter(
        businesses,
        key=lambda b: b.business.id == action.business_idS
    )

    result = None
    
    for b in businesses: 
        result = b
        break
    
    return InspectBusinessResponse(
        business = result
    )