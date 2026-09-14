"""Lexical search implementation for the simple marketplace."""

import logging
import math
import numpy as np

from magentic_marketplace.platform.database.base import BaseDatabaseController
from magentic_marketplace.platform.database.queries.agents import query as agent_query
from magentic_marketplace.platform.database.queries.base import RangeQueryParams

from ...actions import Search, SearchResponse
from .utils import convert_agent_rows_to_businesses

logger = logging.getLogger(__name__)


async def execute_lexical_search(
    search: Search,
    database: BaseDatabaseController,
    intervention: str
) -> SearchResponse:
    """Execute lexical search using shingle overlap ranking."""
    # Get all business agents
    business_filter = agent_query(path="$.business", value=None, operator="!=")
    all_agent_rows = await database.agents.find(business_filter, RangeQueryParams())
    
    # Convert to BusinessAgentProfile objects
    businesses = await convert_agent_rows_to_businesses(all_agent_rows)

    # Rating rank before lexical rank to help with:
    # 1. Tie breaking in lexical search
    # 2. Handle sort in no-query case
    businesses = sorted(
        businesses,
        key=lambda b: b.business.rating,
        reverse=True,
    )

    # Rank by lexical similarity if query provided
    if search.query:
        from .lexical_algo import lexical_rank

        businesses = lexical_rank(search.query, businesses)
        
    # Apply pagination and search limit
    paginated_businesses = businesses
    total_pages = 1

    if search.limit and search.limit > 0:
        start = (search.page - 1) * search.limit
        end = start + search.limit
        paginated_businesses = businesses[start:end]
        total_pages = math.ceil(len(businesses) / search.limit)

    if intervention == "logits":
        alpha = 2
        mu = 0.25

        weights = {}

        for business_agent in paginated_businesses:
            business = business_agent.business

            cur_menu = business.menu_features
            average_prices = sum(cur_menu.values()) / len(cur_menu)

            exponent = (alpha - average_prices) / mu
            weights[business.id] = exponent

        if weights:
            max_exponent = max(weights.values())

            exp_weights = {
                k: np.exp(v - max_exponent)
                for k, v in weights.items()
            }

            total = sum(exp_weights.values())

            weights = {
                k: v / total
                for k, v in exp_weights.items()
            }

            probs = np.array([
                weights[b.business.id]
                for b in paginated_businesses
            ])

            indices = np.random.choice(
                len(paginated_businesses),
                size=len(paginated_businesses),
                replace=False,
                p=probs,
            )

            paginated_businesses = [
                paginated_businesses[i]
                for i in indices
            ] 
            
    return SearchResponse(
        businesses=paginated_businesses,
        search_algorithm=search.search_algorithm,
        total_possible_results=len(businesses),
        total_pages=total_pages,
    )
