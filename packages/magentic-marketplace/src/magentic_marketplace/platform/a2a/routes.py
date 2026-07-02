
from a2a.server.request_handlers import DefaultRequestHandler

from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)

from .marketplace_agent_executor import MarketplaceAgentExecutor

def get_agent_card() -> AgentCard: 
    search_skill = AgentSkill(
            id='search',
            name='Business Search',
            description='Search businesses for relevant businesses.',
            input_modes=['text/plain'],
            output_modes=['text/plain'],
            tags=['a2a', 'search'],
            examples=['I want a business that serves Pineapple Aqua Fresca and takes reservations.'],
        )

    agent_card = AgentCard(
            name='Marketplace Agent',
            description='Centralized Marketplace Agent',
            version='0.0.1',
            default_input_modes=['text/plain'],
            default_output_modes=['text/plain'],
            capabilities=AgentCapabilities(streaming=True),
            supported_interfaces=[
                AgentInterface(
                    protocol_binding='JSONRPC',
                    url='http://127.0.0.1:9999',
                )
            ],
            skills=[search_skill],
        )
    
    return agent_card

def get_request_handler() -> DefaultRequestHandler:
    
    agent_card = get_agent_card()
    
    request_handler = DefaultRequestHandler(
        agent_executor=MarketplaceAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    
    return request_handler