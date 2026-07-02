from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types.a2a_pb2 import TaskState
from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)

from .marketplace_agent import MarketplaceAgent

class MarketplaceAgentExecutor(AgentExecutor):
    
    def __init__(self) -> None:
        self.agent = MarketplaceAgent()
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        
        if context.current_task:
            task = context.current_task
        
        else:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
    
        task_updater = TaskUpdater(
            event_queue=event_queue, task_id=task.id, context_id=task.context_id
        )
        
        await task_updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=new_text_message('Processing request...'),
        )
        
        query = get_message_text(context.message)
        skill_id = context.skill_id
        
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        return