"""MessageBus — Agent间消息通信。

Maps to agent_os_initial_plan.md §6.3 (SEND_MESSAGE), §11.1 (Message Bus).
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Message:
    message_id: str
    from_agent: str
    to_agent: str
    payload: dict = field(default_factory=dict)
    message_type: str = "task_result"
    timestamp: str = ""


class MessageBus:
    """In-memory message queue for agent-to-agent communication."""

    def __init__(self):
        self._queues: dict[str, list[Message]] = {}

    def send(self, msg: Message) -> None:
        if not msg.timestamp:
            msg.timestamp = datetime.now(timezone.utc).isoformat()
        self._queues.setdefault(msg.to_agent, []).append(msg)

    def receive(self, agent_id: str) -> list[Message]:
        return self._queues.get(agent_id, [])

    def broadcast(self, from_agent: str, payload: dict, msg_type: str = "notification") -> None:
        for agent_id in self._queues:
            if agent_id != from_agent:
                self.send(Message(
                    message_id=f"msg_{uuid.uuid4().hex[:12]}",
                    from_agent=from_agent, to_agent=agent_id,
                    payload=payload, message_type=msg_type,
                ))

    def clear(self, agent_id: str) -> None:
        self._queues.pop(agent_id, None)
