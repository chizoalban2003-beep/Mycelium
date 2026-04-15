"""Stage 93 — AgentComms: lightweight multi-agent messaging bus.

Enables multiple named agents to exchange structured messages through
a central in-process broker.  The broker is intentionally simple to
keep tests fast and dependency-free.

Classes
-------
Message
    An envelope carrying content from one agent to another (or broadcast).
AgentComms
    Central message broker.  Agents subscribe to topics and publish
    messages; the broker routes them to the appropriate inboxes.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """A communication message between agents.

    Attributes
    ----------
    sender : str
        Name of the sending agent.
    topic : str
        Logical channel / subject of the message.
    content : Any
        Payload.
    recipient : str
        Target agent name, or ``"*"`` for broadcast.
    timestamp : float
        Unix time when the message was created.
    """

    sender: str
    topic: str
    content: Any
    recipient: str = "*"
    timestamp: float = field(default_factory=time.time)


class AgentComms:
    """Central messaging broker for multi-agent communication.

    All state is in-process (no network I/O) so tests remain fast.

    Attributes
    ----------
    subscriptions_ : dict[str, list[str]]
        Mapping topic → list of subscriber agent names.
    inboxes_ : dict[str, list[Message]]
        Pending messages for each agent.
    log_ : list[Message]
        Full history of published messages.
    """

    def __init__(self) -> None:
        self.subscriptions_: Dict[str, List[str]] = defaultdict(list)
        self.inboxes_: Dict[str, List[Message]] = defaultdict(list)
        self.log_: List[Message] = []

    # ------------------------------------------------------------------
    def subscribe(self, agent: str, topic: str) -> None:
        """Subscribe *agent* to *topic*.

        Parameters
        ----------
        agent : str
            Agent identifier.
        topic : str
            Topic to subscribe to.
        """
        if agent not in self.subscriptions_[topic]:
            self.subscriptions_[topic].append(agent)

    def unsubscribe(self, agent: str, topic: str) -> None:
        """Unsubscribe *agent* from *topic*."""
        subs = self.subscriptions_.get(topic, [])
        if agent in subs:
            subs.remove(agent)

    # ------------------------------------------------------------------
    def publish(self, message: Message) -> int:
        """Publish *message* and route it to recipients.

        Parameters
        ----------
        message : Message
            The message to deliver.

        Returns
        -------
        int
            Number of inboxes the message was delivered to.
        """
        self.log_.append(message)
        delivered = 0

        if message.recipient != "*":
            # Direct message
            self.inboxes_[message.recipient].append(message)
            delivered = 1
        else:
            # Broadcast to topic subscribers
            for agent in self.subscriptions_.get(message.topic, []):
                if agent != message.sender:
                    self.inboxes_[agent].append(message)
                    delivered += 1

        return delivered

    # ------------------------------------------------------------------
    def receive(self, agent: str, topic: Optional[str] = None) -> List[Message]:
        """Drain *agent*'s inbox, optionally filtered by *topic*.

        Parameters
        ----------
        agent : str
            The receiving agent.
        topic : str, optional
            If provided, only messages on this topic are returned.

        Returns
        -------
        list[Message]
            Messages drained from the inbox.
        """
        inbox = self.inboxes_.get(agent, [])
        if topic:
            matching = [m for m in inbox if m.topic == topic]
            self.inboxes_[agent] = [m for m in inbox if m.topic != topic]
        else:
            matching = list(inbox)
            self.inboxes_[agent] = []
        return matching

    def pending(self, agent: str) -> int:
        """Return the number of unread messages for *agent*."""
        return len(self.inboxes_.get(agent, []))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AgentComms(topics={list(self.subscriptions_.keys())}, "
            f"messages_logged={len(self.log_)})"
        )
