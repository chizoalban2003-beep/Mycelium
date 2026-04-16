"""Stage 57 — KnowledgeGraph: lightweight directed knowledge graph for agent reasoning.

Stores typed facts (nodes) and labelled relationships (edges).  Useful for
keeping track of domain concepts, feature relationships, and learned causal
links the agent discovers during its lifetime.

Key classes
-----------
* :class:`KnowledgeNode` — a named node with a type tag and optional payload.
* :class:`KnowledgeGraph` — directed graph with add/query/path-finding API.

Usage
-----
::

    from physml.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    kg.add_node("temperature", node_type="feature")
    kg.add_node("ice_melts",   node_type="event")
    kg.add_edge("temperature", "ice_melts", relation="causes", weight=0.9)

    print(kg.neighbors("temperature"))   # ["ice_melts"]
    print(kg.path("temperature", "ice_melts"))  # ["temperature", "ice_melts"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeNode:
    """A single node in the knowledge graph.

    Attributes
    ----------
    name : str
        Unique identifier for the node.
    node_type : str
        Semantic category (e.g. "feature", "concept", "event").
    payload : dict
        Arbitrary key-value metadata attached to the node.
    """

    name: str
    node_type: str = "concept"
    payload: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:  # needed for sets/dicts
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, KnowledgeNode):
            return self.name == other.name
        return NotImplemented


@dataclass
class _Edge:
    source: str
    target: str
    relation: str = "related_to"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """Directed knowledge graph with typed nodes and labelled edges.

    Parameters
    ----------
    directed : bool, default True
        When *False*, edges are treated as undirected (stored in both
        directions automatically).
    """

    def __init__(self, directed: bool = True) -> None:
        self.directed = directed
        self._nodes: dict[str, KnowledgeNode] = {}
        self._adj: dict[str, list[_Edge]] = {}   # source → list of edges

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        node_type: str = "concept",
        **payload: Any,
    ) -> KnowledgeNode:
        """Add or update a node.  Returns the (possibly new) node."""
        if name not in self._nodes:
            self._nodes[name] = KnowledgeNode(name=name, node_type=node_type, payload=payload)
            self._adj[name] = []
        else:
            existing = self._nodes[name]
            existing.node_type = node_type
            existing.payload.update(payload)
        return self._nodes[name]

    def remove_node(self, name: str) -> None:
        """Remove a node and all edges involving it."""
        if name not in self._nodes:
            return
        del self._nodes[name]
        del self._adj[name]
        for edges in self._adj.values():
            edges[:] = [e for e in edges if e.target != name]

    def has_node(self, name: str) -> bool:
        return name in self._nodes

    def get_node(self, name: str) -> KnowledgeNode | None:
        return self._nodes.get(name)

    def node_count(self) -> int:
        return len(self._nodes)

    def nodes(self) -> list[KnowledgeNode]:
        return list(self._nodes.values())

    def nodes_by_type(self, node_type: str) -> list[KnowledgeNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str = "related_to",
        weight: float = 1.0,
        **metadata: Any,
    ) -> None:
        """Add a directed edge (auto-creating nodes if missing)."""
        for name in (source, target):
            if name not in self._nodes:
                self.add_node(name)

        edge = _Edge(source=source, target=target, relation=relation,
                     weight=weight, metadata=metadata)
        self._adj[source].append(edge)
        if not self.directed:
            rev = _Edge(source=target, target=source, relation=relation,
                        weight=weight, metadata=metadata)
            self._adj[target].append(rev)

    def remove_edge(self, source: str, target: str, relation: str | None = None) -> int:
        """Remove matching edges.  Returns number of edges removed."""
        if source not in self._adj:
            return 0
        before = len(self._adj[source])
        if relation is None:
            self._adj[source] = [e for e in self._adj[source] if e.target != target]
        else:
            self._adj[source] = [
                e for e in self._adj[source]
                if not (e.target == target and e.relation == relation)
            ]
        removed = before - len(self._adj[source])
        if not self.directed and removed:
            self._adj[target] = [
                e for e in self._adj[target]
                if e.target != source
            ]
        return removed

    def has_edge(self, source: str, target: str, relation: str | None = None) -> bool:
        if source not in self._adj:
            return False
        for e in self._adj[source]:
            if e.target == target and (relation is None or e.relation == relation):
                return True
        return False

    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._adj.values())

    def neighbors(self, name: str, relation: str | None = None) -> list[str]:
        """Return names of directly reachable nodes."""
        if name not in self._adj:
            return []
        edges = self._adj[name]
        if relation is not None:
            edges = [e for e in edges if e.relation == relation]
        return [e.target for e in edges]

    def edges_from(self, name: str) -> list[dict[str, Any]]:
        """Return edge dicts for all edges leaving *name*."""
        return [
            {"source": e.source, "target": e.target, "relation": e.relation,
             "weight": e.weight, **e.metadata}
            for e in self._adj.get(name, [])
        ]

    # ------------------------------------------------------------------
    # Path finding (BFS)
    # ------------------------------------------------------------------

    def path(self, source: str, target: str) -> list[str] | None:
        """Return a shortest (hop-count) path, or *None* if unreachable."""
        if source == target:
            return [source]
        if source not in self._adj:
            return None
        visited = {source}
        queue: list[list[str]] = [[source]]
        while queue:
            current_path = queue.pop(0)
            node = current_path[-1]
            for nbr in self.neighbors(node):
                if nbr == target:
                    return current_path + [nbr]
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(current_path + [nbr])
        return None

    def reachable(self, source: str) -> set[str]:
        """Return all nodes reachable from *source* (excluding itself)."""
        visited: set[str] = set()
        stack = [source]
        while stack:
            node = stack.pop()
            for nbr in self.neighbors(node):
                if nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)
        return visited

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "directed": self.directed,
            "nodes": [
                {"name": n.name, "node_type": n.node_type, "payload": n.payload}
                for n in self._nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target, "relation": e.relation,
                 "weight": e.weight, **e.metadata}
                for edges in self._adj.values()
                for e in edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeGraph":
        kg = cls(directed=data.get("directed", True))
        for nd in data.get("nodes", []):
            kg.add_node(nd["name"], node_type=nd.get("node_type", "concept"),
                        **nd.get("payload", {}))
        for ed in data.get("edges", []):
            src, tgt = ed["source"], ed["target"]
            rel = ed.get("relation", "related_to")
            w = ed.get("weight", 1.0)
            meta = {k: v for k, v in ed.items()
                    if k not in {"source", "target", "relation", "weight"}}
            kg.add_edge(src, tgt, relation=rel, weight=w, **meta)
        return kg

    def __repr__(self) -> str:
        return (f"KnowledgeGraph(nodes={self.node_count()}, "
                f"edges={self.edge_count()}, directed={self.directed})")
