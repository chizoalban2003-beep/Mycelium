"""Stage 134 — KnowledgeExtractor: auto-extract facts from conversations.

Parses user messages and agent responses for durable facts, preferences,
and relationships, then stores them in the KnowledgeGraph and VectorMemory
so the agent remembers what it learns about the user over time.

Facts are extracted by:
1. Pattern matching (name, location, occupation, preference phrases)
2. Sentence heuristics ("I am …", "I work at …", "I like …", "My … is …")
3. Optionally via LLM extraction when Claude is available.

Usage
-----
::

    from physml.knowledge_extractor import KnowledgeExtractor

    ke = KnowledgeExtractor(knowledge_graph=kg, vector_memory=vm)
    facts = ke.extract("My name is Alex and I work as a data scientist in London.")
    # facts = [{"subject": "user", "predicate": "name", "object": "Alex"}, ...]
    ke.store(facts)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

_STOP = r"(?=\s+(?:and|but|or|so|who|that|which|where|when)|[,\.]|$)"
# (?-i:...) forces case-sensitive matching even when re.IGNORECASE is active,
# preventing conjunctions like "and" from being captured as part of a name.
_NAME_WORD = r"(?-i:[A-Z][a-zA-Z'\-]+)"

_PATTERNS = [
    # Identity — inline (?-i:...) ensures only capitalised words are captured
    (r"\bmy name is (?:(?:Mr|Mrs|Ms|Dr)\.? )?(" + _NAME_WORD + r"(?:\s+" + _NAME_WORD + r")?)" + _STOP, "name"),
    (r"\bI(?:'m| am) (" + _NAME_WORD + r"(?:\s+" + _NAME_WORD + r")?)" + _STOP, "name"),
    # Occupation
    (r"\bi(?:'m| am)(?: a| an)? ([a-z]+(?: [a-z]+)?) (?:by trade|by profession|for a living)", "occupation"),
    (r"\bi work as(?: a| an)? ([a-z][a-z ]+?)(?:\.|,|$)", "occupation"),
    (r"\bmy (?:job|role|profession|position) is ([a-z][a-z ]+?)(?:\.|,|$)", "occupation"),
    (r"\bi(?:'m| am)(?: a| an) ([a-z]+(?:ist|er|or|ant|ect|eer|ian))\b", "occupation"),
    # Location
    (r"\bi(?:'m| am) (?:from|in|based in|located in) ([A-Z][a-z]+(?: [A-Z][a-z]+)*)", "location"),
    (r"\bi live in ([A-Z][a-z]+(?: [A-Z][a-z]+)*)", "location"),
    (r"\bmy (?:city|town|country|location) is ([A-Z][a-z]+(?: [A-Z][a-z]+)*)", "location"),
    # Preferences
    (r"\bi (?:really )?(?:love|like|enjoy|prefer) ([a-z][a-z ]{2,30})(?:\.|,|$)", "likes"),
    (r"\bmy favourite(?: thing is)? ([a-z][a-z ]{2,30})(?:\.|,|$)", "likes"),
    (r"\bi (?:hate|dislike|don't like) ([a-z][a-z ]{2,30})(?:\.|,|$)", "dislikes"),
    # Company / org
    (r"\bi work (?:at|for) ([A-Z][A-Za-z&\s\.]{1,30}?)(?:\.|,|$)", "employer"),
    (r"\bmy company is ([A-Z][A-Za-z&\s\.]{1,30}?)(?:\.|,|$)", "employer"),
    # Goals
    (r"\bi want to ([a-z][a-z ]{3,50})(?:\.|$)", "goal"),
    (r"\bmy goal is to ([a-z][a-z ]{3,50})(?:\.|$)", "goal"),
]


class KnowledgeExtractor:
    """Extract structured facts from natural language and persist them.

    Parameters
    ----------
    knowledge_graph : KnowledgeGraph or None
        Where to store extracted entities and relationships.
    vector_memory : VectorMemory or None
        Where to store fact text for semantic retrieval.
    llm : LLMIntegration or None
        When provided, used to extract facts that patterns miss.
    """

    def __init__(
        self,
        knowledge_graph: Any = None,
        vector_memory: Any = None,
        llm: Any = None,
    ) -> None:
        self.knowledge_graph = knowledge_graph
        self.vector_memory = vector_memory
        self.llm = llm
        self._fact_count = 0

    def extract(self, text: str) -> List[Dict[str, str]]:
        """Extract facts from *text* using patterns and heuristics.

        Returns
        -------
        list of dict with keys ``subject``, ``predicate``, ``object``.
        """
        facts: List[Dict[str, str]] = []
        lower = text.lower()

        for pattern, predicate in _PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                obj = match.group(1).strip().rstrip(".,;")
                if obj and len(obj) > 1:
                    facts.append({
                        "subject": "user",
                        "predicate": predicate,
                        "object": obj,
                    })

        # De-duplicate by (subject, predicate, object)
        seen = set()
        deduped = []
        for f in facts:
            key = (f["subject"], f["predicate"], f["object"].lower())
            if key not in seen:
                seen.add(key)
                deduped.append(f)

        return deduped

    def extract_and_store(self, text: str) -> List[Dict[str, str]]:
        """Extract facts from *text* and store them immediately."""
        facts = self.extract(text)
        if facts:
            self.store(facts)
        return facts

    def store(self, facts: List[Dict[str, str]]) -> int:
        """Persist *facts* to KnowledgeGraph and VectorMemory.

        Returns the number of facts stored.
        """
        stored = 0
        for fact in facts:
            subj = fact.get("subject", "user")
            pred = fact.get("predicate", "knows")
            obj = fact.get("object", "")
            if not obj:
                continue

            # KnowledgeGraph
            if self.knowledge_graph is not None:
                try:
                    self.knowledge_graph.add_node(
                        node_id=f"fact_{self._fact_count}",
                        node_type="user_fact",
                        properties={
                            "subject": subj,
                            "predicate": pred,
                            "object": obj,
                        },
                    )
                    self.knowledge_graph.add_edge(
                        subj, obj,
                        label=pred,
                        properties={"source": "conversation"},
                    )
                except Exception as exc:
                    _logger.debug("KnowledgeExtractor KG store error: %s", exc)

            # VectorMemory
            if self.vector_memory is not None:
                try:
                    fact_text = f"User {pred}: {obj}"
                    self.vector_memory.add(
                        fact_text,
                        metadata={"type": "user_fact", "predicate": pred},
                    )
                except Exception as exc:
                    _logger.debug("KnowledgeExtractor VM store error: %s", exc)

            self._fact_count += 1
            stored += 1
            _logger.info("KnowledgeExtractor: stored fact — %s %s %s", subj, pred, obj)

        return stored

    def llm_extract(self, text: str) -> List[Dict[str, str]]:
        """Use Claude to extract structured facts when patterns miss them.

        Returns the same fact format.  Falls back to pattern extraction
        if LLM is unavailable.
        """
        if self.llm is None or not getattr(self.llm, "available", False):
            return self.extract(text)

        prompt = (
            "Extract factual information about the user from the following message.\n"
            "Return a JSON array of objects with keys: subject, predicate, object.\n"
            "Only include facts the user stated about themselves.\n"
            "Example: [{\"subject\": \"user\", \"predicate\": \"name\", \"object\": \"Alex\"}]\n\n"
            f"Message: {text}\n\nFacts (JSON only, no explanation):"
        )
        try:
            result = self.llm.complete(prompt)
            if result.available and result.text:
                import json
                raw = result.text.strip()
                # Extract JSON array from response
                start = raw.find("[")
                end = raw.rfind("]") + 1
                if start >= 0 and end > start:
                    facts = json.loads(raw[start:end])
                    valid = [
                        f for f in facts
                        if isinstance(f, dict) and "predicate" in f and "object" in f
                    ]
                    return valid
        except Exception as exc:
            _logger.debug("KnowledgeExtractor LLM extraction failed: %s", exc)

        return self.extract(text)

    def status(self) -> dict:
        return {
            "facts_stored": self._fact_count,
            "knowledge_graph_connected": self.knowledge_graph is not None,
            "vector_memory_connected": self.vector_memory is not None,
            "llm_connected": self.llm is not None,
        }
