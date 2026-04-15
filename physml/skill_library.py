"""Stage 97 — SkillLibrary: store, retrieve, and invoke reusable agent skills.

Provides a registry of named callable *skills* that the agent can look up
by name or by semantic tag match.  Each skill is wrapped in a
:class:`Skill` descriptor that records call counts and outcomes.

Classes
-------
Skill
    A wrapped, named callable with usage metadata.
SkillLibrary
    Registry for skills; supports tag-based retrieval and invocation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Skill:
    """One registered skill.

    Attributes
    ----------
    name : str
        Unique skill identifier.
    fn : Callable
        The callable that implements the skill.
    tags : list[str]
        Descriptive tags for semantic lookup.
    description : str
        Human-readable explanation.
    call_count : int
        How many times the skill has been invoked.
    last_called : float or None
        Unix timestamp of the most recent call.
    """

    name: str
    fn: Callable
    tags: List[str] = field(default_factory=list)
    description: str = ""
    call_count: int = 0
    last_called: Optional[float] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        self.last_called = time.time()
        return self.fn(*args, **kwargs)


class SkillLibrary:
    """Registry for reusable agent skills.

    Parameters
    ----------
    None

    Attributes
    ----------
    skills_ : dict[str, Skill]
        All registered skills keyed by name.
    """

    def __init__(self) -> None:
        self.skills_: Dict[str, Skill] = {}

    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        fn: Callable,
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> Skill:
        """Register a callable as a named skill.

        Parameters
        ----------
        name : str
            Unique name.  Raises ``ValueError`` if already registered.
        fn : Callable
        tags : list[str], optional
        description : str, optional

        Returns
        -------
        Skill
        """
        if name in self.skills_:
            raise ValueError(f"Skill '{name}' is already registered.")
        skill = Skill(name=name, fn=fn, tags=list(tags or []), description=description)
        self.skills_[name] = skill
        return skill

    def update(
        self,
        name: str,
        fn: Callable,
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> Skill:
        """Register or overwrite a skill.

        Returns
        -------
        Skill
        """
        skill = Skill(name=name, fn=fn, tags=list(tags or []), description=description)
        self.skills_[name] = skill
        return skill

    # ------------------------------------------------------------------
    def get(self, name: str) -> Skill:
        """Return the :class:`Skill` registered under *name*.

        Raises
        ------
        KeyError
            If *name* is not in the library.
        """
        if name not in self.skills_:
            raise KeyError(f"Skill '{name}' not found.")
        return self.skills_[name]

    def has(self, name: str) -> bool:
        """Return ``True`` if a skill named *name* exists."""
        return name in self.skills_

    # ------------------------------------------------------------------
    def find_by_tag(self, tag: str) -> List[Skill]:
        """Return all skills whose ``tags`` list contains *tag*.

        Parameters
        ----------
        tag : str
            Tag to search for (case-insensitive).

        Returns
        -------
        list[Skill]
        """
        tag_lower = tag.lower()
        return [s for s in self.skills_.values() if tag_lower in [t.lower() for t in s.tags]]

    # ------------------------------------------------------------------
    def invoke(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke the skill named *name* with the given arguments.

        Parameters
        ----------
        name : str
        *args, **kwargs
            Forwarded to the skill callable.

        Returns
        -------
        Any
            Return value of the skill callable.
        """
        return self.get(name)(*args, **kwargs)

    # ------------------------------------------------------------------
    def remove(self, name: str) -> bool:
        """Remove a registered skill.

        Returns
        -------
        bool
            ``True`` if the skill existed and was removed, ``False``
            otherwise.
        """
        if name in self.skills_:
            del self.skills_[name]
            return True
        return False

    def list_names(self) -> List[str]:
        """Return a sorted list of all registered skill names."""
        return sorted(self.skills_.keys())

    def __len__(self) -> int:
        return len(self.skills_)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SkillLibrary(skills={self.list_names()})"
