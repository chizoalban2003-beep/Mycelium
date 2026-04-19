"""Stage 131 — PermissionManager: gate all OS-level actions.

Every file write, process execution, browser action, and screen automation
request passes through this manager.  By default actions are allowed in
safe categories; anything destructive requires explicit user grant.

Permissions are persisted to ~/.mycelium/permissions.json so granted
rights survive restarts.

Usage
-----
::

    from physml.permission_manager import PermissionManager, PermissionLevel

    pm = PermissionManager()
    pm.grant("file.write")
    pm.grant("browser.navigate")

    if pm.check("file.write"):
        # safe to write
        ...
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

from physml._log import get_logger

_logger = get_logger(__name__)


class PermissionLevel(str, Enum):
    DENY = "deny"
    ASK = "ask"
    ALLOW = "allow"


# Default permission policy — conservative safe defaults
_DEFAULTS: Dict[str, str] = {
    # Always allowed — read-only, harmless
    "file.read": PermissionLevel.ALLOW,
    "screen.screenshot": PermissionLevel.ALLOW,
    "browser.navigate": PermissionLevel.ALLOW,
    "browser.read": PermissionLevel.ALLOW,
    "predict": PermissionLevel.ALLOW,
    "train": PermissionLevel.ALLOW,
    # Ask first — potentially side-effecting
    "file.write": PermissionLevel.ASK,
    "file.delete": PermissionLevel.ASK,
    "browser.click": PermissionLevel.ASK,
    "browser.fill": PermissionLevel.ASK,
    "screen.click": PermissionLevel.ASK,
    "screen.type": PermissionLevel.ASK,
    "process.run": PermissionLevel.ASK,
    "notification.send": PermissionLevel.ALLOW,
    # Deny by default — dangerous
    "process.kill": PermissionLevel.DENY,
    "file.delete_bulk": PermissionLevel.DENY,
    "system.reboot": PermissionLevel.DENY,
}


class PermissionManager:
    """Central permission gating for all OS-level companion actions.

    Parameters
    ----------
    config_path : str, default "~/.mycelium/permissions.json"
        Where to persist grants.
    auto_ask_callback : callable or None
        If set, called with (action, description) when a permission is ``ASK``.
        Should return True to allow or False to deny.
        When None, ``ASK`` actions are auto-denied (safe default).
    """

    def __init__(
        self,
        config_path: str = "~/.mycelium/permissions.json",
        auto_ask_callback=None,
    ) -> None:
        self._path = Path(config_path).expanduser()
        self._policy: Dict[str, str] = dict(_DEFAULTS)
        self._granted: Set[str] = set()
        self._denied: Set[str] = set()
        self._auto_ask = auto_ask_callback
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, action: str) -> bool:
        """Return True if *action* is allowed right now."""
        if action in self._denied:
            return False
        if action in self._granted:
            return True

        level = self._policy.get(action, PermissionLevel.ASK)
        if level == PermissionLevel.ALLOW:
            return True
        if level == PermissionLevel.DENY:
            _logger.info("PermissionManager: DENY %s (default policy)", action)
            return False

        # ASK
        if self._auto_ask is not None:
            allowed = self._auto_ask(action, f"Myco wants to perform: {action}")
            if allowed:
                _logger.info("PermissionManager: user GRANTED %s", action)
                self.grant(action)
            else:
                _logger.info("PermissionManager: user DENIED %s", action)
                self.deny(action)
            return allowed

        _logger.info("PermissionManager: ASK→DENY %s (no callback)", action)
        return False

    def grant(self, action: str, persist: bool = True) -> None:
        """Grant *action* permanently."""
        self._granted.add(action)
        self._denied.discard(action)
        if persist:
            self._save()

    def deny(self, action: str, persist: bool = False) -> None:
        """Deny *action* for this session."""
        self._denied.add(action)
        self._granted.discard(action)

    def reset(self, action: str) -> None:
        """Remove any explicit grant/deny — revert to default policy."""
        self._granted.discard(action)
        self._denied.discard(action)
        self._save()

    def set_policy(self, action: str, level: PermissionLevel) -> None:
        """Override the default policy for *action*."""
        self._policy[action] = level
        self._save()

    def summary(self) -> dict:
        return {
            "granted": sorted(self._granted),
            "denied": sorted(self._denied),
            "policy": dict(self._policy),
        }

    def allowed_actions(self) -> List[str]:
        """List all actions currently allowed."""
        result = []
        for action, level in self._policy.items():
            if level == PermissionLevel.ALLOW or action in self._granted:
                if action not in self._denied:
                    result.append(action)
        return sorted(result)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {"granted": sorted(self._granted), "policy": self._policy}
            self._path.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            _logger.debug("PermissionManager save failed: %s", exc)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._granted = set(data.get("granted", []))
            self._policy.update(data.get("policy", {}))
        except Exception as exc:
            _logger.debug("PermissionManager load failed: %s", exc)
