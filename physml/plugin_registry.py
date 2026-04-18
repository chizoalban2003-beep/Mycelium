"""Stage 112 — PluginRegistry: user-installable plugin system.

Plugins are Python files placed in ``~/.mycelium/plugins/`` (or a custom
directory) that expose a ``register(registry)`` function.  On startup, all
``.py`` files are auto-discovered, imported, and their ``register``
function called.  Each plugin may register :class:`~physml.tools.Tool`
objects into the shared :class:`~physml.tools.ToolRegistry`.

Loading is isolated: if a plugin raises any exception during import or
registration it is recorded in ``failed`` but does not prevent other
plugins from loading.

Usage
-----
::

    from physml.plugin_registry import PluginRegistry

    registry = PluginRegistry(plugin_dir="~/.mycelium/plugins")
    registry.load_all()
    print(registry.loaded)         # ["my_plugin", ...]
    print(registry.failed)         # {"bad_plugin": "ImportError: ..."}
    print(registry.tool_registry)  # ToolRegistry with registered tools
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from physml._log import get_logger
from physml.tools import ToolRegistry

_logger = get_logger(__name__)


class PluginRegistry:
    """Auto-discover and load user plugins.

    Parameters
    ----------
    plugin_dir : str, default "~/.mycelium/plugins"
        Directory to scan for plugin files.
    tool_registry : ToolRegistry or None
        Shared tool registry.  A new one is created if ``None``.
    """

    def __init__(
        self,
        plugin_dir: str = "~/.mycelium/plugins",
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.plugin_dir = Path(plugin_dir).expanduser()
        self.tool_registry: ToolRegistry = tool_registry or ToolRegistry()
        self._loaded: List[str] = []
        self._failed: Dict[str, str] = {}
        self._callables: Dict[str, Any] = {}  # arbitrary in-process callables

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def loaded(self) -> List[str]:
        """Names of successfully loaded plugins."""
        return list(self._loaded)

    @property
    def failed(self) -> Dict[str, str]:
        """Mapping of plugin name → error message for failed plugins."""
        return dict(self._failed)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Discover and load all ``.py`` files in the plugin directory.

        Each file must define a ``register(registry)`` function that
        accepts a :class:`ToolRegistry`.
        """
        if not self.plugin_dir.exists():
            _logger.info(
                "PluginRegistry: plugin directory %s does not exist; skipping",
                self.plugin_dir,
            )
            return

        py_files = sorted(self.plugin_dir.glob("*.py"))
        _logger.info("PluginRegistry: found %d plugin files in %s", len(py_files), self.plugin_dir)

        for plugin_path in py_files:
            if plugin_path.name.startswith("_"):
                continue  # skip __init__.py, _private.py, etc.
            self._load_one(plugin_path)

    def load_file(self, path: str) -> bool:
        """Load a single plugin file.

        Parameters
        ----------
        path : str
            Path to the plugin ``.py`` file.

        Returns
        -------
        bool
            ``True`` on success.
        """
        return self._load_one(Path(path).expanduser().resolve())

    def _load_one(self, plugin_path: Path) -> bool:
        name = plugin_path.stem
        module_name = f"_mycelium_plugin_{name}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot create module spec for {plugin_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]

            if not hasattr(module, "register"):
                raise AttributeError(
                    f"Plugin {name!r} has no 'register' function"
                )
            module.register(self.tool_registry)
            self._loaded.append(name)
            _logger.info("PluginRegistry: loaded plugin %r", name)
            return True
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            self._failed[name] = msg
            _logger.warning("PluginRegistry: failed to load plugin %r: %s", name, msg)
            return False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reload(self, name: str) -> bool:
        """Reload a previously loaded or failed plugin by name.

        Parameters
        ----------
        name : str
            Plugin stem name (without ``.py``).

        Returns
        -------
        bool
        """
        plugin_path = self.plugin_dir / f"{name}.py"
        if not plugin_path.exists():
            _logger.warning("PluginRegistry: cannot reload %r: file not found", name)
            return False
        # Remove from loaded/failed
        if name in self._loaded:
            self._loaded.remove(name)
        self._failed.pop(name, None)
        # Remove from sys.modules to force re-import
        module_name = f"_mycelium_plugin_{name}"
        sys.modules.pop(module_name, None)
        return self._load_one(plugin_path)

    # ------------------------------------------------------------------
    # Convenience: in-process plugin registration
    # ------------------------------------------------------------------

    def register(self, name: str, fn: Any) -> None:
        """Register a callable directly (without a .py file).

        Parameters
        ----------
        name : str
            Unique name for this plugin function.
        fn : callable
            The callable to register.
        """
        self._callables[name] = fn
        if name not in self._loaded:
            self._loaded.append(name)

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a registered plugin by name.

        Parameters
        ----------
        name : str
        *args, **kwargs
            Forwarded to the callable.

        Returns
        -------
        Any
        """
        if name in self._callables:
            return self._callables[name](*args, **kwargs)
        raise KeyError(f"PluginRegistry: no plugin named {name!r}")

    def list(self) -> List[str]:
        """Return names of all registered plugins/tools."""
        return list(self._loaded)

    def __repr__(self) -> str:
        return (
            f"PluginRegistry("
            f"loaded={len(self._loaded)}, "
            f"failed={len(self._failed)}, "
            f"tools={len(self.tool_registry.list_tools())})"
        )
