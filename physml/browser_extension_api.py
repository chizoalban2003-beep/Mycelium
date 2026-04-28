"""physml.browser_extension_api — FastAPI router for the Mycelium browser extension.

Provides REST endpoints that the browser extension POSTs to when the user:

* Visits a page (``/ext/page-visit``)
* Selects / highlights text (``/ext/selection``)
* Saves a bookmark (``/ext/bookmark``)
* Sends a command (``/ext/command``)

All data is fed into :class:`~physml.multimodal_ingester.MultiModalIngester`
for knowledge extraction and semantic storage.

Mount this router in ``server.py``::

    from physml.browser_extension_api import router as ext_router
    app.include_router(ext_router, prefix="/ext")

CORS is pre-configured to allow the browser extension origin (``null`` origin
for extension pages + ``*`` for local dev).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Try to import FastAPI; if missing, provide a stub that raises informatively
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore
    BaseModel = object  # type: ignore


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    from pydantic import BaseModel as _BM

    class PageVisitRequest(_BM):
        url: str
        title: str = ""
        text: str = ""
        timestamp: Optional[float] = None
        session_id: str = "default"

    class SelectionRequest(_BM):
        url: str
        selected_text: str
        page_title: str = ""
        timestamp: Optional[float] = None

    class BookmarkRequest(_BM):
        url: str
        title: str = ""
        note: str = ""
        tags: List[str] = []

    class CommandRequest(_BM):
        command: str
        args: Dict[str, Any] = {}

else:
    class PageVisitRequest:  # type: ignore
        pass
    class SelectionRequest:  # type: ignore
        pass
    class BookmarkRequest:  # type: ignore
        pass
    class CommandRequest:  # type: ignore
        pass


# ---------------------------------------------------------------------------
# Lazy ingester
# ---------------------------------------------------------------------------

_ingester: Any = None


def _get_ingester() -> Any:
    global _ingester
    if _ingester is None:
        try:
            from physml.multimodal_ingester import MultiModalIngester
            _ingester = MultiModalIngester()
        except Exception as exc:
            _logger.warning("browser_extension_api: MultiModalIngester unavailable: %s", exc)
    return _ingester


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    router = APIRouter(tags=["browser_extension"])

    @router.post("/page-visit")
    def page_visit(req: PageVisitRequest) -> Dict[str, Any]:
        """Record a browser page visit and ingest its text content."""
        ingester = _get_ingester()
        result: Dict[str, Any] = {"url": req.url, "ingested": False}
        if ingester is not None:
            try:
                text = f"Page visit: {req.title}\nURL: {req.url}\n\n{req.text}"
                ir = ingester.ingest(text.strip(), topic="browsing")
                result["ingested"] = ir.success
                result["facts"] = len(ir.facts)
            except Exception as exc:
                _logger.warning("page_visit ingest error: %s", exc)
        _logger.info("ext/page-visit: %s (ingested=%s)", req.url[:80], result["ingested"])
        return result

    @router.post("/selection")
    def selection(req: SelectionRequest) -> Dict[str, Any]:
        """Ingest highlighted/selected text from the browser."""
        if not req.selected_text.strip():
            raise HTTPException(status_code=400, detail="selected_text is empty")
        ingester = _get_ingester()
        result: Dict[str, Any] = {"ingested": False}
        if ingester is not None:
            try:
                text = f"User selected from {req.page_title} ({req.url}):\n{req.selected_text}"
                ir = ingester.ingest(text, topic="reading")
                result["ingested"] = ir.success
                result["facts"] = len(ir.facts)
            except Exception as exc:
                _logger.warning("selection ingest error: %s", exc)
        return result

    @router.post("/bookmark")
    def bookmark(req: BookmarkRequest) -> Dict[str, Any]:
        """Record a bookmark and optionally fetch + ingest the page."""
        ingester = _get_ingester()
        result: Dict[str, Any] = {"url": req.url, "ingested": False}
        if ingester is not None:
            try:
                note = req.note or req.title or ""
                text = f"Bookmark: {req.title}\nURL: {req.url}\nTags: {', '.join(req.tags)}\n{note}"
                ir = ingester.ingest(req.url if not note else text.strip(), topic="bookmark")
                result["ingested"] = ir.success
            except Exception as exc:
                _logger.warning("bookmark ingest error: %s", exc)
        return result

    @router.post("/command")
    def command(req: CommandRequest) -> Dict[str, Any]:
        """Run a Mycelium command from the browser extension popup."""
        cmd = req.command.strip().lower()
        if cmd == "status":
            ing = _get_ingester()
            return {"status": "ok", "ingested": ing.ingested_count if ing else 0}
        elif cmd == "summary":
            ing = _get_ingester()
            return {"summary": ing.summary() if ing else {}}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown command: {cmd!r}")

    @router.get("/status")
    def ext_status() -> Dict[str, Any]:
        """Health check for the extension endpoint."""
        ing = _get_ingester()
        return {
            "ok": True,
            "ingester": ing.status() if ing else None,
        }

else:
    # Stub when FastAPI is not installed
    router = None  # type: ignore
    _logger.debug("browser_extension_api: FastAPI not installed — router unavailable")
