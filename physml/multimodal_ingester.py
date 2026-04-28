"""physml.multimodal_ingester — Unified multi-modal learning pipeline.

Routes *any* input (file path, URL, raw text, bytes) through:

1. :class:`~physml.doc_processor.DocumentProcessor` — extract text + metadata
2. :class:`~physml.knowledge_extractor.KnowledgeExtractor` — extract facts → KnowledgeGraph
3. :class:`~physml.vector_memory.VectorMemory` — semantic storage
4. :class:`~physml.user_profile.UserProfileLearner` — update user profile

Supported input types
---------------------
* Plain text strings
* File paths: ``.txt``, ``.md``, ``.py``, ``.js``, ``.csv``, ``.json``,
  ``.pdf``, ``.xlsx``, ``.png``, ``.jpg``, ``.mp3``, ``.wav``
* HTTP/HTTPS URLs

Usage::

    from physml.multimodal_ingester import MultiModalIngester

    ingester = MultiModalIngester()
    result = ingester.ingest("notes.txt")
    result2 = ingester.ingest("https://docs.python.org/3/library/pathlib.html")
    result3 = ingester.ingest("My name is Alex and I love Python.")

    print(result.text[:200])
    print(result.facts)        # extracted knowledge triples
    print(result.memory_id)    # id in VectorMemory
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)

_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".sh", ".bash",
    ".zsh", ".sql", ".r", ".m", ".scala", ".lua",
}

_DOC_EXTENSIONS = {
    ".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml",
    ".toml", ".xml", ".html", ".htm", ".pdf", ".xlsx", ".xls",
}

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".opus"}


@dataclass
class IngestResult:
    """Result from processing one input through the pipeline.

    Attributes
    ----------
    source : str
        Original source path/URL/description.
    text : str
        Extracted plain text.
    facts : list[dict]
        Knowledge triples extracted: [{subject, predicate, object}].
    memory_id : str or None
        ID under which the text was stored in VectorMemory.
    metadata : dict
        Source-specific metadata (type, size, language, etc.).
    elapsed : float
        Processing time in seconds.
    success : bool
    error : str or None
    """

    source: str
    text: str = ""
    facts: List[Dict[str, str]] = field(default_factory=list)
    memory_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0
    success: bool = True
    error: Optional[str] = None


class MultiModalIngester:
    """Ingest any content type and feed it into Mycelium's learning stack.

    Parameters
    ----------
    vector_memory : VectorMemory or None
        Semantic memory store. Created with defaults when ``None``.
    knowledge_graph : KnowledgeGraph or None
        Graph store. Created with defaults when ``None``.
    knowledge_extractor : KnowledgeExtractor or None
        Fact extractor. Created with defaults when ``None``.
    user_profile : UserProfileLearner or None
        User profile updater. Created with defaults when ``None``.
    max_text_chars : int
        Maximum characters to store per document (truncated beyond this).
    deduplicate : bool
        Skip documents whose content hash has already been ingested.
    """

    def __init__(
        self,
        vector_memory: Any = None,
        knowledge_graph: Any = None,
        knowledge_extractor: Any = None,
        user_profile: Any = None,
        max_text_chars: int = 50_000,
        deduplicate: bool = True,
    ) -> None:
        self.max_text_chars = max_text_chars
        self.deduplicate = deduplicate
        self._seen_hashes: set = set()
        self._ingested: List[IngestResult] = []

        # Lazy-init subsystems
        self._vm = vector_memory
        self._kg = knowledge_graph
        self._ke = knowledge_extractor
        self._up = user_profile

    # ------------------------------------------------------------------
    # Lazy subsystem init
    # ------------------------------------------------------------------

    def _get_vm(self) -> Any:
        if self._vm is None:
            try:
                from physml.vector_memory import VectorMemory
                self._vm = VectorMemory(
                    persist_path=str(Path("~/.mycelium/vector_memory.json").expanduser())
                )
            except Exception as exc:
                _logger.debug("MultiModalIngester: VectorMemory unavailable: %s", exc)
        return self._vm

    def _get_kg(self) -> Any:
        if self._kg is None:
            try:
                from physml.knowledge_graph import KnowledgeGraph
                self._kg = KnowledgeGraph()
            except Exception as exc:
                _logger.debug("MultiModalIngester: KnowledgeGraph unavailable: %s", exc)
        return self._kg

    def _get_ke(self) -> Any:
        if self._ke is None:
            try:
                from physml.knowledge_extractor import KnowledgeExtractor
                self._ke = KnowledgeExtractor(
                    knowledge_graph=self._get_kg(),
                    vector_memory=self._get_vm(),
                )
            except Exception as exc:
                _logger.debug("MultiModalIngester: KnowledgeExtractor unavailable: %s", exc)
        return self._ke

    def _get_up(self) -> Any:
        if self._up is None:
            try:
                from physml.user_profile import UserProfileLearner
                self._up = UserProfileLearner()
            except Exception as exc:
                _logger.debug("MultiModalIngester: UserProfileLearner unavailable: %s", exc)
        return self._up

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, source: str, topic: str = "document") -> IngestResult:
        """Ingest *source* through the full multi-modal pipeline.

        Parameters
        ----------
        source : str
            File path, HTTP/HTTPS URL, or raw text string.
        topic : str
            Topic tag for user-profile tracking.

        Returns
        -------
        IngestResult
        """
        t0 = time.time()
        result = IngestResult(source=source)

        try:
            # 1. Extract text
            text, metadata = self._extract_text(source)
            if not text:
                result.success = False
                result.error = "No text extracted"
                result.elapsed = time.time() - t0
                return result

            # 2. Truncate
            text = text[:self.max_text_chars]
            result.text = text
            result.metadata = metadata

            # 3. Deduplication
            content_hash = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
            if self.deduplicate and content_hash in self._seen_hashes:
                result.metadata["deduplicated"] = True
                result.elapsed = time.time() - t0
                self._ingested.append(result)
                return result
            self._seen_hashes.add(content_hash)

            # 4. Store in VectorMemory
            vm = self._get_vm()
            if vm is not None:
                try:
                    entry = vm.add(
                        text[:2000],  # keep embedding cost low
                        metadata={
                            "source": source,
                            "type": metadata.get("type", "unknown"),
                            "ingested_at": time.time(),
                        },
                    )
                    result.memory_id = getattr(entry, "id", None) or content_hash
                except Exception as exc:
                    _logger.debug("VectorMemory.add error: %s", exc)

            # 5. Extract facts + store in KnowledgeGraph
            ke = self._get_ke()
            if ke is not None:
                try:
                    facts = ke.extract_and_store(text[:5000])
                    result.facts = facts
                except Exception as exc:
                    _logger.debug("KnowledgeExtractor.extract_and_store error: %s", exc)

            # 6. Update user profile
            up = self._get_up()
            if up is not None:
                try:
                    up.record_interaction(
                        intent="ingest",
                        feedback="neutral",
                        topic=topic or metadata.get("type", "document"),
                        metadata={"source": source},
                    )
                except Exception as exc:
                    _logger.debug("UserProfileLearner.record_interaction error: %s", exc)

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            _logger.warning("MultiModalIngester.ingest error for %r: %s", source, exc)

        result.elapsed = time.time() - t0
        self._ingested.append(result)
        return result

    def ingest_many(self, sources: List[str], topic: str = "document") -> List[IngestResult]:
        """Ingest a list of sources sequentially."""
        return [self.ingest(s, topic=topic) for s in sources]

    def ingest_directory(
        self,
        directory: str,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> List[IngestResult]:
        """Ingest all supported files in *directory*."""
        allowed = set(extensions) if extensions else (_CODE_EXTENSIONS | _DOC_EXTENSIONS | _IMAGE_EXTENSIONS)
        root = Path(directory).expanduser()
        if not root.is_dir():
            return []
        glob = root.rglob("*") if recursive else root.glob("*")
        paths = [p for p in glob if p.is_file() and p.suffix.lower() in allowed]
        return self.ingest_many([str(p) for p in paths])

    @property
    def ingested_count(self) -> int:
        return len(self._ingested)

    @property
    def last_result(self) -> Optional[IngestResult]:
        return self._ingested[-1] if self._ingested else None

    def summary(self) -> Dict[str, Any]:
        """Return ingestion stats."""
        results = self._ingested
        return {
            "total": len(results),
            "succeeded": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "facts_extracted": sum(len(r.facts) for r in results),
            "deduplicated": sum(1 for r in results if r.metadata.get("deduplicated")),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "ingested": self.ingested_count,
            "vector_memory": self._get_vm() is not None,
            "knowledge_graph": self._get_kg() is not None,
            "knowledge_extractor": self._get_ke() is not None,
            "user_profile": self._get_up() is not None,
        }

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_text(self, source: str) -> tuple[str, dict]:
        """Return (text, metadata) for any source type."""
        # Raw text (not a path and not a URL)
        if not source.startswith(("http://", "https://")) and not os.path.exists(source):
            if len(source) < 2000:
                return source, {"type": "raw_text"}
            return source, {"type": "long_text", "chars": len(source)}

        # Try DocumentProcessor first
        try:
            from physml.doc_processor import DocumentProcessor
            proc = DocumentProcessor(max_text_chars=self.max_text_chars)
            result = proc.process(source)
            if result.success and result.text:
                return result.text, result.metadata
        except Exception as exc:
            _logger.debug("DocumentProcessor failed for %r: %s", source, exc)

        # Fallback for code files — just read them
        path = Path(source)
        if path.is_file() and path.suffix.lower() in _CODE_EXTENSIONS:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                lang = path.suffix.lstrip(".")
                return text, {"type": "code", "language": lang, "path": str(path)}
            except Exception as exc:
                _logger.debug("Code file read failed for %r: %s", source, exc)

        # Fallback for audio — transcribe with whisper if available
        if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS:
            text = self._transcribe_audio(str(path))
            if text:
                return text, {"type": "audio_transcript", "path": str(path)}

        return "", {"type": "unknown", "source": source}

    def _transcribe_audio(self, path: str) -> str:
        try:
            import whisper  # type: ignore
            model = whisper.load_model("base")
            result = model.transcribe(path)
            return result.get("text", "").strip()
        except Exception as exc:
            _logger.debug("Audio transcription failed for %r: %s", path, exc)
            return ""

    def __repr__(self) -> str:
        return f"MultiModalIngester(ingested={self.ingested_count}, dedup={self.deduplicate})"
