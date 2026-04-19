"""Stage 110 — DocumentProcessor: local document ingestion.

Processes local documents AND remote URLs/images into text features
the agent can learn from.

Supported formats:
* Plain text (``.txt``) — read as-is.
* CSV / TSV — loaded into a ``pandas`` DataFrame.
* JSON — read and summarised.
* PDF — extracted via ``pdfplumber`` or ``PyPDF2`` (graceful fallback).
* Images (``.png``, ``.jpg``, ``.jpeg``, ``.bmp``, ``.gif``) — OCR via
  pytesseract (optional) or description via Claude vision (optional).
* URLs (``http://`` / ``https://``) — fetched and stripped to plain text.
* Excel (``.xlsx``, ``.xls``) — loaded via openpyxl/xlrd into DataFrame.

Returns a :class:`DocumentResult` with extracted text, metadata, and
optional tabular data.

Usage
-----
::

    from physml.doc_processor import DocumentProcessor

    proc = DocumentProcessor()
    result = proc.process("report.csv")
    result2 = proc.process("https://example.com/data.html")
    result3 = proc.process("screenshot.png")
    print(result.text)
    print(result.df)         # pandas DataFrame if tabular
    print(result.metadata)   # {"type": "csv", "rows": 120, ...}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


@dataclass
class DocumentResult:
    """Result of processing a document.

    Attributes
    ----------
    text : str
        Plain-text representation of the document content.
    df : pandas.DataFrame or None
        Tabular data if the document is a CSV/TSV (or ``None``).
    metadata : dict
        Document metadata (type, rows, columns, size, etc.).
    source : str
        Absolute path to the source file.
    success : bool
        ``True`` if processing succeeded.
    error : str or None
        Error message when *success* is ``False``.
    """

    text: str
    df: Any  # Optional[pd.DataFrame]
    metadata: Dict[str, Any]
    source: str
    success: bool = True
    error: Optional[str] = None


class DocumentProcessor:
    """Process local documents into text / tabular features.

    Parameters
    ----------
    encoding : str, default "utf-8"
        Default text encoding for plain-text and CSV files.
    max_text_chars : int, default 200_000
        Truncate extracted text to this many characters.
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        max_text_chars: int = 200_000,
    ) -> None:
        self.encoding = encoding
        self.max_text_chars = max_text_chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, path: str) -> DocumentResult:
        """Process a document, URL, or image and return a :class:`DocumentResult`.

        Parameters
        ----------
        path : str
            Path to the document file, OR a URL starting with http/https.

        Returns
        -------
        DocumentResult
        """
        # Handle URLs
        if path.startswith("http://") or path.startswith("https://"):
            return self._process_url(path)

        p = Path(path).expanduser().resolve()
        if not p.exists():
            return DocumentResult(
                text="",
                df=None,
                metadata={"type": "unknown"},
                source=str(p),
                success=False,
                error=f"File not found: {p}",
            )

        suffix = p.suffix.lower()
        try:
            if suffix in (".csv", ".tsv"):
                return self._process_csv(p)
            elif suffix == ".json":
                return self._process_json(p)
            elif suffix == ".pdf":
                return self._process_pdf(p)
            elif suffix in (".xlsx", ".xls"):
                return self._process_excel(p)
            elif suffix in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"):
                return self._process_image(p)
            else:
                # Treat everything else as plain text
                return self._process_text(p)
        except Exception as exc:
            _logger.warning("DocumentProcessor: failed to process %s: %s", p, exc)
            return DocumentResult(
                text="",
                df=None,
                metadata={"type": suffix.lstrip(".") or "unknown"},
                source=str(p),
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Format handlers
    # ------------------------------------------------------------------

    def _process_text(self, p: Path) -> DocumentResult:
        content = p.read_text(encoding=self.encoding, errors="replace")
        content = content[: self.max_text_chars]
        return DocumentResult(
            text=content,
            df=None,
            metadata={
                "type": "text",
                "size_bytes": p.stat().st_size,
                "chars": len(content),
            },
            source=str(p),
        )

    def _process_csv(self, p: Path) -> DocumentResult:
        try:
            import pandas as pd  # type: ignore

            sep = "\t" if p.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(p, sep=sep, encoding=self.encoding)
            text_repr = df.head(20).to_string(index=False)
            text_repr = text_repr[: self.max_text_chars]
            return DocumentResult(
                text=text_repr,
                df=df,
                metadata={
                    "type": p.suffix.lower().lstrip("."),
                    "rows": len(df),
                    "columns": list(df.columns),
                    "n_columns": len(df.columns),
                    "size_bytes": p.stat().st_size,
                },
                source=str(p),
            )
        except ImportError:
            # Fallback: read as plain text
            _logger.warning("DocumentProcessor: pandas not installed, falling back to text mode for %s", p)
            return self._process_text(p)

    def _process_json(self, p: Path) -> DocumentResult:
        raw = p.read_text(encoding=self.encoding, errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return DocumentResult(
                text=raw[: self.max_text_chars],
                df=None,
                metadata={"type": "json", "valid": False, "size_bytes": p.stat().st_size},
                source=str(p),
                success=False,
                error=str(e),
            )

        # Try to coerce a list of dicts to a DataFrame
        df = None
        if isinstance(data, list) and data and isinstance(data[0], dict):
            try:
                import pandas as pd  # type: ignore

                df = pd.DataFrame(data)
            except Exception as e:
                _logger.warning("DocumentProcessor: could not create DataFrame from JSON list: %s", e)

        text_repr = json.dumps(data, indent=2)
        text_repr = text_repr[: self.max_text_chars]

        meta: Dict[str, Any] = {
            "type": "json",
            "valid": True,
            "size_bytes": p.stat().st_size,
        }
        if isinstance(data, list):
            meta["n_records"] = len(data)
        elif isinstance(data, dict):
            meta["keys"] = list(data.keys())[:20]

        return DocumentResult(text=text_repr, df=df, metadata=meta, source=str(p))

    def _process_pdf(self, p: Path) -> DocumentResult:
        text = ""
        backend = "none"
        error = None

        # Try pdfplumber first
        try:
            import pdfplumber  # type: ignore

            with pdfplumber.open(str(p)) as pdf:
                pages = []
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    pages.append(t)
                text = "\n\n".join(pages)
            backend = "pdfplumber"
        except ImportError:
            _logger.warning("DocumentProcessor: pdfplumber not installed, trying PyPDF2")
        except Exception as exc:
            _logger.warning("DocumentProcessor: pdfplumber failed: %s", exc)
            error = str(exc)

        # Try PyPDF2 as fallback
        if not text:
            try:
                import PyPDF2  # type: ignore

                with open(p, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    pages = [
                        reader.pages[i].extract_text() or ""
                        for i in range(len(reader.pages))
                    ]
                text = "\n\n".join(pages)
                backend = "PyPDF2"
                error = None
            except ImportError:
                _logger.warning(
                    "DocumentProcessor: neither pdfplumber nor PyPDF2 installed; "
                    "returning empty text for PDF %s",
                    p,
                )
                if error is None:
                    error = "No PDF backend available (install pdfplumber or PyPDF2)"
            except Exception as exc:
                _logger.warning("DocumentProcessor: PyPDF2 failed: %s", exc)
                if error is None:
                    error = str(exc)

        text = text[: self.max_text_chars]
        success = bool(text)
        return DocumentResult(
            text=text,
            df=None,
            metadata={
                "type": "pdf",
                "backend": backend,
                "size_bytes": p.stat().st_size,
                "chars": len(text),
            },
            source=str(p),
            success=success,
            error=error if not success else None,
        )

    def _process_url(self, url: str) -> DocumentResult:
        """Fetch a URL and strip to plain text."""
        import re
        import urllib.request
        text = ""
        error = None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mycelium/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            # Strip HTML tags
            text = re.sub(r"<style[^>]*>.*?</style>", " ", raw, flags=re.S)
            text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.S)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s{2,}", " ", text).strip()
            text = text[: self.max_text_chars]
        except Exception as exc:
            error = str(exc)
            _logger.warning("DocumentProcessor URL fetch failed %s: %s", url, exc)

        return DocumentResult(
            text=text,
            df=None,
            metadata={"type": "url", "url": url, "chars": len(text)},
            source=url,
            success=bool(text),
            error=error,
        )

    def _process_excel(self, p: Path) -> DocumentResult:
        """Load Excel file into a DataFrame."""
        try:
            import pandas as pd
            df = pd.read_excel(str(p))
            text_repr = df.head(20).to_string(index=False)[: self.max_text_chars]
            return DocumentResult(
                text=text_repr,
                df=df,
                metadata={
                    "type": "excel",
                    "rows": len(df),
                    "columns": list(df.columns),
                    "n_columns": len(df.columns),
                    "size_bytes": p.stat().st_size,
                },
                source=str(p),
            )
        except Exception as exc:
            return DocumentResult(
                text="", df=None,
                metadata={"type": "excel"},
                source=str(p), success=False, error=str(exc),
            )

    def _process_image(self, p: Path) -> DocumentResult:
        """Extract text from an image via OCR (pytesseract) if available."""
        text = ""
        backend = "none"
        error = None

        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
            img = Image.open(str(p))
            text = pytesseract.image_to_string(img)
            backend = "pytesseract"
        except ImportError:
            error = "pytesseract/Pillow not installed; install them for image OCR"
            _logger.info("DocumentProcessor: %s", error)
        except Exception as exc:
            error = str(exc)
            _logger.warning("DocumentProcessor image OCR failed: %s", exc)

        text = text.strip()[: self.max_text_chars]
        return DocumentResult(
            text=text,
            df=None,
            metadata={
                "type": "image",
                "backend": backend,
                "size_bytes": p.stat().st_size,
                "chars": len(text),
            },
            source=str(p),
            success=bool(text),
            error=error if not text else None,
        )

    def __repr__(self) -> str:
        return f"DocumentProcessor(encoding={self.encoding!r})"
