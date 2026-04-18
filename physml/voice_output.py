"""Stage 122 — VoiceOutputAdapter: text-to-speech synthesis.

Provides local, privacy-first TTS for the Mycelium companion.  Backends
(in priority order):

1. **pyttsx3** — fully offline, cross-platform (Windows/macOS/Linux).
2. **gTTS** — online Google TTS (fallback when offline TTS unavailable).
3. **print** — silent passthrough (returns text, plays nothing).

Usage
-----
::

    from physml.voice_output import VoiceOutputAdapter

    tts = VoiceOutputAdapter()
    tts.speak("Hello! I am Mycelium, your local AI companion.")
    tts.speak("Prediction complete: 94% confidence.", save_path="/tmp/result.mp3")
    print(tts.active_backend)   # "pyttsx3" | "gtts" | "silent"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from physml._log import get_logger

_logger = get_logger(__name__)


@dataclass
class TTSResult:
    """Result of a TTS synthesis call.

    Attributes
    ----------
    text : str
        The original text that was synthesised.
    backend : str
        Backend used (``"pyttsx3"``, ``"gtts"``, ``"silent"``).
    saved_to : str or None
        Path if audio was saved to a file.
    success : bool
    error : str or None
    """

    text: str
    backend: str
    saved_to: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class VoiceOutputAdapter:
    """Text-to-speech adapter with graceful backend fallback.

    Parameters
    ----------
    preferred_backend : str, default "auto"
        ``"pyttsx3"``, ``"gtts"``, ``"silent"``, or ``"auto"``.
    rate : int, default 175
        Speech rate in words per minute (pyttsx3 only).
    volume : float, default 0.9
        Volume in [0.0, 1.0] (pyttsx3 only).
    language : str, default "en"
        Language code (gTTS only).
    """

    def __init__(
        self,
        preferred_backend: str = "auto",
        rate: int = 175,
        volume: float = 0.9,
        language: str = "en",
    ) -> None:
        self.preferred_backend = preferred_backend
        self.rate = rate
        self.volume = volume
        self.language = language
        self._available_backends = self._detect_backends()
        _logger.info("VoiceOutputAdapter: available backends: %s", self._available_backends)

    def _detect_backends(self) -> list[str]:
        backends: list[str] = []
        if self.preferred_backend in ("pyttsx3", "auto"):
            try:
                import pyttsx3  # type: ignore  # noqa: F401

                backends.append("pyttsx3")
            except ImportError:
                pass
        if self.preferred_backend in ("gtts", "auto"):
            try:
                from gtts import gTTS  # type: ignore  # noqa: F401

                backends.append("gtts")
            except ImportError:
                pass
        backends.append("silent")
        return backends

    @property
    def active_backend(self) -> str:
        if self.preferred_backend != "auto" and self.preferred_backend in self._available_backends:
            return self.preferred_backend
        return self._available_backends[0]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def speak(self, text: str, save_path: Optional[str] = None) -> TTSResult:
        """Synthesise *text* to audio and/or play it.

        Parameters
        ----------
        text : str
            Text to speak.
        save_path : str or None
            If given, save the audio to this path instead of (or in addition
            to) playing it aloud.

        Returns
        -------
        TTSResult
        """
        backend = self.active_backend
        if backend == "pyttsx3":
            return self._pyttsx3_speak(text, save_path)
        elif backend == "gtts":
            return self._gtts_speak(text, save_path)
        else:
            _logger.info("VoiceOutputAdapter (silent): %r", text[:60])
            return TTSResult(text=text, backend="silent", saved_to=save_path)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _pyttsx3_speak(self, text: str, save_path: Optional[str] = None) -> TTSResult:
        try:
            import pyttsx3  # type: ignore

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            if save_path:
                engine.save_to_file(text, save_path)
                engine.runAndWait()
                return TTSResult(text=text, backend="pyttsx3", saved_to=save_path)
            else:
                engine.say(text)
                engine.runAndWait()
                return TTSResult(text=text, backend="pyttsx3")
        except Exception as exc:
            _logger.warning("VoiceOutputAdapter (pyttsx3): %s", exc)
            return TTSResult(
                text=text,
                backend="pyttsx3",
                success=False,
                error=str(exc),
            )

    def _gtts_speak(self, text: str, save_path: Optional[str] = None) -> TTSResult:
        try:
            from gtts import gTTS  # type: ignore
            import tempfile
            import os

            tts = gTTS(text=text, lang=self.language)
            path = save_path or tempfile.mktemp(suffix=".mp3")
            tts.save(path)

            if not save_path:
                # Try to play with playsound / os default player
                try:
                    import playsound  # type: ignore

                    playsound.playsound(path)
                except Exception:
                    pass
                finally:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                return TTSResult(text=text, backend="gtts")
            else:
                return TTSResult(text=text, backend="gtts", saved_to=path)
        except Exception as exc:
            _logger.warning("VoiceOutputAdapter (gtts): %s", exc)
            return TTSResult(
                text=text,
                backend="gtts",
                success=False,
                error=str(exc),
            )

    def __repr__(self) -> str:
        return (
            f"VoiceOutputAdapter("
            f"backend={self.active_backend!r}, "
            f"language={self.language!r})"
        )
