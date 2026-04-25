"""Stage 111 — VoiceInputAdapter: voice-to-text transcription.

Primary backend: ``whisper-cpp-python`` (optional, graceful fallback).
Secondary backend: ``speech_recognition`` library (also optional).
When neither is available, operates in text-passthrough mode.

Returns a :class:`VoiceResult` with transcribed text, confidence, and
backend used.

Usage
-----
::

    from physml.voice_adapter import VoiceInputAdapter

    adapter = VoiceInputAdapter()
    result = adapter.transcribe_file("audio.wav")
    print(result.text)       # transcribed text
    print(result.backend)    # "whisper" | "speechrecognition" | "passthrough"
    print(result.confidence)

    # Text passthrough (for testing / fallback)
    result = adapter.from_text("hello world")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


@dataclass
class VoiceResult:
    """Result of a voice transcription.

    Attributes
    ----------
    text : str
        Transcribed (or passed-through) text.
    confidence : float
        Confidence score in [0, 1] (1.0 for passthrough / unknown).
    backend : str
        ``"whisper"``, ``"speechrecognition"``, or ``"passthrough"``.
    metadata : dict
        Backend-specific metadata.
    success : bool
        ``True`` if transcription succeeded.
    error : str or None
        Error message when *success* is ``False``.
    """

    text: str
    confidence: float = 1.0
    backend: str = "passthrough"
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class VoiceInputAdapter:
    """Voice-to-text adapter with graceful backend fallback.

    Parameters
    ----------
    preferred_backend : str, default "auto"
        ``"whisper"``, ``"speechrecognition"``, ``"passthrough"``, or
        ``"auto"`` (tries in order).
    language : str, default "en"
        Language code for transcription.
    """

    def __init__(
        self,
        preferred_backend: str = "auto",
        language: str = "en",
    ) -> None:
        self.preferred_backend = preferred_backend
        self.language = language
        self._available_backends = self._detect_backends()
        _logger.info(
            "VoiceInputAdapter: available backends: %s", self._available_backends
        )

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    def _detect_backends(self) -> list[str]:
        backends: list[str] = []
        # faster-whisper: best local option (GPU or CPU, very fast)
        if self.preferred_backend in ("faster_whisper", "auto"):
            try:
                import faster_whisper  # type: ignore  # noqa: F401

                backends.append("faster_whisper")
            except ImportError:
                pass
        # openai-whisper: the original Python package
        if self.preferred_backend in ("openai_whisper", "auto"):
            try:
                import whisper  # type: ignore  # noqa: F401

                backends.append("openai_whisper")
            except ImportError:
                pass
        # whispercpp: C++ bindings (lower RAM)
        if self.preferred_backend in ("whisper", "auto"):
            try:
                import whispercpp  # type: ignore  # noqa: F401

                backends.append("whisper")
            except ImportError:
                pass
        # speech_recognition: cloud-backed fallback
        if self.preferred_backend in ("speechrecognition", "auto"):
            try:
                import speech_recognition  # type: ignore  # noqa: F401

                backends.append("speechrecognition")
            except ImportError:
                pass
        backends.append("passthrough")
        return backends

    @property
    def active_backend(self) -> str:
        """The backend that will be used for transcription."""
        if self.preferred_backend != "auto" and self.preferred_backend in self._available_backends:
            return self.preferred_backend
        return self._available_backends[0]

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe_file(self, audio_path: str) -> VoiceResult:
        """Transcribe an audio file.

        Parameters
        ----------
        audio_path : str
            Path to a WAV/MP3/FLAC audio file.

        Returns
        -------
        VoiceResult
        """
        backend = self.active_backend

        if backend == "faster_whisper":
            return self._faster_whisper_transcribe(audio_path)
        elif backend == "openai_whisper":
            return self._openai_whisper_transcribe(audio_path)
        elif backend == "whisper":
            return self._whisper_transcribe(audio_path)
        elif backend == "speechrecognition":
            return self._sr_transcribe(audio_path)
        else:
            _logger.warning(
                "VoiceInputAdapter: no real backend available; "
                "returning passthrough for %s",
                audio_path,
            )
            return VoiceResult(
                text=f"[passthrough: {audio_path}]",
                confidence=0.0,
                backend="passthrough",
                success=False,
                error="No speech recognition backend installed",
            )

    def from_text(self, text: str) -> VoiceResult:
        """Pass *text* through without transcription (testing / fallback).

        Parameters
        ----------
        text : str

        Returns
        -------
        VoiceResult
        """
        return VoiceResult(
            text=text,
            confidence=1.0,
            backend="passthrough",
            metadata={"source": "text_input"},
        )

    def record_from_microphone(
        self,
        duration: float = 5.0,
        save_path: Optional[str] = None,
    ) -> VoiceResult:
        """Record from the default microphone and transcribe.

        Parameters
        ----------
        duration : float
            Recording duration in seconds.
        save_path : str or None
            If given, save the recorded WAV to this path.

        Returns
        -------
        VoiceResult
        """
        try:
            import sounddevice as sd  # type: ignore
            import wave
            import tempfile
            import os

            sample_rate = 16000
            _logger.info("VoiceInputAdapter: recording %.1fs from microphone…", duration)
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()

            tmp = save_path or tempfile.mktemp(suffix=".wav")
            with wave.open(tmp, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio.tobytes())

            result = self.transcribe_file(tmp)

            if save_path is None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

            return result

        except ImportError:
            return VoiceResult(
                text="",
                confidence=0.0,
                backend="passthrough",
                success=False,
                error="sounddevice not installed — pip install sounddevice",
            )
        except Exception as exc:
            _logger.warning("VoiceInputAdapter.record_from_microphone: %s", exc)
            return VoiceResult(
                text="",
                confidence=0.0,
                backend="passthrough",
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _faster_whisper_transcribe(self, audio_path: str) -> VoiceResult:
        try:
            from faster_whisper import WhisperModel  # type: ignore

            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments, info = model.transcribe(audio_path, language=self.language)
            text = " ".join(seg.text for seg in segments).strip()
            return VoiceResult(
                text=text,
                confidence=0.93,
                backend="faster_whisper",
                metadata={"language": info.language, "model": "tiny"},
            )
        except Exception as exc:
            _logger.warning("VoiceInputAdapter (faster_whisper): %s", exc)
            return VoiceResult(
                text="",
                confidence=0.0,
                backend="faster_whisper",
                success=False,
                error=str(exc),
            )

    def _openai_whisper_transcribe(self, audio_path: str) -> VoiceResult:
        try:
            import whisper  # type: ignore

            model = whisper.load_model("tiny")
            result = model.transcribe(audio_path, language=self.language)
            text = result.get("text", "").strip()
            return VoiceResult(
                text=text,
                confidence=0.92,
                backend="openai_whisper",
                metadata={"model": "tiny"},
            )
        except Exception as exc:
            _logger.warning("VoiceInputAdapter (openai_whisper): %s", exc)
            return VoiceResult(
                text="",
                confidence=0.0,
                backend="openai_whisper",
                success=False,
                error=str(exc),
            )

    def _whisper_transcribe(self, audio_path: str) -> VoiceResult:
        try:
            import whispercpp  # type: ignore

            w = whispercpp.Whisper.from_pretrained("tiny")
            result = w.transcribe(audio_path)
            text = result if isinstance(result, str) else str(result)
            return VoiceResult(
                text=text.strip(),
                confidence=0.9,
                backend="whisper",
                metadata={"model": "tiny"},
            )
        except Exception as exc:
            _logger.warning("VoiceInputAdapter (whisper): %s", exc)
            # Fall through to passthrough
            return VoiceResult(
                text="",
                confidence=0.0,
                backend="whisper",
                success=False,
                error=str(exc),
            )

    def _sr_transcribe(self, audio_path: str) -> VoiceResult:
        try:
            import speech_recognition as sr  # type: ignore

            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language=self.language)
            return VoiceResult(
                text=text.strip(),
                confidence=0.85,
                backend="speechrecognition",
                metadata={"engine": "google"},
            )
        except Exception as exc:
            _logger.warning("VoiceInputAdapter (speechrecognition): %s", exc)
            return VoiceResult(
                text="",
                confidence=0.0,
                backend="speechrecognition",
                success=False,
                error=str(exc),
            )

    def __repr__(self) -> str:
        return (
            f"VoiceInputAdapter("
            f"backend={self.active_backend!r}, "
            f"language={self.language!r})"
        )
