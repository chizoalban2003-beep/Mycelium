"""Stage 125 — VoiceLoop: continuous voice interaction pipeline.

Wires :class:`~physml.voice_adapter.VoiceInputAdapter` (STT) and
:class:`~physml.voice_output.VoiceOutputAdapter` (TTS) into a real-time
listen → transcribe → companion.chat → speak loop.

Features
--------
* Continuous microphone listen with configurable record duration.
* Amplitude-based silence detection (skips empty audio).
* Optional wake-word gate: only process after hearing the wake phrase.
* Background-thread mode: ``start()`` / ``stop()`` for non-blocking use.
* Text-only fallback: ``run_once(text)`` processes a text prompt without audio.

Usage
-----
::

    from physml.voice_loop import VoiceLoop
    from physml.companion import MyceliumCompanion

    companion = MyceliumCompanion()
    companion.start()

    loop = VoiceLoop(
        companion=companion,
        wake_word="hey myco",
        record_seconds=5,
    )

    # Single turn
    loop.run_once_from_mic()

    # Continuous background mode
    loop.start()
    # ... user talks to the companion ...
    loop.stop()
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


class VoiceLoop:
    """Continuous voice interaction loop.

    Parameters
    ----------
    companion : MyceliumCompanion or None
        The companion that handles chat responses.
    wake_word : str or None
        If set, the loop only processes speech after detecting this phrase.
        Set to ``None`` to process every utterance.
    record_seconds : float, default 5.0
        How long to record for each utterance.
    silence_threshold : float, default 0.01
        Amplitude threshold below which audio is considered silence.
    speak_response : bool, default True
        If ``True``, speak the companion response using TTS.
    on_transcription : callable or None
        Optional callback ``fn(text)`` called with each transcription.
    on_response : callable or None
        Optional callback ``fn(text)`` called with each companion response.
    """

    def __init__(
        self,
        companion: Any = None,
        wake_word: Optional[str] = None,
        record_seconds: float = 5.0,
        silence_threshold: float = 0.01,
        speak_response: bool = True,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.companion = companion
        self.wake_word = wake_word.lower().strip() if wake_word else None
        self.record_seconds = record_seconds
        self.silence_threshold = silence_threshold
        self.speak_response = speak_response
        self.on_transcription = on_transcription
        self.on_response = on_response

        self._stt: Any = None
        self._tts: Any = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._wake_word_active = self.wake_word is None

        self._init_adapters()

    def _init_adapters(self) -> None:
        try:
            from physml.voice_adapter import VoiceInputAdapter

            self._stt = VoiceInputAdapter(preferred_backend="auto")
        except Exception as e:
            _logger.warning("VoiceLoop: STT init failed: %s", e)

        if self.speak_response:
            try:
                from physml.voice_output import VoiceOutputAdapter

                self._tts = VoiceOutputAdapter(preferred_backend="auto")
            except Exception as e:
                _logger.warning("VoiceLoop: TTS init failed: %s", e)

    # ------------------------------------------------------------------
    # Single-turn interfaces
    # ------------------------------------------------------------------

    def run_once(self, text: str) -> str:
        """Process a text prompt without audio.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
            The companion's response.
        """
        if self.companion is None:
            return "No companion connected."

        response = self.companion.chat(text)

        if self.on_response:
            self.on_response(response)

        if self.speak_response and self._tts is not None:
            try:
                self._tts.speak(response)
            except Exception as e:
                _logger.warning("VoiceLoop: TTS speak failed: %s", e)

        return response

    def run_once_from_mic(self) -> str:
        """Record from microphone, transcribe, and run through companion.

        Returns
        -------
        str
            The companion's response, or an empty string on silence/error.
        """
        if self._stt is None:
            _logger.warning("VoiceLoop: STT not available")
            return ""

        try:
            result = self._stt.record_from_microphone(duration=self.record_seconds)
            if not result.success or not result.text.strip():
                _logger.debug("VoiceLoop: silence or transcription failed")
                return ""

            text = result.text.strip()
            _logger.info("VoiceLoop transcribed: %r", text)

            if self.on_transcription:
                self.on_transcription(text)

            # Wake-word gate
            if self.wake_word and not self._wake_word_active:
                if self.wake_word in text.lower():
                    self._wake_word_active = True
                    _logger.info("VoiceLoop: wake word detected")
                    return self.run_once("Hello! I'm listening.")
                return ""

            return self.run_once(text)

        except Exception as exc:
            _logger.warning("VoiceLoop.run_once_from_mic: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the voice loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="VoiceLoop")
        self._thread.start()
        _logger.info("VoiceLoop started (wake_word=%r)", self.wake_word)

    def stop(self) -> None:
        """Stop the background voice loop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.record_seconds + 2)
        _logger.info("VoiceLoop stopped")

    def _loop(self) -> None:
        while self._running:
            try:
                self.run_once_from_mic()
            except Exception as exc:
                _logger.warning("VoiceLoop._loop error: %s", exc)
                time.sleep(0.5)

    @property
    def running(self) -> bool:
        """``True`` if the background loop is active."""
        return self._running

    @property
    def stt_backend(self) -> str:
        """Active speech-to-text backend name."""
        if self._stt is None:
            return "unavailable"
        return self._stt.active_backend

    @property
    def tts_backend(self) -> str:
        """Active text-to-speech backend name."""
        if self._tts is None:
            return "unavailable"
        return self._tts.active_backend

    def __repr__(self) -> str:
        return (
            f"VoiceLoop("
            f"running={self._running}, "
            f"stt={self.stt_backend!r}, "
            f"tts={self.tts_backend!r}, "
            f"wake_word={self.wake_word!r})"
        )
