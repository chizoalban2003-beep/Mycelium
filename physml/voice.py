"""physml.voice — Voice interface with graceful degradation.

:class:`VoiceInterface` wraps speech recognition (STT) and text-to-speech
(TTS) into a simple ``listen()`` / ``speak()`` / ``run_loop()`` API.

Optional dependencies
---------------------
* ``speech_recognition`` — microphone input (``pip install SpeechRecognition``).
  Without it, :meth:`available` returns ``False`` and the loop falls back to
  ``input()``.
* ``pyttsx3`` — offline TTS (``pip install pyttsx3``).
  Without it, :meth:`speak` falls back to ``print()``.

Usage::

    from physml.llm import PromptSystem, ClaudeClient
    from physml.llm.action_dispatcher import ActionDispatcher
    from physml.voice import VoiceInterface

    ps = PromptSystem()
    dispatcher = ActionDispatcher()
    voice = VoiceInterface(prompt_system=ps, dispatcher=dispatcher, tts=True)

    if voice.available:
        voice.run_loop()          # blocking voice REPL
    else:
        print("No microphone library — running in text mode")
        voice.run_loop()          # falls back to text input
"""

from __future__ import annotations

from typing import Any, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


class VoiceInterface:
    """Voice-enabled REPL with graceful degradation.

    Parameters
    ----------
    prompt_system : PromptSystem or None
        Routes transcribed text to intents.  If ``None`` a new one is created.
    dispatcher : ActionDispatcher or None
        Executes the dispatched action.  If ``None`` a new one is created.
    tts : bool, default True
        Enable text-to-speech output when pyttsx3 is available.
    language : str, default "en-US"
        BCP-47 language tag for speech recognition.
    timeout : float, default 5.0
        Seconds to wait for speech before giving up (per phrase).
    phrase_limit : float, default 10.0
        Maximum seconds of audio to record per phrase.
    """

    def __init__(
        self,
        prompt_system: Any = None,
        dispatcher: Any = None,
        tts: bool = True,
        language: str = "en-US",
        timeout: float = 5.0,
        phrase_limit: float = 10.0,
    ) -> None:
        self.language = language
        self.timeout = timeout
        self.phrase_limit = phrase_limit
        self._tts_enabled = tts
        self._running = False

        # Lazy-resolve prompt system
        if prompt_system is not None:
            self._ps = prompt_system
        else:
            from physml.llm import PromptSystem
            self._ps = PromptSystem()

        # Lazy-resolve dispatcher
        if dispatcher is not None:
            self._dispatcher = dispatcher
        else:
            from physml.llm.action_dispatcher import ActionDispatcher
            self._dispatcher = ActionDispatcher()

        # Check for optional deps
        self._sr_available = self._check_sr()
        self._tts_engine = self._init_tts() if tts else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """``True`` when ``speech_recognition`` is installed."""
        return self._sr_available

    def listen(self) -> str:
        """Listen from the microphone and return transcribed text.

        Falls back to ``input()`` when speech_recognition is not available.

        Returns
        -------
        str
            Transcribed text, or empty string on failure.
        """
        if not self._sr_available:
            try:
                return input("you (text)> ").strip()
            except (EOFError, KeyboardInterrupt):
                return ""

        try:
            import speech_recognition as sr  # type: ignore
            recogniser = sr.Recognizer()
            with sr.Microphone() as source:
                print("Listening…", flush=True)
                recogniser.adjust_for_ambient_noise(source, duration=0.3)
                try:
                    audio = recogniser.listen(
                        source,
                        timeout=self.timeout,
                        phrase_time_limit=self.phrase_limit,
                    )
                except sr.WaitTimeoutError:
                    return ""
            text = recogniser.recognize_google(audio, language=self.language)
            print(f"you> {text}", flush=True)
            return str(text)
        except Exception as exc:
            _logger.debug("VoiceInterface.listen error: %s", exc)
            # Graceful fallback to text input on recognition error
            try:
                return input("you (text)> ").strip()
            except (EOFError, KeyboardInterrupt):
                return ""

    def speak(self, text: str) -> None:
        """Speak *text* aloud (or print it when TTS is unavailable).

        Parameters
        ----------
        text : str
            The text to speak.
        """
        print(f"myco> {text}", flush=True)
        if self._tts_engine is not None:
            try:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
            except Exception as exc:
                _logger.debug("VoiceInterface.speak TTS error: %s", exc)

    def run_loop(self) -> None:
        """Run a continuous voice interaction loop.

        Press Ctrl-C or say "exit" / "quit" to stop.
        Falls back to text input when speech_recognition is not available.
        """
        mode = "voice" if self._sr_available else "text"
        print(f"Mycelium voice interface ({mode} mode) — say 'exit' or Ctrl-C to stop.\n")
        self._running = True
        try:
            while self._running:
                text = self.listen()
                if not text:
                    continue
                if text.lower() in ("exit", "quit", "bye", "stop"):
                    self.speak("Goodbye!")
                    break

                action = self._ps.route(text)
                response = self._dispatcher.dispatch(action)
                self.speak(response)
                print()
        except KeyboardInterrupt:
            print("\nBye!")
        finally:
            self._running = False

    def run_once(self, text: str) -> str:
        """Process a single text prompt and return the response (no audio).

        Useful for testing.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        action = self._ps.route(text)
        return self._dispatcher.dispatch(action)

    def transcribe_text(self, text: str) -> str:
        """Process a text string as if it were spoken (for testing).

        Equivalent to :meth:`run_once` — routes *text* through the
        :class:`~physml.llm.prompt_system.PromptSystem` and
        :class:`~physml.llm.action_dispatcher.ActionDispatcher` without
        any audio I/O.

        Parameters
        ----------
        text : str
            The text to process.

        Returns
        -------
        str
            The dispatcher's plain-text response.
        """
        return self.run_once(text)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_sr(self) -> bool:
        try:
            import speech_recognition  # noqa: F401  # type: ignore
            return True
        except ImportError:
            return False

    def _init_tts(self) -> Optional[Any]:
        try:
            import pyttsx3  # type: ignore
            engine = pyttsx3.init()
            return engine
        except Exception as exc:
            _logger.debug("VoiceInterface: pyttsx3 not available: %s", exc)
            return None

    def __repr__(self) -> str:
        return (
            f"VoiceInterface(available={self.available}, "
            f"tts={self._tts_engine is not None}, "
            f"language={self.language!r})"
        )
