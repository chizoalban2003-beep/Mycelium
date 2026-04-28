"""physml.health — Health-check for optional dependencies and project version.

:func:`check` returns a dict reporting whether key optional dependencies are
importable, plus the current physml version.  Use it to diagnose a missing
dependency before filing a bug report.

Usage::

    from physml.health import check

    status = check()
    # {
    #   "anthropic": True,
    #   "scipy": True,
    #   "pandas": True,
    #   "speech_recognition": False,
    #   "pyttsx3": False,
    #   "version": "0.32.2",
    # }
"""

from __future__ import annotations


def check() -> dict:
    """Return a health-check dict for optional dependencies and physml version.

    Returns
    -------
    dict
        Keys:

        * ``anthropic`` (bool) — Anthropic SDK importable.
        * ``scipy`` (bool) — SciPy importable.
        * ``pandas`` (bool) — pandas importable.
        * ``speech_recognition`` (bool) — SpeechRecognition importable.
        * ``pyttsx3`` (bool) — pyttsx3 importable.
        * ``whisper`` (bool) — OpenAI Whisper importable (offline STT).
        * ``sounddevice`` (bool) — sounddevice importable (required by Whisper STT).
        * ``version`` (str) — current physml version string.
    """
    def _importable(name: str) -> bool:
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    from physml import __version__

    return {
        "anthropic": _importable("anthropic"),
        "scipy": _importable("scipy"),
        "pandas": _importable("pandas"),
        "speech_recognition": _importable("speech_recognition"),
        "pyttsx3": _importable("pyttsx3"),
        "whisper": _importable("whisper"),
        "sounddevice": _importable("sounddevice"),
        "version": __version__,
    }
