# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any

import io
import sounddevice as sd
import soundfile as sf

__all__ = (
    "TTSProvider",
)


class TTSProvider:
    """Base class for all text-to-speech providers.

    Currently, only PiperTTS is available however custom providers may
    be implemented using this class.
    """

    def speak(self, text: str) -> None:
        """Synthesizes the provided text as speech and plays it.

        Parameters
        ----------
        text: :class:`str`
            The text to convert to speech.
        """


class PiperTTS(TTSProvider):
    """Piper-based TTS provider.

    This requires piper-tts to be installed.

    Parameters
    ----------
    voice_path: :class:`str`
        The path to voice to use.
    voice_options:
        Additional options to pass to PiperVoice constructor.
    """
    def __init__(self, voice_path: str, voice_options: dict[str, Any] | None = None) -> None:
        try:
            from piper import PiperVoice
        except Exception:
            raise RuntimeError("PiperTTS requires piper-tts to be installed")
        if voice_options is None:
            voice_options = {}

        self.voice = PiperVoice.load(voice_path, **voice_options)

    def speak(self, text: str) -> None:
        for chunk in self.voice.synthesize(text):
            sd.play(chunk.audio_float_array, chunk.sample_rate)
            sd.wait()
