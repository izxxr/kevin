# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any

import threading
import sounddevice as sd

__all__ = (
    "TTSProvider",
)


class TTSProvider:
    """Base class for all text-to-speech providers.

    Currently, only PiperTTS is available however custom providers may
    be implemented using this class.
    """

    def is_speaking(self) -> bool:
        """Indicates if TTS provider is currently processing a text.

        This is used by assistant for pausing the no speech timeout while
        assistant is speaking allowing the user to listen to assistant.
        """
        raise NotImplementedError

    def speak(self, text: str, background: bool = True) -> None:
        """Synthesizes the provided text as speech and plays it.

        Parameters
        ----------
        text: :class:`str`
            The text to convert to speech.
        background: :class:`bool`
            Whether to speak in background. Defaults to true.
        """
        raise NotImplementedError


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
        self._lock = threading.Lock()
        self._speeches = 0

    def _signal_speech_start(self) -> None:
        with self._lock:
            self._speeches += 1

    def _signal_speech_end(self) -> None:
        with self._lock:
            self._speeches -= 1

    def _speak_worker(self, text: str) -> None:
        self._signal_speech_start()

        for chunk in self.voice.synthesize(text):
            sd.wait()
            sd.play(chunk.audio_float_array, chunk.sample_rate)

        sd.wait()
        self._signal_speech_end()

    def is_speaking(self) -> bool:
        return self._speeches > 0

    def speak(self, text: str, background: bool = True) -> None:
        if background:
            threading.Thread(target=self._speak_worker, args=(text,), daemon=True).start()
        else:
            self._speak_worker(text)
