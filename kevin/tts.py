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
        """Indicates if a speech is being synthesized or scheduled for synthesis."""
        raise NotImplementedError

    def speak(self, text: str, blocking: bool = False) -> None:
        """Synthesizes the provided text as speech and plays it.

        Parameters
        ----------
        text: :class:`str`
            The text to convert to speech.
        blocking: :class:`bool`
            Whether to synthesize text while blocking the caller
            thread. Defaults to false.

            This parameter set to false performs the TTS processing in
            background. If speak() is called while another speech synthesis
            is in progress, most implementations should wait for previous
            synthesis to finish before starting the new one.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Stops the ongoing speech.

        This method should only stop the ongoing speech. Any queued synthesis
        requests should not be canceled.
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
    def __init__(self, voice_path: str, *, voice_options: dict[str, Any] | None = None) -> None:
        try:
            from piper import PiperVoice
        except Exception:
            raise RuntimeError("PiperTTS requires piper-tts to be installed")

        if voice_options is None:
            voice_options = {}

        self.voice = PiperVoice.load(voice_path, **voice_options)
        self._lock = threading.Lock()
        self._stopper = threading.Event()
        self._speeches = 0

    def _speak_worker(self, text: str) -> None:
        self._speeches += 1

        with self._lock:
            self._stopper.clear()

            for chunk in self.voice.synthesize(text):
                if self._stopper.is_set():
                    break

                sd.wait()
                sd.play(chunk.audio_float_array, samplerate=chunk.sample_rate)

            # wait for last audio chunk to finish before releasing the lock
            sd.wait()
            self._speeches -= 1

    def speak(self, text: str, blocking: bool = False) -> None:
        if blocking:
            self._speak_worker(text)
        else:
            threading.Thread(
                target=self._speak_worker,
                args=(text,),
                daemon=True
            ).start()

    def stop(self) -> None:
        self._stopper.set()
        sd.stop()

    def is_speaking(self) -> bool:
        return self._speeches > 0
