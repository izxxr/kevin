# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any, Sequence, TYPE_CHECKING

import dataclasses
import pyaudio as pa
import numpy as np

__all__ = (
    "HotwordDetector",
    "HotwordAudioSpec",
    "PorcupineHotwordDetector",
)


@dataclasses.dataclass
class HotwordAudioSpec:
    """Data class for defining the audio spec for hotword detection.

    The instance of this class should be returned by :meth:`HotwordDetector.get_audio_spec`
    method. See its documentation for more details.

    Attributes
    ----------
    sample_rate: :class:`int`
        The audio sample rate. Defaults to 16000.
    channels: :class:`int`
        Number of audio channels. Defaults to 1.
    dtype: :class:`str`
        The (numpy) data type to read audio data in. Defaults to float32.
    frame_size: :class:`int` | None
        The size of a single audio frame. Defaults to None (arbitrary sized frame).
    """
    sample_rate: int = 16_000
    channels: int = 1
    dtype: str = "float32"
    frame_size: int | None = None


class HotwordDetector:
    """Base class for hotword detectors.

    This class provides a common interface for implementing hotword detection
    using various hotword detection engines.

    For now, Kevin only provides :class:`PorcupineHotwordDetector` however
    this class may be extended to implement support for another engine. All
    subclasses must implement the :meth:`.process` method.
    """

    def get_audio_spec(self) -> HotwordAudioSpec:
        """Returns the audio spec definition used by the detector.

        This method should be overridden when the underlying hotword
        detection engine constrains the audio data to follow specific
        spec such as fixed frame length or sample rate.

        This method returns a :class:`HotwordAudioSpec` instance defining
        the format for reading audio data. By default, this method returns
        the audio spec with default values.
        """
        return HotwordAudioSpec()

    def reset(self) -> None:
        """Resets the internal state.

        This method is called by assistant at the end of a
        hotword cycle and could be used for clearing up any
        underlying audio buffer, if any.

        By default, this method does not do anything.
        """

    def process(self, data: np.ndarray[np._AnyShapeT, np.dtype[Any]]) -> bool:
        """Processes the audio data.

        This method processes the received audio data through any underlying
        detection engine and returns True if a hotword is detected.

        Parameters
        ----------
        data:
            The (numpy) representing the audio frame to be processed returned
            by :meth:`sounddevice.InputStream.read` method.
        """
        raise NotImplementedError


class PorcupineHotwordDetector(HotwordDetector):
    """Hotword detection using Porcupine engine.

    This hotword detector requires ``pvporcupine`` library to be installed.

    Parameters
    ----------
    access_key: :class:`str`
        The Porcupine access key.
    keywords: Sequence[:class:`str`]
        The list of hotwords supported by Porcupine.
    keyword_paths: Sequence[:class:`str`]
        The list of paths for custom keywords.
    porcupine_options: Dict[:class:`str`, Any]
        Additional options passed to :func:`pvporcupine.create` function.
    """

    def __init__(
        self,
        access_key: str | None = None,
        keywords: Sequence[str] | None = None,
        keyword_paths: Sequence[str] | None = None,
        porcupine_options: dict[str, Any] | None = None,
    ):
        try:
            import pvporcupine
        except ImportError:
            raise RuntimeError("pvporcupine must be installed to use PorcupineHotwordDetector")
        
        if porcupine_options is None:
            porcupine_options = {}

        porcupine_options.setdefault("access_key", access_key)
        porcupine_options.setdefault("keywords", keywords)
        porcupine_options.setdefault("keyword_paths", keyword_paths)

        self._porcupine = pvporcupine.create(**porcupine_options)

    def get_audio_spec(self) -> HotwordAudioSpec:
        return HotwordAudioSpec(
            sample_rate=self._porcupine.sample_rate,
            channels=1,
            dtype="int16",
            frame_size=self._porcupine.frame_length,
        )

    def process(self, data: np.ndarray[np._AnyShapeT, np.dtype[Any]]) -> bool:
        return self._porcupine.process(data[:, 0]) >= 0  # type: ignore
