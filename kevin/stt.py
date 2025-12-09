# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from dataclasses import dataclass

from speech_recognition.recognizers.whisper_local import faster_whisper as fw_recognizer
from speech_recognition import AudioData

import faster_whisper

if TYPE_CHECKING:
    from kevin.assistant import Kevin

__all__ = (
    "STTProvider",
    "STTResult",
    "FasterWhisperSTT",
)


@dataclass
class STTResult:
    """Result of a single speech-to-text transcription.

    Attributes
    ----------
    text: :class:`str`
        The transcribed text.
    has_speech: :class:`bool`
        Indicates if the result has speech. This is set (or not set) by individual STT
        providers and may not always reflect the truth.

        By default, if STT provider does not set this, this is true.
    state:
        Data injected by specific STT provider that produced the transcription.
    """

    text: str
    has_speech: bool = True
    state: Any = None


class STTProvider:
    """Base class that all STT providers are built upon.

    All children must implement the :meth:`.transcribe` method. For now, only
    faster-whisper is used.

    This class only exists for future proofing purposes in case other STT options
    become available or if we end up using some other alternative such as WhisperCPP
    or Vosk.
    """

    def transcribe(self, data: AudioData, assistant: Kevin) -> STTResult:
        """Transcribes the given audio data.

        .. note::

            For strong typing, this method must return the transcription result as
            :class:`STTResult` dataclass. To attach any additional information returned
            by the underlying ASR provider, use the :attr:`STTResult.state` attribute.

        Parameters
        ----------
        data: :class:`speech_recognition.AudioData`
            The audio data to transcribe.
        assistant: :class:`Kevin`
            The assistant that called this method.

        Returns
        -------
        :class:`STTResult`
            The result of transcription.
        """
        raise NotImplementedError


class FasterWhisperSTT(STTProvider):
    """Speech recognition using the faster-whisper implementation.

    Parameters
    ----------
    model_size_or_path:
        The name or path of Whisper model to use. This argument is simply
        equivalent to first argument of faster whisper model class.
    whisper_model_options: dict
        Additional keyword arguments for :class:`faster_whisper.WhisperModel`

        Note that ``model_size_or_path`` is ignored in these options. Use the first
        argument instead.
    no_speech_thresh: :class:`bool`
        The minimum threshold to classify a transcript as speech. This is used to
        set :attr:`STTResult.has_speech` attribute.

        If any segment has a threshold higher than this, then the result is considered
        to have a speech.
    """

    def __init__(
        self,
        model_size_or_path: str,
        no_speech_thresh: float = 0.5,
        whisper_model_options: dict[str, Any] | None = None,
    ):
        if whisper_model_options is None:
            whisper_model_options = {}

        whisper_model_options.pop("model_size_or_path", None)

        # The standard Recognizer.recognize_faster_whisper() creates the WhisperModel
        # instance on each call which is extremely slow so I have to use the internals
        # of speech_recognition.
        self.model = faster_whisper.WhisperModel(model_size_or_path, **whisper_model_options)
        self.recognizer = fw_recognizer.WhisperCompatibleRecognizer(
            fw_recognizer.TranscribableAdapter(self.model)  # type: ignore
        )
        self.no_speech_thresh = no_speech_thresh

    def transcribe(self, data: AudioData, assistant: Kevin) -> STTResult:
        """Transcribes the audio data.

        The :attr:`STTResult.state` attribute is the raw transcription
        result data containing:

        - `segments`: The :class:`faster_whisper.Segment` objects.
        - `language`: A string representing the language of text.
        """
        transcript = self.recognizer.recognize(data, show_dict=True)

        if not isinstance(transcript, dict):
            raise TypeError(f"recognize() did not return a dictionary; got {type(transcript)}")

        result = STTResult(text=transcript.pop("text").strip(), state=transcript)
        result.has_speech = any(s.no_speech_prob < self.no_speech_thresh for s in transcript["segments"])

        return result
