# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import logging
import threading
import speech_recognition as sr

if TYPE_CHECKING:
    from kevin.stt import STTProvider, STTResult
    from kevin.inference import InferenceBackend

__all__ = (
    "Kevin",
)

_log = logging.getLogger(__name__)


class Kevin:
    """Core class providing high level interface for assistant.

    Parameters
    ----------
    stt: :class:`STTProvider`
        Speech-to-text provider for transcribing commands and follow up
        instructions.
    hotword_stt: :class:`STTProvider` | None
        Optional speech-to-text provider to use for detecting hotword.
        
        Generally, it is recommended to provide a STT provider that uses a
        smaller ASR model than :attr:`.stt` because this provider will be actively
        transcribing in background.

        If this is not provided, :attr:`.stt` is used for hotword detection as well.
    recognizer: :class:`speech_recognition.Recognizer` | None
        The recognizer instance used for taking microphone input. This argument may
        be used for setting a recognizer with customized configuration.

        If not provided, the default recognizer is used with suitable options.
    hotword_detect_func:
        Function called on speech when assistant is asleep to detect a hotword.

        The function returns True if a hotword is detected and if assistant
        should be woken up.

        For a better interface, it is recommended to use :meth:`.hotword_detect`
        decorator in most cases unless an existing function is to be set.
    sleep_on_done: bool
        Whether to sleep after command or action execution is done. This should
        generally never be set to false. Default is true.
    text_mode: :class:`bool`
        When enabled, the commands and responses are text based (standard I/O)
    """

    def __init__(
        self,
        inference: InferenceBackend,
        stt: STTProvider | None = None,
        hotword_stt: STTProvider | None = None,
        recognizer: sr.Recognizer | None = None,
        hotword_detect_func: Callable[[STTResult], bool | None] | None = None,
        sleep_on_done: bool = True,
        text_mode: bool = False,
    ):
        if not text_mode and stt is None:
            raise TypeError("stt must be provided when text_mode=False")

        if recognizer is None:
            recognizer = sr.Recognizer()

        if hotword_stt is None:
            hotword_stt = stt

        self.inference = inference
        self.stt = stt
        self.hotword_stt = hotword_stt
        self.recognizer = sr.Recognizer()
        self.hotword_detect_func = hotword_detect_func
        self.sleep_on_done = sleep_on_done
        self.text_mode = text_mode

        self._awake = threading.Event()
        self._started = False

    # command processing

    def _process_command(self, command: str):
        if not self.awake():
            self.wake_up()

        _log.info("Command: %r", command)

        if self.sleep_on_done:
            self.sleep()

    # speech processing

    def _process_audio(self, data: sr.AudioData) -> None:
        awake = self.awake()
        stt = self.stt if awake else self.hotword_stt

        assert stt is not None

        result = stt.transcribe(data, self)
        if not result.has_speech or not result.text:
            return

        _log.info(f"Speech received: {result.text!r}")

        if not awake:
            if self.hotword_detect_func is None:
                return

            if self.hotword_detect_func(result):
                _log.info("Received hotword, waking up")
                self.wake_up()

            return

    # Decorators

    def hotword_detect(self, func: Callable[[STTResult], bool | None] | None = None):
        """Sets the hotword detection function.

        The passed function should take one parameter, the :class:`STTResult`
        instance returned by :meth:`STTProvider.transcribe`.

        This can either be called directly to set a function e.g.::

            assistant.hotword_detect(func)

        ... or this method can be used as a decorator::

            @assistant.hotword_detect
            def func(result):
                return result.text.lower().startswith("hey kevin")
        """
        if func is not None:
            self.hotword_detect_func = func
            return func

        def __wrapper(func: Callable[[STTResult], bool | None]):
            self.hotword_detect_func = func
            return func

        return __wrapper
    
    # Awake state management

    def awake(self) -> bool:
        """Returns true if the assistant is awake."""
        return self._awake.is_set()
    
    def wake_up(self):
        """Wakes up the assistant."""
        self._awake.set()

    def sleep(self):
        """Puts the model into sleep state."""
        self._awake.clear()

    def wait_until_awake(self, timeout: float | None = None):
        """Blocks until the model has awaken.

        Returns immediately if model is already awake.

        Parameters
        ----------
        timeout: float | None
            If supplied, raises TimeoutError after the specified timeout duration in
            seconds. Defaults to None (no timeout).
        """
        self._awake.wait(timeout)

    # Assistant started state

    def _start_text_mode(self):
        while self._started:
            self._process_command(input("$$ > "))

    def _start_speech_mode(self):
        with sr.Microphone() as source:
            _log.info(f"Calibrating recognizer for ambient noise from device {source.device_index}")
            self.recognizer.adjust_for_ambient_noise(source)

            try:
                while self._started:
                    self._process_audio(self.recognizer.listen(source))
            except KeyboardInterrupt:
                self.stop()

    def start(self, setup_logging: bool = True) -> None:
        """Starts the assistant.

        This is a blocking method and does not return until the assistant
        has stopped.

        Parameters
        ----------
        setup_logging: :class:`bool`
            Whether to enable logs based on given logging configuration for the
            assistant. Default is true.
        """
        if self._started:
            raise RuntimeError("Assistant is already running")
        
        if setup_logging:
            logging.basicConfig(level=logging.INFO)

        _log.info("KEVIN is starting")
        self._started = True

        if self.text_mode:
            self._start_text_mode()
        else:
            self._start_speech_mode()

    def stop(self) -> None:
        """Stops the assistant.

        Not to be confused with awake state management, this method
        completely terminates the main loop.

        Returns without any operation if assistant is already stopped.
        """
        if not self._started:
            return

        _log.info("KEVIN is stopping")
        self._started = False
