# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from kevin.data import Message, InferenceChatResponse
from kevin.defs import DEFAULT_SYSTEM_PROMPT, DEFAULT_VARIATION_SYSTEM_PROMPT
from kevin.utils.plugins import PluginsMixin
from kevin.tools.context import ToolCallContext

import logging
import threading
import collections
import sounddevice as sd
import speech_recognition as sr

if TYPE_CHECKING:
    from kevin.stt import STTProvider
    from kevin.tts import TTSProvider
    from kevin.inference import InferenceBackend
    from kevin.hotwords import HotwordDetector
    from kevin.tools.base import Tool

__all__ = (
    "Kevin",
)

_log = logging.getLogger(__name__)


class Kevin(PluginsMixin):
    """Core class providing high level interface for assistant.

    Parameters
    ----------
    stt: :class:`STTProvider`
        Speech-to-text provider for transcribing commands and follow up
        instructions. If not provided, text mode is enabled and prompts
        are given through standard input.
    tts: :class:`TTSProvider`
        The text-to-speech provider to use. If no provider is supplied, the responses
        are only logged.
    recognizer: :class:`speech_recognition.Recognizer` | None
        The recognizer instance used for taking microphone input. This argument may
        be used for setting a recognizer with customized configuration.

        If not provided, the default recognizer is used with suitable options.
    microphone: :class:`speech_recognition.Microphone` | None
        The microphone to take speech input from. This can be provided if a specific
        audio input device is to be used.

        If not provided, the default microphone is used.
    hotword_detect_func:
        Function called on speech when assistant is asleep to detect a hotword.

        The function returns True if a hotword is detected and if assistant
        should be woken up.

        For a better interface, it is recommended to use :meth:`.hotword_detect`
        decorator in most cases unless an existing function is to be set.
    sleep_on_done: :class:`bool`
        Whether to sleep after command or action execution is done. This should
        generally never be set to false. Default is true.
    system_prompts: list[:class:`str`] | None
        The system prompts to provide to language model.

        If `include_default_prompt` is true (default), the default system prompt
        (defined in `kevin.data.DEFAULT_SYSTEM_PROMPT`) will be included in the
        set of system prompts.
    include_default_prompt: :class:`bool`
        Whether to include the default system prompt in given system prompts. Defaults
        to true. If false, `assistant_name` and `user_name` are disregarded.
    assistant_name: :class:`str`
        The custom name of assistant. Defaults to KEVIN. This is only formatted into
        default system prompt and disregarded when providing `include_default_prompt=False`.
    user_name: :class:`str`
        The name of user. Defaults to generic User. This is only formatted into
        default system prompt and disregarded when providing `include_default_prompt=False`.
    max_history_messages: :class:`int`
        The maximum messages to remember as history. This is exclusive of any system prompts and
        only inclusive of the assistant/user messages. Default is 5.
    listen_timeout: :class:`float`
        The number of seconds to wait before sleeping if no speech is detected after hotword. Defaults
        to 10 seconds. Setting this to None removes timeout.
    """

    def __init__(
        self,
        inference: InferenceBackend,
        stt: STTProvider | None = None,
        recognizer: sr.Recognizer | None = None,
        microphone: sr.Microphone | None = None,
        tts: TTSProvider | None = None,
        hotword_detector: HotwordDetector | None = None,
        sleep_on_done: bool = True,
        system_prompts: list[str] | None = None,
        include_default_prompt: bool = True,
        assistant_name: str = "KEVIN",
        user_name: str = "<unnamed>",
        max_history_messages: int = 5,
        listen_timeout: float | None = 5.0,
    ):
        # TODO: add support for other wake-up mechanisms e.g. push-to-talk
        if stt is not None and hotword_detector is None:
            raise TypeError("hotword_detector must be provider if stt is set")

        if recognizer is None:
            recognizer = sr.Recognizer()

            if listen_timeout is not None:
                # This is usually needed for listen_timeout to work properly.
                recognizer.dynamic_energy_threshold = False

        if system_prompts is None:
            system_prompts = []

        if include_default_prompt:
            system_prompts.insert(0, self._get_default_system_prompt(assistant_name, user_name))

        self.inference = inference
        self.stt = stt
        self.recognizer = recognizer
        self.microphone = microphone if microphone is not None else sr.Microphone()
        self.tts = tts
        self.hotword_detector = hotword_detector
        self.sleep_on_done = sleep_on_done
        self.text_input_mode = stt is None
        self.text_output_mode = tts is None
        self.listen_timeout = listen_timeout
        self.system_prompts = [Message(role="system", content=prompt) for prompt in system_prompts]

        self._tools = {}
        self._plugins = {}
        self._name = "main"
        self._tools_data = None
        self._awake = threading.Event()
        self._started = False
        self._history = collections.deque[Message](maxlen=max_history_messages)

    # command processing

    def _get_messages(self, command: str):
        msg = Message(role="user", content=command)
        self._history.append(msg)

        copy = self.system_prompts.copy()
        copy.extend(self._history)
        copy.append(msg)

        return copy
 
    def _get_default_system_prompt(self, assistant_name: str, user_name: str) -> str:
        return DEFAULT_SYSTEM_PROMPT.format(assistant_name=assistant_name, user_name=user_name)

    def _get_tools_data(self) -> list[dict[str, Any]]:
        if self._tools_data is None:
            self._tools_data = [t.dump() for t in self._tools.values()]

        return self._tools_data
    
    def _make_call_context(self, tool: Tool) -> ToolCallContext:
        return ToolCallContext(
            assistant=self,
            tool=tool,
        )
    
    def _call_tools_from_response(self, response: InferenceChatResponse):
        for call in response.tool_calls:
            try:
                tool_tp = self._tools[call.name]
            except KeyError:
                _log.warning("Inference response contains invalid tool name")
            else:
                tool = tool_tp(**call.arguments)
                ctx = self._make_call_context(tool)
                ctx._call()

    def process_command(self, command: str):
        """Processes the given command.

        This method processes the command through inference backend
        and handles any tool calls in the response.

        This API is only exposed for niche or complex cases and should
        almost never be called manually.

        Parameters
        ----------
        command: :class:`str`
            The command to process.
        """
        _log.info("Command: %r", command)

        response = self.inference.chat(
            messages=self._get_messages(command),
            tools_data=self._get_tools_data(),
        )

        _log.info("Response: %r", response.content)

        if response.content and self.tts:
            self.tts.speak(response.content)
            self._history.append(Message(role="assistant", content=response.content))

        self._call_tools_from_response(response)

    # speech processing

    def _process_speech(self, data: sr.AudioData) -> None:
        if self.stt is None or not self.awake():
            return

        result = self.stt.transcribe(data, self)

        if not result.has_speech or not result.text:
            return

        _log.info(f"Speech received: {result.text!r}")
        self.process_command(result.text)

        # XXX: Currently, the assistant immediately sleeps after processing the command.
        # Behavior to wait for a follow up is easy to implement. However, that involves
        # not sleeping after processing the command i.e. assistant is still listening.
        # This leads to echo of assistant response being recognized as speech essentially
        # leading to self talking loop. One possible approach is to pause STT while assistant
        # is speaking and keeping hotword detector active. Once hotword is detected, stop the
        # TTS (silence assistant) and move to normal listen-process cycle. However, this solution
        # is slightly hacky and importantly, requires adjusting recognizer and hotword detector
        # to be able to recognize the hotword while assistant voice's noise is present in
        # background. Further more (TODO) TTSProvider currently does not provider a proper
        # 'silencing' interface. I would rather prefer implementing that first.
        # For now, we are limited to require repeated wake up calls until a feasible solution
        # is available.
        if self.sleep_on_done:
            self.sleep()

    def _wake_assistant(self) -> None:
        if self.hotword_detector is None:
            return

        spec = self.hotword_detector.get_audio_spec()

        with sd.InputStream(
            samplerate=spec.sample_rate,
            blocksize=spec.frame_size,
            dtype=spec.dtype,
            channels=spec.channels,
            device=self.microphone.device_index,
        ) as stream:
            while not self.awake():
                data, _ = stream.read(spec.frame_size)

                if not self.hotword_detector.process(data):
                    continue

                # stop ongoing TTS speeches
                while self.tts and self.tts.is_speaking():
                    self.tts.stop()

                self.wake_up()
                self.hotword_detector.reset()

    def _listen_speech(self, source: sr.AudioSource) -> None:
        if not self.awake() and self.hotword_detector is not None:
            self._wake_assistant()
        else:
            try:
                speech = self.recognizer.listen(source, timeout=self.listen_timeout)
            except sr.WaitTimeoutError:
                self.sleep()
            else:
                self._process_speech(speech)

    # Text mode

    def _read_input(self) -> None:
        command = input("$$ > ")

        # stop ongoing TTS speeches
        while self.tts and self.tts.is_speaking():
            self.tts.stop()

        self.process_command(command)

    # Awake state management

    def awake(self) -> bool:
        """Returns true if the assistant is awake.

        This is only used in speech mode. If assistant is awake,
        microphone input is processed by speech-to-text provider.
        """
        return self._awake.is_set()
    
    def wake_up(self):
        """Wakes up the assistant.

        This is only used in speech mode. If assistant is awake,
        microphone input is processed by speech-to-text provider.
        """
        _log.info("Waking up assistant.")
        self._awake.set()

    def sleep(self):
        """Puts the model into sleep state.

        This is only used in speech mode. If assistant is in sleep state,
        speech-to-text is disabled and microphone input is processed through
        hotword detector to wake the assistant.
        """
        _log.info("Assistant going to sleep.")
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
            self._read_input()

    def _start_speech_mode(self):
        with self.microphone as source:
            _log.info(f"Calibrating recognizer for ambient noise from device {source.device_index}")
            self.recognizer.adjust_for_ambient_noise(source)

            try:
                while self._started:
                    self._listen_speech(source)
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

        if self.text_input_mode:
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

    # convenience methods

    def say(
        self,
        content: str,
        speak: bool = True,
        background: bool = True,
        history_role: str = "assistant",
    ) -> None:
        """Says the given message.

        Parameters
        ----------
        content: :class:`str`
            The content of message to say.
        speak: :class:`bool`
            Whether to say the message through TTS if available. Defaults to True.
            If no TTS is available or this is set to false, the message is simply logged.
        background: :class:`bool`
            If speakable message, whether to speak the message in background (non-blocking).
            Defaults to true.
        history_role: :class:`str` | None
            The role with which the message is saved in history. Defaults to assistant.
            If set to None, the message is not added to history.
        """
        if speak and self.tts:
            self.tts.speak(content, background=background)
        else:
            if history_role:
                output = f"Message ({history_role}): {content}"
            else:
                output = f"Message: {content}"

            _log.info(output)

        if history_role is not None:
            self.add_message_to_history(history_role, content)

    # Chat history management

    def purge_history(self) -> None:
        """Clears the message history of assistant."""
        self._history.clear()

    def get_history(self) -> collections.deque[Message]:
        """collections.deque object for assistant's messages history.

        .. warning::

            This returns the underlying history as-is instead of a copy.
        """
        return self._history

    def add_message_to_history(self, role: str, content: str) -> Message:
        """Adds a message to assistant's chat history.

        Parameters
        ----------
        role: :class:`str`
            The role that the message represents. This should be "assistant"
            for representing messages from assistant or "user" for user originating
            messages.
        content: :class:`str`
            The content of message to add.

        Returns
        -------
        :class:`Message`
            The created message.
        """
        m = Message(role=role, content=content)
        self._history.append(m)

        return m
