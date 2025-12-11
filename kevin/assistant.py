# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Callable, Any, Mapping, TYPE_CHECKING
from types import MappingProxyType
from kevin.data import Message, InferenceChatResponse
from kevin.defs import DEFAULT_SYSTEM_PROMPT
from kevin.tools import Tool

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
    sleep_on_done: :class:`bool`
        Whether to sleep after command or action execution is done. This should
        generally never be set to false. Default is true.
    text_mode: :class:`bool`
        When enabled, the commands and responses are text based (standard I/O)
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
        system_prompts: list[str] | None = None,
        include_default_prompt: bool = True,
        assistant_name: str = "KEVIN",
        user_name: str = "<unnamed>"
    ):
        if not text_mode and stt is None:
            raise TypeError("stt must be provided when text_mode=False")

        if recognizer is None:
            recognizer = sr.Recognizer()

        if hotword_stt is None:
            hotword_stt = stt

        if system_prompts is None:
            system_prompts = []

        if include_default_prompt:
            system_prompts.insert(0, self._get_default_system_prompt(assistant_name, user_name))

        self.inference = inference
        self.stt = stt
        self.hotword_stt = hotword_stt
        self.recognizer = sr.Recognizer()
        self.hotword_detect_func = hotword_detect_func
        self.sleep_on_done = sleep_on_done
        self.text_mode = text_mode
        self.system_prompts = [Message(role="system", content=prompt) for prompt in system_prompts]

        self._tools: dict[str, type[Tool]] = {}
        self._tools_data = None
        self._awake = threading.Event()
        self._started = False

    # command processing

    def _get_messages(self, command: str):
        copy = self.system_prompts.copy()
        copy.append(Message(role="user", content=command))
        return copy
    
    def _get_default_system_prompt(self, assistant_name: str, user_name: str) -> str:
        return DEFAULT_SYSTEM_PROMPT.format(assistant_name=assistant_name, user_name=user_name)

    def _get_tools_data(self) -> list[dict[str, Any]]:
        if self._tools_data is None:
            self._tools_data = [t.dump() for t in self._tools.values()]

        return self._tools_data
    
    def _call_tools_from_response(self, response: InferenceChatResponse):
        for call in response.tool_calls:
            tool_tp = self._tools[call.name]
            tool_tp(**call.arguments).callback(self)

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
        if not self.awake():
            self.wake_up()

        _log.info("Command: %r", command)

        response = self.inference.chat(
            messages=self._get_messages(command),
            tools_data=self._get_tools_data(),
        )

        _log.info("Response: %r", response.content)

        self._call_tools_from_response(response)

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
        
        self.process_command(result.text)

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

    def tool(self, tool_tp: type[Tool] | None = None):
        """Registers a decorated class as tool.

        This is a decorator based interface for :meth:`.add_tool`. Example
        usage::

            @assistant.tool()
            class CheckWeather(kevin.tools.Function):
                '''Checks weather of the given location.'''
                location: str

                def callback(self, assistant):
                    # ... some meaningful weather fetching code ...
                    print(f"Weather at {self.location} is sunny")
        """
        if tool_tp is not None:
            self.add_tool(tool_tp)
            return tool_tp

        def __wrapper(tool_tp: type[Tool]):
            self.add_tool(tool_tp)
            return tool_tp

        return __wrapper

    # Tools management

    def add_tool(self, tool: type[Tool], override: bool = False) -> None:
        """Registers a tool that can be called by the assistant.

        Parameters
        ----------
        tool: type[:class:`Tool`]
            The tool to add.
        override: :class:`bool`
            Whether to override the tool if an existing one is registered with the
            same name. Defaults to False.

            An error is raised if this is false and a tool is being added that has
            the same name as one already added.
        """
        if self._tools.get(tool.__tool_name__) is not None and not override:
            raise ValueError(f"Tool with name {tool.__tool_name__!r} already registered")

        self._tools[tool.__tool_name__] = tool

    def remove_tool(self, tool: type[Tool] | str, raise_error: bool = True) -> None:
        """Removes an already registered tool.

        Parameters
        ----------
        tool: type[:class:`Tool`]
            The tool's name or the tool class.
        raise_error: :class:`bool`
            If true (default), raise an error if tool to be removed does not exist.
        """
        if not isinstance(tool, str):
            tool = tool.__tool_name__

        try:
            self._tools.pop(tool)
        except KeyError:
            raise ValueError("Invalid tool name") from None

    def tools(self) -> Mapping[str, type[Tool]]:
        """Returns an immutable mapping of registered tools."""
        return MappingProxyType(self._tools)

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
            self.process_command(input("$$ > "))

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
