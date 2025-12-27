# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from rich.console import Console
from rich.table import Table
from rich.rule import Rule

from kevin import defs
from kevin.data.chat_completion import Message, InferenceChatResponse
from kevin.utils.plugins import PluginsMixin
from kevin.tools.context import ToolCallContext

import logging
import threading
import collections
import sounddevice as sd
import speech_recognition as sr

if TYPE_CHECKING:
    from kevin.stt import STTProvider, STTResult
    from kevin.tts import TTSProvider
    from kevin.inference import InferenceBackend, InferenceChatResponse
    from kevin.waker import Waker
    from kevin.tools.base import Tool

__all__ = (
    "Kevin",
)

_default: Any = object()
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
    waker: :class:`Waker`
        The waker to use for managing assistant's awake state. If an STT provider is set,
        this must be provided.
    sleep_on_done: :class:`bool`
        Whether to sleep after command or action execution is done. This should
        generally never be set to false. Default is true.
    system_prompts: list[:class:`str`] | None
        The system prompts to provide to language model.

        If `include_default_prompt` is true (default), the default system prompt
        (defined in `kevin.defs.DEFAULT_SYSTEM_PROMPT`) will be included in the
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
        The number of seconds to wait before sleeping if no speech is detected after waking up.
        Defaults to 10 seconds. Setting this to None removes timeout.
    """

    def __init__(
        self,
        inference: InferenceBackend,
        stt: STTProvider | None = None,
        recognizer: sr.Recognizer | None = None,
        microphone: sr.Microphone | None = None,
        tts: TTSProvider | None = None,
        waker: Waker | None = None,
        sleep_on_done: bool = True,
        system_prompts: list[str] | None = None,
        include_default_prompt: bool = True,
        assistant_name: str = "KEVIN",
        username: str = "<undefined>",
        max_history_messages: int = 5,
        listen_timeout: float | None = 5.0,
    ):
        # TODO: add support for other wake-up mechanisms e.g. push-to-talk
        if stt is not None and waker is None:
            raise TypeError("waker must be provider if stt is set")

        if recognizer is None:
            recognizer = sr.Recognizer()

            if listen_timeout is not None:
                # This is usually needed for listen_timeout to work properly.
                recognizer.dynamic_energy_threshold = False

        if system_prompts is None:
            system_prompts = []

        if include_default_prompt:
            system_prompts.insert(0, self._get_default_system_prompt(assistant_name, username))

        self.inference = inference
        self.stt = stt
        self.recognizer = recognizer
        self.microphone = microphone if microphone is not None else sr.Microphone()
        self.tts = tts
        self.waker = waker
        self.sleep_on_done = sleep_on_done
        self.text_input_mode = stt is None
        self.text_output_mode = tts is None
        self.listen_timeout = listen_timeout
        self.system_prompts = [Message(role="system", content=prompt) for prompt in system_prompts]

        self._tools = {}
        self._plugins = {}
        self._name = "main"
        self._tools_data = None
        self._started = False
        self._history = collections.deque[Message](maxlen=max_history_messages)
        self._max_history_messages = max_history_messages
        self._username = username
        self._assistant_name = assistant_name
        self._rich_output = True
        self._console = Console()

    # command processing

    def _get_messages(self, command: str):
        msg = Message(role="user", content=command)
        self._history.append(msg)

        copy = self.system_prompts.copy()
        copy.extend(self._history)
        copy.append(msg)

        return copy
 
    def _get_default_system_prompt(self, assistant_name: str, user_name: str) -> str:
        return defs.DEFAULT_SYSTEM_PROMPT.format(assistant_name=assistant_name, user_name=user_name)

    def _get_tools_data(self) -> list[dict[str, Any]]:
        if self._tools_data is None:
            self._tools_data = [t.dump() for t in self._tools.values()]

        return self._tools_data
    
    def _make_call_context(self, tool: Tool) -> ToolCallContext:
        return ToolCallContext(
            assistant=self,
            tool=tool,
        )
    
    def _call_tools_from_response(
        self,
        response: InferenceChatResponse,
        tools: dict[str, type[Tool]] | None = None
    ) -> list[str]:

        if tools is None:
            tools = self._tools

        names: list[str] = []

        for call in response.tool_calls:
            try:
                tool_tp = tools[call.name]
            except KeyError:
                _log.warning("Inference response contains invalid tool name")
            else:
                tool = tool_tp(**call.arguments)
                ctx = self._make_call_context(tool)
                ctx._call()

            names.append(call.name)

        return names

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

        if response.content:
            self._log_assistant(response.content)

        _log.info("Response: %r", response.content)

        if response.content and self.tts:
            self.tts.speak(response.content)
            self._history.append(Message(role="assistant", content=response.content))

        called_names = self._call_tools_from_response(response)

        if called_names and not response.content:
            self._log_assistant(
                f"[italic]Tool{'s' if len(called_names) > 2 else ''} Called: {', '.join(called_names)}[/italic]",
            )

    # speech processing

    def _process_speech(self, data: sr.AudioData, return_result: bool = False) -> STTResult | None:
        if self.stt is None or not self._awake():
            return

        result = self.stt.transcribe(data)

        if not result.has_speech or not result.text:
            return

        _log.info(f"Speech received: {result.text!r}")
        self._log_user(result.text)

        if return_result:
            return result

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
        # background. For now, we are limited to require repeated wake up calls until a
        # feasible solution is available.
        if self.sleep_on_done and self.waker is not None:
            self._log_rich("💤", "Sleeping...")
            self.waker.sleep()

    def _awake(self) -> bool:
        return self.waker is not None and self.waker.awake()

    def _wake_assistant(self) -> None:
        assert self.waker is not None
        self.waker.try_wake()

        if not self.waker.awake():
            return

        # stop ongoing TTS speeches
        while self.tts and self.tts.is_speaking():
            self.tts.stop()

        self._log_rich("🎤", "Listening...", prepend_newline=True)

    def _listen_speech(self, source: sr.AudioSource, return_result: bool = False, listen_timeout: float | None = _default) -> STTResult | None:
        if not self._awake() and not return_result:
            return self._wake_assistant()

        if listen_timeout is _default:
            listen_timeout = self.listen_timeout

        try:
            speech = self.recognizer.listen(source, timeout=listen_timeout)
        except sr.WaitTimeoutError:
            if return_result:
                raise TimeoutError("Timed out while waiting for speech input") from None

            self._log_rich("💤", "Sleeping...")
            self.waker.sleep()  # type: ignore
        else:
            result = self._process_speech(speech, return_result=return_result)

            if return_result:
                return result

    # Text mode

    def _read_input(self, return_input: bool = False) -> str | None:
        if self._rich_output:
            command = self._console.input(f"{'' if return_input else '\n'} 👤 │ ")  # type: ignore
        else:
            command = input("$$ > ")

        if return_input:
            return command

        # stop ongoing TTS speeches
        while self.tts and self.tts.is_speaking():
            self.tts.stop()

        self.process_command(command)

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

    def _print_splash(self):
        if not self._rich_output:
            return

        self._console.clear()
        self._console.print(defs.KEVIN_ASCII_ART, "\n", highlight=False, justify="center")

        table = Table("Key", "Value", title="Startup Information")

        table.add_row("Assistant Name", self._assistant_name)
        table.add_row("User Name", self._username)
        table.add_row("History Size", str(self._max_history_messages))
        table.add_row("Registered Plugins", str(len(self._plugins)))

        self._console.print(table, justify="center")
        self._console.print(Rule(align="center"))

        self._console.print()
        self._log_rich(
            "🚀", f"Running in [yellow]{'💬 text' if self.text_input_mode else '🎤 speech'}[/yellow] mode",
        )

    def _log_rich(
        self,
        emoji: str,
        text: str,
        prepend_newline: bool = False,
        return_result: bool = False,
        **kwargs: Any
    ) -> Any:
        if not self._rich_output:
            return

        table = Table(
            show_header=False,
            show_edge=False,
            show_footer=False,
            show_lines=False
        )

        table.add_row(f"{emoji}", text)

        if return_result:
            return table

        if prepend_newline:
            self._console.line()

        self._console.print(table, **kwargs)

    def _log_user(self, text: str): 
        self._log_rich("👤", text)

    def _log_assistant(self, text: str):
        self._log_rich("💡", text)

    def start(self, *, verbose_logging: bool = False) -> None:
        """Starts the assistant.

        This is a blocking method and does not return until the assistant
        has stopped.

        Parameters
        ----------
        verbose_logging: :class:`bool`
            Whether to enable verbose logs. By default, with this set to false,
            assistant starts with an interactive chatting terminal interface.

            This parameter can be set while debugging to get verbose logging outputs
            without any interactive interface.
        """
        if self._started:
            raise RuntimeError("Assistant is already running")
        
        self._rich_output = not verbose_logging
        self._print_splash()

        if verbose_logging:
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

        self._log_rich("❌", "Exiting...")
        self._started = False

    # convenience methods

    def say(
        self,
        content: str,
        *,
        speak: bool = True,
        blocking: bool = False,
        role: str | None = "assistant",
    ) -> None:
        """Says the given message.

        Parameters
        ----------
        content: :class:`str`
            The content of message to say.
        speak: :class:`bool`
            Whether to say the message through TTS if available. Defaults to True.
            If no TTS is available or this is set to false, the message is simply logged.
        blocking: :class:`bool`
            If speakable message, whether to block the caller thread while synthesis
            of speech. This is same as ``blocking`` parameter in :meth:`TTSProvider.speak`.
        role: :class:`str` | None
            The role with which the message is saved in history. Defaults to assistant.
            If set to None, the message is not added to history.
        """
        self._log_assistant(content)

        if speak and self.tts:
            self.tts.speak(content, blocking=blocking)

        if role is not None:
            self.add_message_to_history(role, content)

    def ask(
        self,
        prompt: str | None = None,
        *,
        listen: bool = True,
        listen_timeout: float | None = _default,
        blocking: bool = True,
    ):
        """Ask or prompt the user for an input.

        Parameters
        ----------
        prompt: :class:`str` | None
            The prompt to show for input. This is simply passed to the
            :meth:`.say` method call before taking the input.
        listen: :class:`bool`
            Whether to listen the input if an STT provider is available (i.e.
            assistant is running in speech mode)

            Defaults to true. If set to false, the input is taken through
            standard input regardless of text or speech mode.
        listen_timeout: :class:`float` | None
            Only applicable for `listen=True` - the number of seconds to wait
            for speech before timing out. If not passed, defaults to the global
            :attr:`.listen_timeout` value.

            If None is passed, the listening will continue indefinitely until
            a speech is recognized.
        blocking: :class:`bool`
            Whether to block the caller thread while speaking the prompt. Only
            applicable for `listen=True` case.

            The default for this is true in this method, unlike in :meth:`.say`,
            to avoid echo back to user input speech. The prompt is first completed
            before listening to input speech.

        Returns
        -------
        :class:`str`
            The received input.

        Raises
        ------
        TimeoutError
            No speech detected in `listen_timeout` seconds.
        """
        if prompt:
            self.say(prompt, blocking=blocking)
        if listen and not self.text_input_mode:
            if self.microphone.stream is not None:
                result = self._listen_speech(
                    self.microphone,
                    return_result=True,
                    listen_timeout=listen_timeout
                )
            else:
                with self.microphone:
                    result = self._listen_speech(
                        self.microphone,
                        return_result=True,
                        listen_timeout=listen_timeout
                    )

            assert result is not None, "_listen_speech() unexpectedly returned None"
            text = result.text
        else:
            text = self._read_input(return_input=True)

        assert text is not None, "_read_input() unexpectedly returned None"
        return text
    
    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[type[Tool]] | None = None,
        tools_data: list[dict[str, Any]] | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> InferenceChatResponse:
        """Performs chat completion inference on given messages.

        This method is similar to :meth:`InferenceBackend.chat` except
        that post processing is performed on inference result such as
        calling relevant tools.

        Note that the result's content is not added to assistant's history
        or processed through TTS. If either of these operations are required,
        they must be done manually.

        All parameters that this method takes are the same as :meth:`InferenceBackend.chat`
        and returns :class:`InferenceChatResponse`.
        """
        result = self.inference.chat(messages, tools=tools, tools_data=tools_data, extra_options=extra_options)

        if tools and result.tool_calls:
            self._call_tools_from_response(result, tools={t.__tool_name__: t for t in tools})

        return result

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
