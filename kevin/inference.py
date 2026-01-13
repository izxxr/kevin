# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from kevin.tools import Tool
from kevin.data.chat_completion import Message, InferenceChatResponse, SchemaT
from kevin.data.tools import tool_call_factory

import json

if TYPE_CHECKING:
    from pydantic import BaseModel
    from huggingface_hub import ChatCompletionOutput

__all__ = (
    "InferenceBackend",
)


class InferenceBackend:
    """Base class for all inference backends.

    This class provides a common interface for interacting with various
    different inference backends (such as HF's InferenceClient) allowing
    other backends to be used.

    Currently, Kevin only provides :class:`HuggingFaceInferenceBackend` as
    a built-in backend however, other backends may be implemented as well.
    """

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[type[Tool]] | None = None,
        tools_data: list[dict[str, Any]] | None = None,
        model_cls: type[SchemaT] | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> InferenceChatResponse[SchemaT]:
        """Performs a chat completion inference.

        Parameters
        ----------
        messages: list[:class:`Message`]
            The list of messages to send.
        tools: list[:class:`Tool`]
            The list of tools that the model can call from. For more information, see
            https://huggingface.co/docs/transformers/main/en/tools

            Normally, this parameter is provided internally by :class:`Kevin` assistant
            based on defined tools. Generally, there is no need to manually provide tools.
        tools_data:
            The serialized tools data. This exists as a "raw" version of ``tools`` parameter
            and may be supplied by :class:`Kevin` to prevent serialization of tool models on
            each call of this method.
        model_cls: type[:class:`BaseModel`]
            The Pydantic model class that the LLM should respond with. This is only usable
            with models that support structured outputs. The returned model will be accessible
            from :attr:`InferenceChatResponse.model` attribute.
        extra_options:
            Extra parameters passed to underlying inference client. 

        Returns
        -------
        :class:`InferenceChatResponse`
        """
        raise NotImplementedError("Inference backend does not support chat completion")


class HuggingFaceInferenceBackend(InferenceBackend):
    """Inference backend based on :class:`huggingface_hub.InferenceClient`

    `huggingface_hub` must be installed to use this backend.

    .. info::

        Common parameters such as model, provider, and token are provided however
        for fine grained control, use the ``client_options`` parameter.

    Parameters
    ----------
    model: :class:`str` | None
        The model to use.
    provider: :class:`str` | None
        The inference provider to use.
    token: :class:`str` | None
        The hugging face authentication token.
    client_options: :class:`str` | None
        The options to pass to :class:`huggingface_hub.InferenceClient` in case
        more granular control is needed of underlying client instance.
    """
    def __init__(
        self,
        model: str | None = None,
        *,
        provider: str | None = None,
        token: str | None = None,
        client_options: dict[str, Any] | None = None
    ):
        try:
            from huggingface_hub import InferenceClient
        except Exception:
            raise RuntimeError("huggingface_hub must be installed to use HuggingFaceInferenceBackend")
        
        options: dict[str, Any] = {
            "model": model,
            "provider": provider,
            "token": token,
        }

        if client_options is not None:
            options.update(client_options)

        self.client = InferenceClient(**options)

    def _make_chat_response(
        self,
        data: ChatCompletionOutput,
        model_cls: type[SchemaT] | None = None,
    ) -> InferenceChatResponse[SchemaT]:
        message = data.choices[0].message

        if model_cls is not None:
            if message.content is None:
                raise ValueError("Chat completion unexpectedly return null content when model_cls was provided")

            try:
                data = json.loads(message.content)
            except Exception:
                raise ValueError("Returned content is not JSON serializable when model_cls was provided")
            
            model = model_cls(**data)
        else:
            model = None

        return InferenceChatResponse(
            content=message.content,
            tool_calls=[
                tool_call_factory(call)
                for call in (message.tool_calls or [])
            ],
            model=model,
        )
    
    def _get_object_response_format(self, model_cls: type[BaseModel]) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": model_cls.__name__,
                "schema": model_cls.model_json_schema(),
                "strict": True,
            }
        }

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[type[Tool]] | None = None,
        tools_data: list[dict[str, Any]] | None = None,
        model_cls: type[SchemaT] | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> InferenceChatResponse[SchemaT]:
        if tools and tools_data:
            raise TypeError("tools and tools_data are mutually exclusive")
        if tools:
            tools_data = [t.dump() for t in tools]
        
        options = {
            "messages": [m.dump() for m in messages],
            "tools": tools_data,
        }

        if model_cls is not None:
            options["response_format"] = self._get_object_response_format(model_cls)

        if extra_options:
            options.update(extra_options)

        data: ChatCompletionOutput = self.client.chat_completion(**options)
        return self._make_chat_response(data, model_cls=model_cls)
