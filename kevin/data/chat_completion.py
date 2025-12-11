# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, Json

__all__ = (
    "Message",
    "ToolCall",
    "InferenceChatResponse",
)


class Message(BaseModel):
    """Represents a message for chat completion inference.

    Attributes
    ----------
    role: :class:`str`
        The role from which the message is originating e.g. system or user.
    content: :class:`str`
        The message content.
    """

    role: str
    content: str


class ToolCall(BaseModel):
    """Represents a tool call from an chat inference response.

    Attributes
    ----------
    name: :class:`str`
        The name of tool that has been called.
    arguments:
        Mapping of supplied arguments. The key is argument name and value
        is the argument value.
    """

    arguments: dict[str, Any] | Json[dict[str, Any]]
    name: str


class InferenceChatResponse(BaseModel):
    """Represents the response of a chat completion inference i.e. :meth:`InferenceBackend.chat`

    Attributes
    ----------
    content: :class:`str`
        The content of chat inference, if there is any; otherwise None.
    tool_class: list[:class:`ToolCall`]
        The list of tools that were called.
    """

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
