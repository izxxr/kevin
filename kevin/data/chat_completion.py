# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any, TypeVar, Generic, Annotated
from pydantic import BaseModel, Field

from kevin.data.tools import ToolCall

__all__ = (
    "Message",
    "InferenceChatResponse",
)


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class Message(BaseModel):
    """Represents a message for chat completion inference.

    Attributes
    ----------
    role: :class:`str`
        The role from which the message is originating e.g. system or user.
    content: :class:`str`
        The message content.
    extras: dict[:class:`str`, Any]
        Additional data to include in message such as tool calls.
    """

    role: str
    content: str
    extras: dict[str, Any] | None = Field(exclude=True, default=None)

    def dump(self) -> dict[str, Any]:
        """Returns the message data with any extras injected."""
        data = self.model_dump()

        if self.extras:
            data.update(self.extras)

        return data


class InferenceChatResponse(BaseModel, Generic[SchemaT]):
    """Represents the response of a chat completion inference i.e. :meth:`InferenceBackend.chat`

    Attributes
    ----------
    content: :class:`str`
        The content of chat inference, if there is any; otherwise None.
    tool_class: list[:class:`ToolCall`]
        The list of tools that were called.
    model: :class:`BaseModel` | dict[:class:`str`, Any]
        The Pydantic model returned as structured output by LLM.
    """

    content: str | None = None
    model: Annotated[SchemaT | None, BaseModel] = Field(default=None)
    tool_calls: list[ToolCall] = Field(default_factory=list)
