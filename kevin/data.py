# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, Json

__all__ = (
    "Message",
    "Parameter",
    "Tool",
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


class Parameter(BaseModel):
    """Represents the parameters a tool takes.

    All parameters must have either a :attr:`.type` or :attr:`.enum` values
    or both.

    Attributes
    ----------
    description: :class:`str`
        The description of when the parameter is used.
    type: :class:`str`
        The JSON schema compatible type of parameter. See
        https://json-schema.org/understanding-json-schema/reference/type
    name: :class:`str`
        The name of parameter.
    enum:
        The list of enumerated values if this parameter restricts
        the values.
    required: :class:`bool`
        Whether this parameter is required.
    """

    description: str
    type: str | None = None
    name: str = Field(exclude=True)
    enum: list[Any] = Field(default_factory=list)
    required: bool = Field(exclude=True, default=True)

    def model_post_init(self, context: Any) -> None:
        if self.type is None and not self.enum:
            raise TypeError("One of type or enum must be provided")


class Tool(BaseModel):
    """Represents a tool that a language model can call.

    Attributes
    ----------
    type: :class:`str`
        The type of tool. For forward compatibility purposes only; this should
        generally never be set manually.
    name: :class:`str`
        The name of tool.
    description: :class:`str`
        The description of when this tool is called.
    parameters: list[:class:`Parameter`]
        The list of parameters that this tool takes.
    """

    type: str = "function"
    name: str
    description: str
    parameters: list[Parameter] = Field(default_factory=list)

    def dump(self) -> dict[str, Any]:
        """Serializes the tool to JSON schema compatible format.

        For developers implementing custom inference backends, this method
        should be called to convert a tool to standard format that can be
        sent for chat completion inference.

        .. warning::

            Do not use the Pydantic's model dump as it returns the tool data
            in incompatible format.
        """
        data = self.model_dump()
        props = {}
        required_props = []

        for p in self.parameters:
            props[p.name] = p.model_dump(exclude_defaults=True)

            if p.required:
                required_props.append(p.name)

        if props:
            data["parameters"] = {
                "type": "object",
                "properties": props,
                "required": required_props,
            }

        return data


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

    arguments: dict[str, Any]
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
