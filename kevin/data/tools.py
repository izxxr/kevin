# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any, Mapping
from pydantic import BaseModel, Json

import json

__all__ = (
    "ToolCall",
    "FunctionCall",
    "tool_call_factory",
)


class ToolCall(BaseModel):
    """Represents a tool call from an chat inference response.

    Attributes
    ----------
    id: :class:`str`
        The ID of this tool call.
    type: :class:`str`
        The type of tool that was called.
    """

    id: str
    type: str

    def dump(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type}


class FunctionCall(ToolCall):
    """Represents a function call.

    This is simply a child of :class:`ToolCall` that has `type=function`

    Attributes
    ----------
    name: :class:`str`
        The name of function that has been called.
    arguments:
        Mapping of supplied arguments. The key is argument name and value
        is the argument value.
    """
    name: str
    arguments: dict[str, Any] | Json[dict[str, Any]]

    def dump(self, json_dumps_arguments: bool = True) -> dict[str, Any]:
        """Converts function call to raw data.

        Parameters
        ----------
        json_dumps_argument: :class:`bool`
            Whether to include arguments as JSON parseable string. Defaults
            to true. This is generally a requirement when sending back tool
            response.
        """
        data = super().dump()
        data.update(
            function={
                "name": self.name,
                "arguments": json.dumps(self.arguments) if json_dumps_arguments else self.arguments,
            }
        )
        return data


def tool_call_factory(data: Mapping[str, Any]) -> ToolCall:
    """Factory method to initalize the appropriate :class:`ToolCall` type.

    This operates based on the `type` key in raw data. If the value of type
    key is invalid, a generic :class:`ToolCall` instance is returned.

    This is mostly reserved for internal use and exposed for custom
    implementations of inference backend.

    Parameters
    ----------
    data:
        The raw tool call data.
    """
    input_data = {
        "id": data["id"],
        "type": data["type"],
    }

    if data["type"] == "function":
        return FunctionCall(
            **input_data,
            name=data["function"]["name"],
            arguments=data["function"].get("arguments", {})
        )

    return ToolCall(**input_data)
