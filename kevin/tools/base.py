# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import ClassVar, Any, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from kevin.assistant import Kevin

__all__ = (
    "Tool",
)

class Tool(BaseModel):
    """Base class for tools.

    Tools are sent to language model that support them and they
    can be called. Currently, the only type of tools are functions.
    """

    __tool_type__: ClassVar[str]
    __tool_name__: ClassVar[str]
    __tool_description__: ClassVar[str]

    @classmethod
    def dump(cls) -> dict[str, Any]:
        """Serializes the tool data.

        This returns a dictionary in standard format that can be sent
        to language models that support tools.
        """
        raise NotImplementedError("dump() must be defined by subclasses")

    def callback(self, assistant: Kevin, /):
        """Callback method for the tool.

        This is called when this tool is called by an LM.

        Parameters
        ----------
        assistant: :class:`Kevin`
            The assistant that called this tool.
        """
