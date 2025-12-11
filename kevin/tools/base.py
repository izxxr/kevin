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
    """Base class for tools."""

    __tool_type__: ClassVar[str]
    __tool_name__: ClassVar[str]
    __tool_description__: ClassVar[str]

    @classmethod
    def dump(cls) -> dict[str, Any]:
        raise NotImplementedError("dump() must be defined by subclasses")

    def callback(self, assistant: Kevin, /):
        """Callback method for this function."""
