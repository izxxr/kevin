# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from kevin.tools.errors import ToolError

if TYPE_CHECKING:
    from kevin.assistant import Kevin
    from kevin.tools.base import Tool


class ToolCallContext:
    """Stateful class for holding contextual information of a tool call.
    
    This class is normally passed to tool callbacks and related methods
    for propagating state.
    """
    def __init__(
        self,
        assistant: Kevin,
        tool: Tool,
    ):
        self.assistant = assistant
        self.tool = tool
        self.extras: dict[str, Any] = {}

    def _call(self) -> None:
        try:
            self.tool.before_callback(self)
            self.tool.callback(self)
            self.tool.after_callback(self)
        except Exception as exc:
            if not isinstance(exc, ToolError):
                exc = ToolError._from_exc(exc)

            self.tool.error_handler(self, exc)
