# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import ClassVar, Any, TYPE_CHECKING
from pydantic import BaseModel

import traceback

if TYPE_CHECKING:
    from kevin.tools.errors import ToolError
    from kevin.tools.context import ToolCallContext

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

    def callback(self, context: ToolCallContext, /) -> Any:
        """Callback method for the tool.

        This is called when this tool is called by an LM.

        Parameters
        ----------
        context: :class:`ToolCallContext`
            The context in which this tool was called.
        """

    def before_callback(self, context: ToolCallContext, /) -> None:
        """Hook method executed before calling :meth:`.callback`.

        Parameters
        ----------
        context: :class:`ToolCallContext`
            The context in which this tool was called.
        """

    def after_callback(self, context: ToolCallContext, /) -> None:
        """Hook method executed after :meth:`.callback` is done executing.

        Parameters
        ----------
        context: :class:`ToolCallContext`
            The context in which this tool was called.
        """

    def error_handler(self, context: ToolCallContext, error: ToolError, /) -> None:
        """Local error handler for the tool.

        Any errors raised by :meth:`.before_callback`, :meth:`.after_callback`,
        or :meth:`.callback` that inherit from :class:`ToolError` are passed to
        this method.

        Other errors are wrapped as :class:`ToolError` and passed to this handler.

        By default, this method only prints the error traceback.
        """
        if error.parent is not None:
            traceback.print_exception(error.parent)
        else:
            traceback.print_exception(error)
