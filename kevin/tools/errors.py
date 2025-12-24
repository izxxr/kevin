# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

__all__ = (
    "ToolError",
)


class ToolError(Exception):
    """Base class for all tools related error.

    This is a special exception class that is specifically handled
    internally in various tool's calls.

    Any errors raised in tool callbacks that inherit from this class
    are passed to tool's local and assistant's global error handlers.

    Parameters
    ----------
    message: :class:`str`
        The message pertaining to this error.

    Attributes
    ----------
    message: :class:`str`
        The message pertaining to this error.
    parent: :class:`BaseException`
        The causative exception. If this error was caused by another
        exception in tool's callback.
    """
    def __init__(self, message: str) -> None:
        self.message = message
        self.parent: BaseException | None = None

    @classmethod
    def _from_exc(cls, parent: BaseException) -> ToolError:
        err = ToolError(message=str(parent))
        err.parent = parent
        return err
