# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any, ClassVar
from kevin.tools.base import Tool

import inspect

__all__ = (
    "Function",
)


class Function(Tool):
    """Base class for all functions that a language model can call."""

    __tool_type__ = "function"

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any):
        description = kwargs.get("description")

        if description is None:
            if cls.__doc__ is None:
                raise TypeError("description must be provided as subclass parameter or as docstring")

            description = inspect.cleandoc(cls.__doc__).splitlines()[0]

        cls.__tool_name__ = kwargs.get("name", cls.__qualname__)
        cls.__tool_description__ = description

    @classmethod
    def dump(cls):
        params = cls.model_json_schema()
        params.pop("title", None)

        return {
            "type": "function",
            "function": {
                "name": cls.__tool_name__,
                "description": cls.__tool_description__,
                "parameters": params,
            },
        }
