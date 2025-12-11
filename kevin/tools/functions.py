# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any, ClassVar
from kevin.tools.base import Tool

import inspect

__all__ = (
    "Function",
)


class Function(Tool):
    """Base class for functions.

    Functions are tools that can be called by language models. In simpler
    words, these are actions that an assistant can perform such as checking
    weather.

    To define a function, :meth:`Assistant.tool` decorator is typically used
    to decorate a class that inherits from this class.

    All functions must define the :meth:`.callback` method and the parameters
    are defined as attributes of this class. This class is a Pydantic model
    so the attributes (parameters for function) support all of Pydantic field
    validation features.
    """

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
