# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Any, TYPE_CHECKING, Callable
from pydantic import create_model, Field
from kevin.tools.base import Tool

import inspect

if TYPE_CHECKING:
    from kevin.tools.context import ToolCallContext

__all__ = (
    "dynamic_function",
    "Function",
)


def dynamic_function(
    __name: str,
    __description: str,
    __callback: Callable[[ToolCallContext], Any],
    /,
    **__params: Any,
) -> type[Function]:
    """Utility method for dynamically creating functions.

    Dynamic functions are useful for temporary short inferences within
    tools. For example, classifying a user input as yes or no::

        from pydantic import Field

        @assistant.tool
        class ClearHistory(kevin.tools.Function):
            '''Clear the user search history.'''

            def callback(self, ctx: kevin.tools.ToolCallContext):
                def response_callback(response_ctx):
                    nonlocal is_yes  # to change the variable in outer scope
                    is_yes = response_ctx.tool_args['is_yes']

                is_yes = False
                ans = ctx.assistant.ask("You really wanna search man?")
                result = ctx.assistant.chat(
                    [
                        kevin.data.Message(
                            role='system',
                            content='Classify the given user response as yes/no and call the tool accordingly. ' \
                                    'Classify as negative as default if response cannot be classified.'
                        ),
                        kevin.data.Message(role='user', content=ans)
                    ],
                    tools=[
                        kevin.tools.dynamic_function(
                            'HandlerUserResponse',  # tool name
                            'Call this tool to classify user response as positive or negative.',  # tool description
                            response_callback,  # callback function taking call context
                            # parameters for this tool
                            is_yes=(bool, Field(default=True, description='Whether user response was positive.'))
                        ),
                    ]
                )

                # is_yes now has value set by tool call
                ctx.assistant.say(f"Clearing history..." if is_yes else "Operation canceled.")

    All arguments to this function are positional only while the tool parameters are passed
    as keyword arguments.

    This function uses `pydantic.create_model` under-the-hood since all tools are pydantic
    models. For more information on how to pass fields (tool parameters), see `create_model`
    documentation: https://docs.pydantic.dev/latest/examples/dynamic_models

    Parameters
    ----------
    name: :class:`str`
        The name of the tool.
    description: :class:`str
        The tool's description of when it is called.
    callback:
        The callback for the tool. This is a callable that takes a single
        parameter that is a :class:`ToolCallContext` instance.
    **params:
        The parameters that the tool takes as keyword arguments. The value of
        each parameter can be a `pydantic.FieldInfo` compatible value e.g. a
        type or tuple.

    Returns
    -------
    type[:class:`Function`]
        The dynamically created function.
    """
    function = create_model(
        __name,
        __doc__=__description,
        __base__=Function,
        **__params
    )
    function.callback = lambda _, ctx: __callback(ctx)
    return function


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
