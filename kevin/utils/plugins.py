# Copyright (C) Izhar Ahmad 2025-2026

from __future__ import annotations

from typing import Mapping
from types import MappingProxyType
from kevin.tools import Tool

__all__ = (
    "Plugin",
)


class PluginsMixin:
    _plugins: dict[str, Plugin]
    _tools: dict[str, type[Tool]]

    def tool(self, tool_tp: type[Tool] | None = None):
        """Registers a decorated class as tool.

        This is a decorator based interface for :meth:`.add_tool`. Example
        usage::

            @assistant.tool()
            class CheckWeather(kevin.tools.Function):
                '''Checks weather of the given location.'''
                location: str

                def callback(self, assistant):
                    # ... some meaningful weather fetching code ...
                    print(f"Weather at {self.location} is sunny")
        """
        if tool_tp is not None:
            self.add_tool(tool_tp)
            return tool_tp

        def __wrapper(tool_tp: type[Tool]):
            self.add_tool(tool_tp)
            return tool_tp

        return __wrapper

    # Tools management

    def add_tool(self, tool: type[Tool], *, override: bool = False) -> None:
        """Registers a tool that can be called by the assistant.

        Parameters
        ----------
        tool: type[:class:`Tool`]
            The tool to add.
        override: :class:`bool`
            Whether to override the tool if an existing one is registered with the
            same name. Defaults to False.

            An error is raised if this is false and a tool is being added that has
            the same name as one already added.
        """
        if self._tools.get(tool.__tool_name__) is not None and not override:
            raise ValueError(f"Tool with name {tool.__tool_name__!r} already registered")

        self._tools[tool.__tool_name__] = tool

    def remove_tool(self, tool: type[Tool] | str, *, raise_error: bool = True) -> None:
        """Removes an already registered tool.

        Parameters
        ----------
        tool: type[:class:`Tool`]
            The tool's name or the tool class.
        raise_error: :class:`bool`
            If true (default), raise an error if tool to be removed does not exist.
        """
        if not isinstance(tool, str):
            tool = tool.__tool_name__

        try:
            self._tools.pop(tool)
        except KeyError:
            if raise_error:
                raise ValueError("Invalid tool name") from None

    def tools(self) -> Mapping[str, type[Tool]]:
        """Returns an immutable mapping of registered tools."""
        return MappingProxyType(self._tools)

    def add_plugin(self, plugin: Plugin) -> None:
        """Register a plugin.

        Parameters
        ----------
        plugin: :class:`Plugin`
            The plugin to register.
        """
        if self._plugins.get(plugin.name) is not None:
            raise ValueError(f"Plugin {plugin.name!r} already registered")

        self._plugins[plugin.name] = plugin

        for tool in plugin._tools.values():
            self.add_tool(tool)

    def eject_plugin(self, plugin: Plugin | str) -> None:
        """Ejects a plugin and corresponding tools from that plugin.
        
        Parameters
        ----------
        plugin: :class:`Plugin` | :class:`str`
            The plugin to eject; either its name or the instance.
        """
        if isinstance(plugin, str):
            plugin_name = plugin
            plugin = self._plugins.get(plugin_name)  # type: ignore
        else:
            plugin_name = plugin.name

        if plugin is None:
            raise ValueError(f"Plugin {plugin!r} does not exist")

        self._plugins.pop(plugin_name)

        # plugin should always be a Plugin instance here
        for tool in plugin._tools.values():  # type: ignore
            self.remove_tool(tool, raise_error=False)

    def plugins(self) -> Mapping[str, Plugin]:
        """Returns a read-only mapping of plugins."""
        return MappingProxyType(self._plugins)


class Plugin(PluginsMixin):
    """Plugin for defining a set of tools.

    Plugins are useful in splitting the assistant code, specifically the
    tools into separate modules.

    Example usage::

        import kevin

        utility = kevin.utils.Plugin("utility")

        @utility.tool
        class CheckWeather(kevin.tools.Function):
            ...  # code for the tool

    ... then in the main file::

        from plugins.utilty import utility
        import kevin

        assistant = kevin.Kevin(...)
        assistant.add_plugin(utility)

        assistant.start()

    Plugins are arbitrarily nestable meaning plugins can be added inside plugins
    allowing complex modular structure.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._tools = {}
        self._plugins = {}
