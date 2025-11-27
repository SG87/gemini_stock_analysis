"""Model Context Protocol (MCP) handler for structured AI interactions."""

import json
from typing import Any, Dict, List, Optional

from ..config import Settings


class MCPHandler:
    """
    Handler for Model Context Protocol (MCP).

    MCP is a protocol for structured communication with AI models,
    allowing for better context management and tool usage.
    """

    def __init__(self, settings: Settings):
        """Initialize the MCP handler."""
        self.settings = settings
        self.context: List[Dict[str, Any]] = []

    def add_context(self, role: str, content: Any, metadata: Optional[Dict] = None) -> None:
        """
        Add context to the MCP context stack.

        Args:
            role: Role of the context (e.g., 'user', 'assistant', 'system', 'tool')
            content: Content of the context
            metadata: Optional metadata for the context
        """
        context_item = {
            "role": role,
            "content": content,
        }
        if metadata:
            context_item["metadata"] = metadata

        self.context.append(context_item)

    def get_context_string(self, max_items: Optional[int] = None) -> str:
        """
        Get formatted context string for use in prompts.

        Args:
            max_items: Maximum number of context items to include (None for all)

        Returns:
            Formatted context string
        """
        items = self.context[-max_items:] if max_items else self.context
        context_parts = []

        for item in items:
            role = item["role"]
            content = item["content"]
            metadata = item.get("metadata", {})

            if isinstance(content, (dict, list)):
                content = json.dumps(content, indent=2)

            context_line = f"[{role.upper()}] {content}"
            if metadata:
                context_line += f" (metadata: {json.dumps(metadata)})"

            context_parts.append(context_line)

        return "\n\n".join(context_parts)

    def clear_context(self) -> None:
        """Clear all context."""
        self.context = []

    def format_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a tool call according to MCP protocol.

        Args:
            tool_name: Name of the tool to call
            parameters: Parameters for the tool call

        Returns:
            Formatted tool call dictionary
        """
        return {
            "type": "tool_call",
            "tool": tool_name,
            "parameters": parameters,
        }

    def format_tool_result(
        self, tool_name: str, result: Any, success: bool = True
    ) -> Dict[str, Any]:
        """
        Format a tool result according to MCP protocol.

        Args:
            tool_name: Name of the tool that was called
            result: Result from the tool
            success: Whether the tool call was successful

        Returns:
            Formatted tool result dictionary
        """
        return {
            "type": "tool_result",
            "tool": tool_name,
            "success": success,
            "result": result,
        }

    def build_prompt(
        self,
        user_prompt: str,
        include_context: bool = True,
        include_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build a complete prompt with context and tools.

        Args:
            user_prompt: The main user prompt
            include_context: Whether to include context history
            include_tools: Optional list of available tools

        Returns:
            Complete formatted prompt
        """
        parts = []

        if include_context and self.context:
            parts.append("CONTEXT:")
            parts.append(self.get_context_string())
            parts.append("")

        if include_tools:
            parts.append("AVAILABLE TOOLS:")
            for tool in include_tools:
                tool_desc = f"- {tool.get('name', 'unknown')}: {tool.get('description', '')}"
                parts.append(tool_desc)
            parts.append("")

        parts.append("USER REQUEST:")
        parts.append(user_prompt)

        return "\n".join(parts)


