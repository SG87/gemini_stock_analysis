"""Gemini API client for AI-powered analysis."""

import json
from typing import List, Optional

import google.generativeai as genai
from google.generativeai import types
from mcp import ClientSession
from mcp.types import CallToolResult

from gemini_stock_analysis.config import Settings


class GeminiClient:
    """Client for interacting with Google's Gemini API."""

    def __init__(self, settings: Settings):
        """
        Initialize the Gemini client with API key.

        :param settings: Settings object
        """
        self.settings = settings
        genai.configure(api_key=settings.gemini_api_key)

        with open("gemini_model_config.json", "r") as gemini_config_file:
            gemini_model_config = json.loads(gemini_config_file.read())

        self.model = genai.GenerativeModel(gemini_model_config.get("model"))
        self.embeddings_model = gemini_model_config.get("embeddings_model")
        self.embedding_size = gemini_model_config.get("embedding_size")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Gemini.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            # Try to use Gemini's embedding model
            # Note: You may need to use 'models/text-embedding-004' or similar
            # depending on what's available in your API version
            result = genai.embed_content(
                model=self.embeddings_model,
                content=text,
                task_type="retrieval_document",
                output_dimensionality=self.embedding_size,
                title="Custom Query",
            )
            embeddings.append(result["embedding"])

        return embeddings

    def analyze(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate analysis using Gemini API.

        Args:
            prompt: The analysis prompt
            context: Optional context to include in the prompt

        Returns:
            Generated analysis text
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt

        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Error generating content: {str(e)}")

    async def chat(self, message: str, mcp_session: ClientSession) -> str:
        # 1. Get available tools from MCP Server
        available_tools = await mcp_session.list_tools()
        gemini_tool_definitions = self.mcp_to_gemini_tools(available_tools)

        chat = self.model.start_chat()

        # 3. Send message to Gemini with tools
        response = chat.send_message(
            message, tools=[types.Tool(function_declarations=gemini_tool_definitions)]
        )

        # 4. Loop: Handle Tool Calls until Gemini is satisfied
        # Note: In a real app, you might want a limit to prevent infinite loops
        while True:
            # Check if Gemini wants to call a function
            function_call = None
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    function_call = part.function_call
                    break

            if not function_call:
                break  # No tool call, we have the final answer

            # Execute the tool on the MCP Server
            print(f"🤖 Gemini requesting tool: {function_call.name}")
            result: CallToolResult = await mcp_session.call_tool(
                function_call.name, arguments=function_call.args
            )

            # Send result back to Gemini
            tool_output = result.content[0].text
            print(f"✅ Tool Result: {tool_output}")

            response = chat.send_message(
                types.Part.from_function_response(
                    name=function_call.name, response={"result": tool_output}
                )
            )

        # 5. Return final text to user
        return {"response": response.text}

    def mcp_to_gemini_tools(self, mcp_tools):
        gemini_tools = []
        for tool in mcp_tools.tools:
            clean_params = self.clean_schema(tool.inputSchema.copy())

            gemini_tools.append(
                {"name": tool.name, "description": tool.description, "parameters": clean_params}
            )
        return gemini_tools

    def clean_schema(self, schema: dict) -> dict:
        """
        Recursively removes fields that Gemini's SDK does not support
        (like 'title', '$schema', 'additionalProperties') from the JSON schema.
        """
        if not isinstance(schema, dict):
            return schema

        # specific fields to remove
        for field in ["title", "$schema", "additionalProperties", "id"]:
            if field in schema:
                del schema[field]

        # Recursively clean 'properties'
        if "properties" in schema:
            for key, value in schema["properties"].items():
                schema["properties"][key] = self.clean_schema(value)

        # Recursively clean 'items' (for arrays)
        if "items" in schema:
            schema["items"] = self.clean_schema(schema["items"])

        return schema
