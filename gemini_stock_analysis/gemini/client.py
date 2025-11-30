"""Gemini API client for AI-powered analysis."""
import json
from typing import Dict, List, Optional

import google.generativeai as genai

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

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Have a conversation with Gemini.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            Response text from Gemini
        """
        try:
            chat = self.model.start_chat(history=[])
            for msg in messages[:-1]:
                if msg["role"] == "user":
                    chat.send_message(msg["content"])
                elif msg["role"] == "assistant":
                    # Add to history if needed
                    pass

            response = chat.send_message(messages[-1]["content"])
            return response.text
        except Exception as e:
            raise Exception(f"Error in chat: {str(e)}")
