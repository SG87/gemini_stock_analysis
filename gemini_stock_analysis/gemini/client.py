"""Gemini API client for AI-powered analysis."""

from typing import Any, Dict, List, Optional

import google.generativeai as genai

from ..config import Settings


class GeminiClient:
    """Client for interacting with Google's Gemini API."""

    def __init__(self, settings: Settings):
        """Initialize the Gemini client with API key."""
        self.settings = settings
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-pro")

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
            try:
                # Try to use Gemini's embedding model
                # Note: You may need to use 'models/text-embedding-004' or similar
                # depending on what's available in your API version
                embedding_model = genai.EmbeddingModel("models/embedding-001")
                result = embedding_model.embed_content(text)
                embeddings.append(result["embedding"])
            except Exception as e:
                # If Gemini embedding model is not available, return None
                # The vector store can use its default embedding function instead
                print(
                    f"Warning: Could not generate embedding with Gemini: {e}. "
                    "Vector store will use default embeddings."
                )
                return None

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

