import json

import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings


# 1. Define the Wrapper Class
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        with open("gemini_model_config.json", "r") as gemini_config_file:
            gemini_model_config = json.loads(gemini_config_file.read())

        embeddings_model = gemini_model_config.get("embeddings_model")

        # Call the Google API
        response = genai.embed_content(
            model=embeddings_model,
            content=input,
            task_type="retrieval_document",
            title="Custom Query",
        )

        # Return the embeddings list
        return response["embedding"]
