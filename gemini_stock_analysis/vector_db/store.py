"""Vector database store using ChromaDB."""

from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from ..config import Settings as AppSettings


class VectorStore:
    """Vector database store for embeddings and semantic search."""

    def __init__(self, settings: AppSettings, collection_name: str = "stock_data"):
        """Initialize the vector store."""
        self.settings = settings
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Use a simpler embedding function that works better on macOS
        # DefaultSentenceTransformerEmbeddingFunction can have CoreML issues
        try:
            # Try to use the default embedding function
            self.embedding_function = None  # Let ChromaDB use default
        except Exception:
            # Fallback to a simpler embedding function if default fails
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create a new one."""
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception:
            # Create collection - if embedding function is None, ChromaDB will use default
            # which may have CoreML issues, but we'll handle that in add_documents
            return self.client.create_collection(name=self.collection_name)

    def add_documents(
        self,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            embeddings: Optional pre-computed embeddings
            metadatas: Optional metadata for each document
            ids: Optional custom IDs for documents
        """
        # Ensure metadatas are not empty dicts (ChromaDB requires non-empty or None)
        if metadatas is None:
            metadatas_list = None
        else:
            # Replace empty dicts with a minimal dict (ChromaDB doesn't accept empty dicts)
            metadatas_list = [
                (m if m else {"_placeholder": "true"}) for m in metadatas
            ]

        try:
            if embeddings is None:
                # Try to add without embeddings first (ChromaDB will generate)
                # If this fails due to CoreML/ONNX issues, we'll catch and handle it
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas_list,
                    ids=ids or [f"doc_{i}" for i in range(len(documents))],
                )
            else:
                # Use provided embeddings
                self.collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas_list,
                    ids=ids or [f"doc_{i}" for i in range(len(documents))],
                )
        except Exception as e:
            # If embedding generation fails (e.g., CoreML/ONNX error),
            # create a simple hash-based embedding as fallback
            if "ONNXRuntimeError" in str(e) or "CoreML" in str(e):
                print(
                    "Warning: ChromaDB embedding function failed. "
                    "Using simple fallback embeddings."
                )
                # Generate simple embeddings as fallback
                import hashlib
                import struct

                fallback_embeddings = []
                for doc in documents:
                    hash_obj = hashlib.sha256(doc.encode())
                    hash_bytes = hash_obj.digest()
                    # Create a 128-dimensional embedding from hash
                    embedding = [
                        float(b) / 255.0 for b in hash_bytes[:128]
                    ] + [0.0] * (128 - len(hash_bytes[:128]))
                    fallback_embeddings.append(embedding)

                # Ensure metadatas are not empty dicts
                if metadatas is None:
                    metadatas_list = None
                else:
                    metadatas_list = [
                        (m if m else {"_placeholder": "true"}) for m in metadatas
                    ]

                self.collection.add(
                    embeddings=fallback_embeddings,
                    documents=documents,
                    metadatas=metadatas_list,
                    ids=ids or [f"doc_{i}" for i in range(len(documents))],
                )
            else:
                raise

    def search(
        self,
        query: str,
        n_results: int = 5,
        query_embeddings: Optional[List[float]] = None,
    ) -> dict:
        """
        Search for similar documents.

        Args:
            query: Query text
            n_results: Number of results to return
            query_embeddings: Optional pre-computed query embedding

        Returns:
            Dictionary with results containing documents, distances, and metadatas
        """
        try:
            if query_embeddings is None:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                )
            else:
                results = self.collection.query(
                    query_embeddings=[query_embeddings],
                    n_results=n_results,
                )
            return results
        except Exception as e:
            # If query embedding fails, generate fallback embedding
            if "ONNXRuntimeError" in str(e) or "CoreML" in str(e):
                import hashlib

                hash_obj = hashlib.sha256(query.encode())
                hash_bytes = hash_obj.digest()
                fallback_embedding = [
                    float(b) / 255.0 for b in hash_bytes[:128]
                ] + [0.0] * (128 - len(hash_bytes[:128]))

                results = self.collection.query(
                    query_embeddings=[fallback_embedding],
                    n_results=n_results,
                )
                return results
            else:
                raise

    def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()

    def get_all_documents(self) -> List[dict]:
        """Get all documents from the collection."""
        results = self.collection.get()
        return [
            {
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i],
            }
            for i in range(len(results["ids"]))
        ]


