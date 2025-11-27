"""Vector database store using ChromaDB."""

from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

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
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create a new one."""
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception:
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
        if embeddings is None:
            # If no embeddings provided, ChromaDB will generate them
            # But we might want to use our own embedding model
            self.collection.add(
                documents=documents,
                metadatas=metadatas or [{}] * len(documents),
                ids=ids or [f"doc_{i}" for i in range(len(documents))],
            )
        else:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas or [{}] * len(documents),
                ids=ids or [f"doc_{i}" for i in range(len(documents))],
            )

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


