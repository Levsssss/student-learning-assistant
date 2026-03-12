"""Base class for vector store repositories using Repository pattern."""

from abc import ABC, abstractmethod
from typing import List, Any

from langchain_core.documents import Document


class VectorStoreRepository(ABC):
    """Abstract base class for vector store operations."""

    @abstractmethod
    def save(self, documents: List[Document]) -> None:
        """
        Save documents to the vector store.

        Args:
            documents: List of documents to save.
        """
        pass

    @abstractmethod
    def load(self) -> bool:
        """
        Load existing vector store.

        Returns:
            True if successfully loaded, False if store doesn't exist.
        """
        pass

    @abstractmethod
    def exists(self) -> bool:
        """
        Check if the vector store exists.

        Returns:
            True if the store exists.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            List of similar documents.
        """
        pass

    @abstractmethod
    def as_retriever(self, k: int = 3) -> Any:
        """
        Get a retriever interface for the vector store.

        Args:
            k: Number of documents to retrieve.

        Returns:
            Retriever instance.
        """
        pass
