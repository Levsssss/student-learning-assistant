"""Base class for embedding providers using Strategy pattern."""

from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """
        Get the embeddings instance.

        Returns:
            Embeddings instance compatible with LangChain.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the embedding model name."""
        pass
