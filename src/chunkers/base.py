"""Base class for document chunkers using Strategy pattern."""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class ChunkingStrategy(ABC):
    """Abstract base class for document chunking strategies."""

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to chunk.

        Returns:
            List of chunked Document objects.
        """
        pass
