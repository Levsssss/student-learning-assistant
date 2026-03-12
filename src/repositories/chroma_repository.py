"""Chroma vector store repository implementation."""

import logging
from pathlib import Path
from typing import List, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.repositories.base import VectorStoreRepository

logger = logging.getLogger(__name__)


class ChromaRepository(VectorStoreRepository):
    """Repository implementation using Chroma vector store."""

    def __init__(self, persist_dir: str, embeddings: Embeddings):
        """
        Initialize the Chroma repository.

        Args:
            persist_dir: Directory to persist the vector store.
            embeddings: Embedding function to use.
        """
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self._vectorstore: Optional[Chroma] = None

    def save(self, documents: List[Document]) -> None:
        """Save documents to Chroma."""
        logger.info("Creating embeddings and saving to Chroma...")
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        logger.info("Documents saved to Chroma successfully")

    def load(self) -> bool:
        """Load existing Chroma vector store."""
        if not self.exists():
            return False

        logger.info("Loading existing vector database...")
        self._vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )
        logger.info("Vector database loaded successfully")
        return True

    def exists(self) -> bool:
        """Check if the Chroma store exists."""
        return Path(self.persist_dir).exists()

    def search(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents."""
        if self._vectorstore is None:
            raise RuntimeError("Vector store not initialized. Call load() or save() first.")
        return self._vectorstore.similarity_search(query, k=k)

    def as_retriever(self, k: int = 3) -> Any:
        """Get a retriever interface."""
        if self._vectorstore is None:
            raise RuntimeError("Vector store not initialized. Call load() or save() first.")
        return self._vectorstore.as_retriever(search_kwargs={"k": k})
