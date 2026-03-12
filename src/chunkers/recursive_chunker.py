"""Recursive character text splitter chunking strategy."""

import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.chunkers.base import ChunkingStrategy

logger = logging.getLogger(__name__)


class RecursiveChunkingStrategy(ChunkingStrategy):
    """Strategy for recursive character-based text chunking."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the recursive chunker.

        Args:
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents using recursive character splitting."""
        logger.info("Using recursive character splitting")
        chunks = self._splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
