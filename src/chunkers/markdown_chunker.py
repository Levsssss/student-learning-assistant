"""Markdown header-based chunking strategy."""

import logging
from typing import List

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from src.chunkers.base import ChunkingStrategy

logger = logging.getLogger(__name__)


class MarkdownHeaderChunkingStrategy(ChunkingStrategy):
    """Strategy for header-based Markdown chunking with structure preservation."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the Markdown header chunker.

        Args:
            chunk_size: Maximum size for secondary splitting.
            chunk_overlap: Overlap for secondary splitting.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._headers_to_split_on,
            strip_headers=False,
        )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents using header-based splitting with fallback."""
        logger.info("Using header-based splitting for Markdown (preserves structure)")

        all_splits: List[Document] = []
        for doc in documents:
            md_splits = self._md_splitter.split_text(doc.page_content)
            all_splits.extend(md_splits)

        logger.info(f"Header-based splitting created {len(all_splits)} sections")

        if all_splits:
            final_splits = self._text_splitter.split_documents(all_splits)
            logger.info(f"Final chunking created {len(final_splits)} chunks")
            return final_splits
        else:
            logger.warning(
                "Header splitting produced no results, using standard splitting"
            )
            return self._text_splitter.split_documents(documents)
