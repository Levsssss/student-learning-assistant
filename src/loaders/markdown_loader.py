"""Markdown document loader strategy (raw text preserving headers)."""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from src.loaders.base import DocumentLoaderStrategy

logger = logging.getLogger(__name__)


class MarkdownLoaderStrategy(DocumentLoaderStrategy):
    """Strategy for loading Markdown documents as raw text."""

    def load(self, file_path: Path) -> List[Document]:
        """Load Markdown file as raw text (preserves # headers)."""
        logger.info(f"Loading Markdown file (raw): {file_path.name}")
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def supports(self, file_path: Path) -> bool:
        """Check if file is a Markdown file."""
        return file_path.suffix.lower() == ".md"