"""Factory for creating document loaders."""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from src.loaders.base import DocumentLoaderStrategy
from src.loaders.pdf_loader import PDFLoaderStrategy
from src.loaders.markdown_loader import MarkdownLoaderStrategy

logger = logging.getLogger(__name__)


class DocumentLoaderFactory:
    """Factory for creating and managing document loaders."""

    _strategies: List[DocumentLoaderStrategy] = [
        MarkdownLoaderStrategy(),
        PDFLoaderStrategy(),
    ]

    @classmethod
    def register_strategy(cls, strategy: DocumentLoaderStrategy) -> None:
        """Register a new loader strategy."""
        cls._strategies.insert(0, strategy)

    @classmethod
    def create(cls, file_path: Path) -> DocumentLoaderStrategy:
        """
        Create appropriate loader for the given file.

        Args:
            file_path: Path to the file.

        Returns:
            Appropriate DocumentLoaderStrategy instance.

        Raises:
            ValueError: If no loader supports the file type.
        """
        for strategy in cls._strategies:
            if strategy.supports(file_path):
                return strategy
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    @classmethod
    def detect_and_load(cls, base_path: str) -> tuple[Path, List[Document]]:
        """
        Detect file type and load documents.

        Prefers .md over .pdf if both exist.

        Args:
            base_path: Base filename without extension.

        Returns:
            Tuple of (file_path, documents).

        Raises:
            FileNotFoundError: If no supported file is found.
        """
        md_path = Path(f"{base_path}.md")
        pdf_path = Path(f"{base_path}.pdf")

        file_path: Optional[Path] = None

        if md_path.exists() and pdf_path.exists():
            logger.warning(
                "Both .md and .pdf found. Using .md (better structure preservation)"
            )
            file_path = md_path
        elif md_path.exists():
            logger.info("Using Markdown file (better for structured content)")
            file_path = md_path
        elif pdf_path.exists():
            logger.info("Using PDF file")
            file_path = pdf_path
        else:
            raise FileNotFoundError(
                f"No context file found. Please add either:\n"
                f"  - {md_path.absolute()}\n"
                f"  - {pdf_path.absolute()}\n\n"
                f"TIP: Markdown (.md) is recommended for better structure and accuracy!"
            )

        loader = cls.create(file_path)
        documents = loader.load(file_path)
        logger.info(f"Loaded {len(documents)} document(s)")

        return file_path, documents
