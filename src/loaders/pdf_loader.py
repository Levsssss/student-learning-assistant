"""PDF document loader strategy."""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.loaders.base import DocumentLoaderStrategy

logger = logging.getLogger(__name__)


class PDFLoaderStrategy(DocumentLoaderStrategy):
    """Strategy for loading PDF documents."""

    def load(self, file_path: Path) -> List[Document]:
        """Load documents from a PDF file."""
        logger.info(f"Loading PDF file: {file_path.name}")
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def supports(self, file_path: Path) -> bool:
        """Check if file is a PDF."""
        return file_path.suffix.lower() == ".pdf"
