"""Document loaders with Strategy pattern."""

from src.loaders.base import DocumentLoaderStrategy
from src.loaders.pdf_loader import PDFLoaderStrategy
from src.loaders.markdown_loader import MarkdownLoaderStrategy
from src.loaders.factory import DocumentLoaderFactory

__all__ = [
    "DocumentLoaderStrategy",
    "PDFLoaderStrategy",
    "MarkdownLoaderStrategy",
    "DocumentLoaderFactory",
]
