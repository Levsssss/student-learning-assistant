"""Base class for document loaders using Strategy pattern."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from langchain_core.documents import Document


class DocumentLoaderStrategy(ABC):
    """Abstract base class for document loading strategies."""

    @abstractmethod
    def load(self, file_path: Path) -> List[Document]:
        """
        Load documents from the given file path.

        Args:
            file_path: Path to the file to load.

        Returns:
            List of Document objects.
        """
        pass

    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """
        Check if this loader supports the given file type.

        Args:
            file_path: Path to check.

        Returns:
            True if this loader can handle the file type.
        """
        pass
