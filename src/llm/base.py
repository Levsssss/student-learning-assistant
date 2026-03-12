"""Base class for LLM providers using Strategy pattern."""

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_llm(self) -> Any:
        """
        Get the LLM instance.

        Returns:
            LLM instance compatible with LangChain.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass
