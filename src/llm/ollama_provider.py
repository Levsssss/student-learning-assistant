"""Ollama LLM provider implementation."""

import logging
from typing import Any

from langchain_community.chat_models import ChatOllama

from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider using local Ollama."""

    def __init__(self, model: str = "qwen2.5:1.5b", temperature: float = 0.1):
        """
        Initialize the Ollama provider.

        Args:
            model: Ollama model name.
            temperature: Temperature for generation.
        """
        self._model = model
        self._temperature = temperature
        self._llm = ChatOllama(model=model, temperature=temperature)
        logger.info(f"Initialized Ollama provider with model: {model}")

    def get_llm(self) -> Any:
        """Get the ChatOllama instance."""
        return self._llm

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model
