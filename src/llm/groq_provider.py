"""Groq LLM provider implementation."""

import logging
import os
from typing import Any

from langchain_community.chat_models import ChatOpenAI
from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """LLM provider using Groq API."""

    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.1):
        """
        Initialize the Groq provider.

        Args:
            model: Groq model name.
            temperature: Temperature for generation.
        """

        self._model = model
        self._temperature = temperature

        # Use OpenAI-compatible endpoint from Groq
        self._llm = ChatOpenAI(
            openai_api_key=os.getenv("GROQ_API_KEY"),
            openai_api_base="https://api.groq.com/openai/v1",
            model_name=model,
            temperature=temperature
        )

        logger.info(f"Initialized Groq provider with model: {model}")

    def get_llm(self) -> Any:
        """Get the Groq LLM instance."""
        return self._llm

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model