"""Factory for creating embedding providers."""

from typing import TYPE_CHECKING

from src.embeddings.base import EmbeddingProvider
from src.embeddings.huggingface_embeddings import HuggingFaceEmbeddingProvider

if TYPE_CHECKING:
    from src.agent.config_loader import AgentConfig


class EmbeddingFactory:
    """Factory for creating embedding providers."""

    @staticmethod
    def create(model_name: str, provider: str = "huggingface") -> EmbeddingProvider:
        """
        Create an embedding provider.

        Args:
            model_name: Name of the embedding model.
            provider: Provider type ("huggingface" supported).

        Returns:
            EmbeddingProvider instance.

        Raises:
            ValueError: If provider is not supported.
        """
        if provider == "huggingface":
            return HuggingFaceEmbeddingProvider(model_name=model_name)
        raise ValueError(f"Unsupported embedding provider: {provider}")

    @staticmethod
    def create_from_agent_config(config: "AgentConfig") -> EmbeddingProvider:
        """
        Create an embedding provider from AgentConfig.

        Args:
            config: Agent configuration.

        Returns:
            EmbeddingProvider instance.
        """
        return EmbeddingFactory.create(
            model_name=config.embedding_model,
            provider=config.embedding_provider,
        )
