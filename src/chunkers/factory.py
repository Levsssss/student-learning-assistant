"""Factory for creating document chunkers."""

from pathlib import Path
from typing import TYPE_CHECKING

from src.chunkers.base import ChunkingStrategy
from src.chunkers.recursive_chunker import RecursiveChunkingStrategy
from src.chunkers.markdown_chunker import MarkdownHeaderChunkingStrategy

if TYPE_CHECKING:
    from src.agent.config_loader import AgentConfig


class ChunkerFactory:
    """Factory for creating appropriate chunking strategies."""

    @staticmethod
    def create(
        file_path: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_md_headers: bool = True,
    ) -> ChunkingStrategy:
        """
        Create appropriate chunker based on file type.

        Args:
            file_path: Path to determine chunking strategy.
            chunk_size: Size of each chunk.
            chunk_overlap: Overlap between chunks.
            use_md_headers: Whether to use header-based splitting for Markdown.

        Returns:
            Appropriate ChunkingStrategy instance.
        """
        if file_path.suffix.lower() == ".md" and use_md_headers:
            return MarkdownHeaderChunkingStrategy(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        return RecursiveChunkingStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def create_from_agent_config(
        file_path: Path,
        config: "AgentConfig",
    ) -> ChunkingStrategy:
        """
        Create a chunker from AgentConfig.

        Args:
            file_path: Path to determine chunking strategy.
            config: Agent configuration.

        Returns:
            Appropriate ChunkingStrategy instance.
        """
        return ChunkerFactory.create(
            file_path=file_path,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            use_md_headers=config.use_md_headers,
        )
