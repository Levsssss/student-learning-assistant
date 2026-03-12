"""YAML configuration loader for RAG agents."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration loaded from agent.yaml."""

    # Agent identity
    name: str
    description: str

    # Teacher authentication
    teacher_password: str

    # Prompts
    system_prompt: str
    human_prompt: str

    # Model settings
    model_provider: str
    model_name: str
    temperature: float

    # Embedding settings
    embedding_provider: str
    embedding_model: str

    # RAG settings
    context_file: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    use_md_headers: bool
    persist_dir: str

    # Test cases
    test_cases: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            AgentConfig instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If required fields are missing.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        logger.info(f"Loading agent configuration from: {path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        return cls._parse_config(data)

    @classmethod
    def _parse_config(cls, data: dict) -> "AgentConfig":
        """Parse YAML data into AgentConfig."""
        prompts = data.get("prompts", {})
        model = data.get("model", {})
        embeddings = data.get("embeddings", {})
        rag = data.get("rag", {})
        teacher = data.get("teacher", {})
        return cls(
            # Agent identity
            name=data.get("name", "RAG Agent"),
            description=data.get("description", ""),

            # Teacher authentication
            teacher_password=teacher.get("password", "prof123"),

            # Prompts
            system_prompt=prompts.get("system", "You are a helpful assistant.\n\nContext: {context}"),
            human_prompt=prompts.get("human", "{input}"),

            # Model settings
            model_provider=model.get("provider", "ollama"),
            model_name=model.get("name", "qwen2.5:1.5b"),
            temperature=model.get("temperature", 0.1),

            # Embedding settings
            embedding_provider=embeddings.get("provider", "huggingface"),
            embedding_model=embeddings.get("model", "all-MiniLM-L6-v2"),

            # RAG settings
            context_file=rag.get("context_file", "context"),
            chunk_size=rag.get("chunk_size", 1000),
            chunk_overlap=rag.get("chunk_overlap", 200),
            retriever_k=rag.get("retriever_k", 3),
            use_md_headers=rag.get("use_md_headers", True),
            persist_dir=rag.get("persist_dir", "./chroma_db"),

            # Test cases
            test_cases=data.get("test_cases", []),
        )

    def to_yaml(self, path: str) -> None:
        """
        Save configuration to a YAML file.

        Args:
            path: Path to save the configuration.
        """
        data = {
            "name": self.name,
            "description": self.description,
            "teacher": {
                "password": self.teacher_password
            },
            "prompts": {
                "system": self.system_prompt,
                "human": self.human_prompt,
            },
            "model": {
                "provider": self.model_provider,
                "name": self.model_name,
                "temperature": self.temperature,
            },
            "embeddings": {
                "provider": self.embedding_provider,
                "model": self.embedding_model,
            },
            "rag": {
                "context_file": self.context_file,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "retriever_k": self.retriever_k,
                "use_md_headers": self.use_md_headers,
                "persist_dir": self.persist_dir,
            },
            "test_cases": self.test_cases,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to: {path}")

import yaml
from pathlib import Path


def load_config(path: str) -> AgentConfig:
    return AgentConfig.from_yaml(path)
