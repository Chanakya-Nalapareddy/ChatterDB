from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]   # .../src
PROJECT_ROOT = BASE_DIR.parent                   # project root
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class RagConfig:
    # Paths (reuse existing project assets)
    semantic_yaml_path: Path = Path(
        os.getenv(
            "RAG_SEMANTIC_YAML_PATH",
            str(PROJECT_ROOT / "src" / "chatterdb" / "catalog" / "chatterdb_semantic_model.yaml"),
        )
    )
    duckdb_path: Path = Path(
        os.getenv(
            "RAG_DUCKDB_PATH",
            str(PROJECT_ROOT / "data" / "warehouse" / "chatterdb.duckdb"),
        )
    )

    # Vector DB (LanceDB)
    vectordb: str = os.getenv("RAG_VECTORDB", "lancedb")
    lancedb_path: Path = Path(os.getenv("RAG_LANCEDB_PATH", str(PROJECT_ROOT / ".lancedb")))
    lancedb_table: str = os.getenv("RAG_LANCEDB_TABLE", "semantic_index_v1")

    # Embeddings (open-source)
    embedding_provider: str = os.getenv("RAG_EMBEDDING_PROVIDER", "local")
    embedding_model_name: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    def validate(self) -> None:
        if not self.semantic_yaml_path.exists():
            raise FileNotFoundError(f"Semantic YAML not found: {self.semantic_yaml_path}")
        if self.vectordb != "lancedb":
            raise ValueError(f"Unsupported vectordb: {self.vectordb}")
        if self.embedding_provider != "local":
            raise ValueError(f"Unsupported embedding_provider: {self.embedding_provider}")
