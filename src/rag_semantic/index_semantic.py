from __future__ import annotations

from src.rag_semantic.config import RagConfig
from src.rag_semantic.semantic_model import load_semantic_model
from src.rag_semantic.doc_builder import build_embedding_docs
from src.rag_semantic.embedder_local import LocalEmbedder
from src.rag_semantic.lancedb_store import recreate_table
from src.rag_semantic.doc_sanitize import sanitize_docs_for_lancedb


def main():
    cfg = RagConfig()
    cfg.validate()

    model = load_semantic_model(str(cfg.semantic_yaml_path))
    docs = build_embedding_docs(model, source=str(cfg.semantic_yaml_path))

    embedder = LocalEmbedder(cfg.embedding_model_name)
    vectors = embedder.embed_texts([d["text"] for d in docs], batch_size=32)

    for d, v in zip(docs, vectors):
        d["vector"] = v

    rows = sanitize_docs_for_lancedb(docs)
    recreate_table(str(cfg.lancedb_path), cfg.lancedb_table, rows)

    print(f"Indexed {len(rows)} docs into LanceDB")
    print(f"  path:  {cfg.lancedb_path}")
    print(f"  table: {cfg.lancedb_table}")
    print(f"  embed: {cfg.embedding_model_name}")


if __name__ == "__main__":
    main()
