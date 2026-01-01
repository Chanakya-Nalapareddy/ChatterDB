from __future__ import annotations

from typing import Any, Dict

from src.rag_semantic.config import RagConfig
from src.rag_semantic.retrieve_semantic import retrieve
from src.rag_semantic.semantic_model import load_semantic_model
from src.rag_semantic.join_planner import plan_join_edges
from src.rag_semantic.table_selector import select_tables
from src.rag_semantic.table_pruner import prune_tables_by_connectivity
from src.rag_semantic.intent_required_tables import extract_required_tables


def plan(question: str, k: int = 12) -> Dict[str, Any]:
    """
    End-to-end semantic planning for a natural language question.

    Pipeline:
      1) Vector retrieval        -> recall
      2) Table selection         -> evidence (debugging only)
      3) Intent extraction       -> required entities (truth)
      4) Graph pruning           -> minimal correct tables
      5) Join planning           -> ordered join edges
    """
    cfg = RagConfig()
    cfg.validate()

    # -------------------------------------------------
    # 1) Vector retrieval (recall-oriented)
    # -------------------------------------------------
    retrieval = retrieve(question, k=k)

    # -------------------------------------------------
    # 2) Load full semantic model (ground truth graph)
    # -------------------------------------------------
    semantic_model = load_semantic_model(str(cfg.semantic_yaml_path))
    all_relationships = semantic_model["relationships"]

    # -------------------------------------------------
    # 3) Evidence-based table selection (NOT authoritative)
    #    Used only for debugging / observability
    # -------------------------------------------------
    picked_tables = select_tables(
        question=question,
        hits=retrieval["hits"],
        max_tables=5,
    )

    # -------------------------------------------------
    # 4) Intent-based REQUIRED tables (authoritative)
    # -------------------------------------------------
    required_tables = extract_required_tables(question)

    # -------------------------------------------------
    # 5) Graph pruning:
    #    Keep only tables needed to connect REQUIRED tables
    # -------------------------------------------------
    pruned_tables = prune_tables_by_connectivity(
        required_tables=required_tables,
        all_relationships=all_relationships,
    )

    # -------------------------------------------------
    # 6) Join planning on pruned tables only
    # -------------------------------------------------
    join_edges = plan_join_edges(
        required_tables=pruned_tables,
        all_relationships=all_relationships,
    )

    return {
        "question": question,

        # Observability / debugging
        "retrieved_tables": retrieval["tables"],
        "picked_tables": picked_tables,

        # Truth / correctness
        "required_tables": required_tables,
        "pruned_tables": pruned_tables,
        "join_edges": join_edges,

        # Optional debugging preview
        "hit_preview": [
            (h["doc_type"], h.get("table"), h.get("column"))
            for h in retrieval["hits"][:8]
        ],
    }


if __name__ == "__main__":
    out = plan("tracks by artist", k=12)

    print("\nQuestion:")
    print(out["question"])

    print("\nRetrieved tables (recall):")
    print(out["retrieved_tables"])

    print("\nPicked tables (evidence only):")
    print(out["picked_tables"])

    print("\nRequired tables (intent / truth):")
    print(out["required_tables"])

    print("\nPruned tables (graph-correct):")
    print(out["pruned_tables"])

    print("\nPlanned join edges:")
    for e in out["join_edges"]:
        print(
            "-",
            f"{e['from_table']}.{e['from_column']}",
            "->",
            f"{e['to_table']}.{e['to_column']}",
            f"(join: {e.get('join_type')})",
        )
