from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

# --- hardcoded semantic hints (safe + transparent) ---
ENTITY_KEYWORDS = {
    "artist": "source.Artist",
    "album": "source.Album",
    "track": "source.Track",
    "playlist": "source.Playlist",
}

BRIDGE_TABLES = {
    "source.PlaylistTrack",
    "source.InvoiceLine",
    "source.OrderItem",
}


def select_tables(
    question: str,
    hits: List[Dict],
    max_tables: int = 3,
) -> List[str]:
    q = question.lower()
    score = defaultdict(float)

    # -------------------------
    # 1) Score from vector hits
    # -------------------------
    for rank, h in enumerate(hits):
        weight = 1.0 / (1.0 + rank)
        dt = h.get("doc_type")
        table = h.get("table")
        meta = h.get("meta") or {}

        if dt == "table" and table:
            score[table] += 3.0 * weight
        elif dt == "column" and table:
            score[table] += 2.0 * weight
        elif dt == "relationship":
            if meta.get("from_table"):
                score[meta["from_table"]] += 1.0 * weight
            if meta.get("to_table"):
                score[meta["to_table"]] += 1.0 * weight

    # --------------------------------
    # 2) Enforce explicit entity intent
    # --------------------------------
    enforced = set()
    for kw, table in ENTITY_KEYWORDS.items():
        if kw in q:
            enforced.add(table)
            score[table] += 10.0  # strong bias

    # -----------------------------
    # 3) Penalize bridge tables
    # -----------------------------
    if not any(k in q for k in ["playlist", "invoice", "order"]):
        for bt in BRIDGE_TABLES:
            score[bt] -= 5.0

    # -----------------------------
    # 4) Final selection
    # -----------------------------
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    picked = [t for t, _ in ranked if t]

    # Ensure enforced tables are included
    for t in enforced:
        if t not in picked:
            picked.insert(0, t)

    return picked[:max_tables]
