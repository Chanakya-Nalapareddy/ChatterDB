from __future__ import annotations

from typing import List


# Keyword -> required table mapping (explicit anchors).
ENTITY_KEYWORDS = {
    # --- music domain ---
    "artist": "source.Artist",
    "album": "source.Album",
    "track": "source.Track",
    "playlist": "source.Playlist",

    # --- revenue / commerce domain ---
    "customer": "source.Customer",
    "invoice": "source.Invoice",
    "country": "source.Customer",
    "billingcountry": "source.Invoice",
    "revenue": "source.InvoiceLine",
    "sales": "source.InvoiceLine",
    "spend": "source.InvoiceLine",
    "spent": "source.InvoiceLine",

    # --- org / reps ---
    "support rep": "source.Employee",
    "supportrep": "source.Employee",
    "support representative": "source.Employee",
    "employee": "source.Employee",
    "rep": "source.Employee",
}


def extract_required_tables(question: str) -> List[str]:
    """
    Extract authoritative "required tables" from explicit intent keywords.
    These anchors drive graph pruning and prevent hallucinated tables.
    """
    q = (question or "").lower()
    required: List[str] = []

    for kw, table in ENTITY_KEYWORDS.items():
        if kw in q and table not in required:
            required.append(table)

    # Revenue rule anchor:
    # If revenue/sales/spend requested, include InvoiceLine and Invoice.
    # (Even if model could answer using Invoice.Total, keeping InvoiceLine present
    # helps avoid later drilldowns hallucinating.)
    if any(k in q for k in ["revenue", "sales", "spend", "spent"]):
        if "source.InvoiceLine" not in required:
            required.append("source.InvoiceLine")
        if "source.Invoice" not in required:
            required.append("source.Invoice")

    # Country implies Customer dimension
    if "country" in q and "source.Customer" not in required:
        required.append("source.Customer")

    # Support rep implies Employee + Customer + Invoice (to connect revenue)
    if any(k in q for k in ["support rep", "supportrep", "support representative"]):
        if "source.Employee" not in required:
            required.append("source.Employee")
        if "source.Customer" not in required:
            required.append("source.Customer")
        if "source.Invoice" not in required:
            required.append("source.Invoice")

    return required
