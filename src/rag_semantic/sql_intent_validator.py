from __future__ import annotations

import re


def _has(sql: str, pattern: str) -> bool:
    return re.search(pattern, sql or "", flags=re.IGNORECASE) is not None


def validate_sql_matches_intent(question: str, sql: str) -> None:
    """
    Lightweight intent validator to prevent "answers the wrong dimension".

    Raises ValueError if SQL doesn't match required intent signals.
    """
    q = (question or "").lower()
    s = sql or ""

    # "by support rep" must reference Employee or SupportRepId
    if "support rep" in q or "supportrep" in q or "support representative" in q:
        ok = _has(s, r"\bsource\.Employee\b") or _has(s, r"\bSupportRepId\b")
        if not ok:
            raise ValueError(
                "Intent mismatch: question asks 'by support rep' but SQL does not reference source.Employee or SupportRepId."
            )

        # Also discourage accidental customer-grouping without rep info
        if _has(s, r"\bGROUP\s+BY\b") and _has(s, r"\bCustomerId\b") and not _has(s, r"\bEmployeeId\b|\bSupportRepId\b"):
            raise ValueError(
                "Intent mismatch: SQL appears grouped by customer, not support rep."
            )

    # "by customer" must reference CustomerId or source.Customer
    if "by customer" in q:
        ok = _has(s, r"\bCustomerId\b") or _has(s, r"\bsource\.Customer\b")
        if not ok:
            raise ValueError("Intent mismatch: question asks 'by customer' but SQL does not reference Customer.")

    # If aggregates exist, GROUP BY should exist (basic sanity)
    if _has(s, r"\bSUM\(|\bCOUNT\(|\bAVG\(|\bMIN\(|\bMAX\(") and not _has(s, r"\bGROUP\s+BY\b"):
        # Allow purely scalar aggregates without GROUP BY (rare but valid)
        # e.g., SELECT SUM(...) FROM ...
        # If user said "by X", this should have been caught earlier anyway.
        return
