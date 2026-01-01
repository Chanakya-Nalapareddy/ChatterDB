from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _walk(obj: Any, path: str = "") -> List[Tuple[str, Any]]:
    """
    Walk the nested YAML structure and return (path, value) pairs for dict keys.
    """
    out: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            out.append((p, v))
            out.extend(_walk(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            out.extend(_walk(v, p))
    return out


def inspect_semantic_yaml(path: str, sample_limit: int = 8) -> None:
    data = load_yaml(path)

    print("\n=== Top-level keys ===")
    if isinstance(data, dict):
        print(sorted(list(data.keys())))
    else:
        print(type(data))

    pairs = _walk(data)

    # Count key names
    key_counter = Counter()
    for p, _ in pairs:
        key = p.split(".")[-1]
        if "[" in key:
            key = key.split("[")[0]
        key_counter[key] += 1

    print("\n=== Most common keys (top 30) ===")
    for k, c in key_counter.most_common(30):
        print(f"{k}: {c}")

    # Look for join-ish keys
    needles = ["relationship", "relationships", "join", "joins", "foreign", "fk", "ref", "refs", "reference", "links"]
    hits = [(p, v) for (p, v) in pairs if any(n in p.lower() for n in needles)]

    print(f"\n=== Paths matching {needles} (showing up to {sample_limit}) ===")
    for i, (p, v) in enumerate(hits[:sample_limit], start=1):
        print(f"\n[{i}] {p}")
        # Print a short preview
        if isinstance(v, (dict, list)):
            import json
            try:
                s = json.dumps(v, indent=2)[:800]
            except Exception:
                s = str(v)[:800]
            print(s)
        else:
            print(str(v)[:800])

    if not hits:
        print("\n(No join-like keys found. Relationships might be encoded differently.)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.rag_semantic.semantic_inspect <path-to-yaml>")
    inspect_semantic_yaml(sys.argv[1])
