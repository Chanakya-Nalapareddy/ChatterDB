from __future__ import annotations

from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple

Relationship = Dict[str, str]


def _build_graph(rels: List[Relationship]) -> Dict[str, Set[str]]:
    g = defaultdict(set)
    for r in rels:
        ft = r.get("from_table")
        tt = r.get("to_table")
        if ft and tt:
            g[ft].add(tt)
            g[tt].add(ft)
    return g


def _shortest_path_tables(
    graph: Dict[str, Set[str]],
    start: str,
    goal: str,
) -> Set[str]:
    if start == goal:
        return {start}

    q = deque([(start, {start})])
    seen = {start}

    while q:
        node, path = q.popleft()
        for nxt in graph.get(node, []):
            if nxt in seen:
                continue
            new_path = path | {nxt}
            if nxt == goal:
                return new_path
            seen.add(nxt)
            q.append((nxt, new_path))

    return set()


def prune_tables_by_connectivity(
    required_tables: List[str],
    all_relationships: List[Relationship],
) -> List[str]:
    """
    Keep only tables that lie on join paths connecting required tables.
    """
    if len(required_tables) <= 1:
        return required_tables

    graph = _build_graph(all_relationships)

    kept: Set[str] = set()
    anchor = required_tables[0]

    for t in required_tables[1:]:
        path_tables = _shortest_path_tables(graph, anchor, t)
        kept |= path_tables

    return sorted(kept)
