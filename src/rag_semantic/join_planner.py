from __future__ import annotations

from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Tuple

Relationship = Dict[str, Any]


def _rel_signature(r: Relationship) -> Tuple[str, str, str, str]:
    return (
        str(r.get("from_table", "")),
        str(r.get("from_column", "")),
        str(r.get("to_table", "")),
        str(r.get("to_column", "")),
    )


def build_table_graph(rels: List[Relationship]) -> Dict[str, List[Tuple[str, Relationship]]]:
    """
    Graph: table -> list of (neighbor_table, relationship_edge)
    We treat edges as undirected for path-finding.
    """
    g = defaultdict(list)
    for r in rels:
        ft = r.get("from_table")
        tt = r.get("to_table")
        if not ft or not tt:
            continue
        g[ft].append((tt, r))
        g[tt].append((ft, r))
    return g


def shortest_path_edges(
    graph: Dict[str, List[Tuple[str, Relationship]]],
    start: str,
    goal: str
) -> Optional[List[Relationship]]:
    """
    BFS shortest path returning relationship edges along the path.
    """
    if start == goal:
        return []

    q = deque([(start, [])])
    seen = {start}

    while q:
        node, path = q.popleft()
        for nxt, rel in graph.get(node, []):
            if nxt in seen:
                continue
            new_path = path + [rel]
            if nxt == goal:
                return new_path
            seen.add(nxt)
            q.append((nxt, new_path))

    return None


def plan_join_edges(required_tables: List[str], all_relationships: List[Relationship]) -> List[Relationship]:
    """
    Greedy join planner:
    - Pick anchor = required_tables[0]
    - For each other table, add shortest path edges from anchor to that table
    - Deduplicate edges

    This works well for small schemas like yours (11 tables) and is easy to debug.
    """
    required_tables = [t for t in required_tables if t]
    if len(required_tables) <= 1:
        return []

    graph = build_table_graph(all_relationships)
    anchor = required_tables[0]

    selected: List[Relationship] = []

    for t in required_tables[1:]:
        path = shortest_path_edges(graph, anchor, t)
        if path:
            selected.extend(path)

    uniq = {}
    for r in selected:
        uniq[_rel_signature(r)] = r

    return list(uniq.values())
