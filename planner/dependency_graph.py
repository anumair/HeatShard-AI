"""Component 4: Partition-Level Heat Maps / Dependency Graph.

Builds a sparse graph of "records that tend to be accessed together"
directly from Component 1's co-access counters (co_access_windows
table), thresholding out low-count pairs to keep it sparse. The
Relocation Planner (Stage 6) queries this graph to avoid separating
frequently-related records across shards (CrossShardPenalty).
"""

import sqlite3
from collections import defaultdict

from collector.storage import DEFAULT_DB_PATH

DEFAULT_MIN_CO_ACCESS = 3  # aggregate co-access count below this is dropped as noise


class DependencyGraph:
    def __init__(self, edges: dict):
        """edges: {frozenset({a, b}): aggregate_count}, already thresholded."""
        self._adjacency = defaultdict(dict)  # record_id -> {neighbor_id: co_access_count}
        for pair, count in edges.items():
            a, b = tuple(pair)
            self._adjacency[a][b] = count
            self._adjacency[b][a] = count

    def neighbors(self, record_id: str) -> dict:
        """{neighbor_id: co_access_count}, sorted by count descending."""
        return dict(sorted(self._adjacency.get(record_id, {}).items(), key=lambda kv: -kv[1]))

    def has_edge(self, a: str, b: str) -> bool:
        return b in self._adjacency.get(a, {})

    def edge_weight(self, a: str, b: str) -> int:
        return self._adjacency.get(a, {}).get(b, 0)

    def records(self) -> list:
        return list(self._adjacency.keys())

    def all_edges(self) -> list:
        """[(a, b, weight), ...], each undirected edge listed once."""
        seen = set()
        edges = []
        for a, neighbors in self._adjacency.items():
            for b, weight in neighbors.items():
                key = frozenset((a, b))
                if key in seen:
                    continue
                seen.add(key)
                edges.append((a, b, weight))
        return edges

    def num_edges(self) -> int:
        return len(self.all_edges())

    def connected_components(self) -> list:
        """List of sets of record_ids -- in this simulator, each component is
        expected to be exactly one product's product/review/order triangle,
        with no cross-links to other products' records."""
        visited = set()
        components = []
        for start in self._adjacency:
            if start in visited:
                continue
            stack = [start]
            component = set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                visited.add(node)
                stack.extend(n for n in self._adjacency[node] if n not in component)
            components.append(component)
        return components

    @classmethod
    def build(cls, db_path=None, min_co_access: int = DEFAULT_MIN_CO_ACCESS) -> "DependencyGraph":
        db_path = db_path or DEFAULT_DB_PATH
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT record_a, record_b, SUM(count) as total FROM co_access_windows GROUP BY record_a, record_b"
        ).fetchall()
        conn.close()

        edges = {
            frozenset((record_a, record_b)): total
            for record_a, record_b, total in rows
            if total >= min_co_access
        }
        return cls(edges)
