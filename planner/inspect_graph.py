"""CLI: query the dependency graph built from co-access counters.

`--record <id>` is the Stage 5 "simple query interface: given a record,
return its graph neighbors above the co-access threshold" deliverable.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.storage import DEFAULT_DB_PATH  # noqa: E402
from planner.dependency_graph import DEFAULT_MIN_CO_ACCESS, DependencyGraph  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Query the dependency graph built from co-access counters")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--min-co-access", type=int, default=DEFAULT_MIN_CO_ACCESS)
    parser.add_argument("--record", help="show neighbors of this record_id")
    parser.add_argument("--top", type=int, default=10, help="show the N most-connected records if --record is omitted")
    args = parser.parse_args()

    graph = DependencyGraph.build(args.db, min_co_access=args.min_co_access)
    print(f"graph: {len(graph.records())} records, {graph.num_edges()} edges, "
          f"{len(graph.connected_components())} connected components (min_co_access={args.min_co_access})")

    if args.record:
        neighbors = graph.neighbors(args.record)
        if not neighbors:
            print(f"\n{args.record} has no neighbors above the threshold")
            return
        print(f"\nneighbors of {args.record}:")
        for neighbor, count in neighbors.items():
            print(f"  {neighbor:<45} co_access={count}")
        return

    degree_sorted = sorted(graph.records(), key=lambda r: -len(graph.neighbors(r)))[: args.top]
    print(f"\ntop {args.top} most-connected records:")
    for record_id in degree_sorted:
        neighbors = graph.neighbors(record_id)
        neighbor_str = ", ".join(f"{n}({c})" for n, c in list(neighbors.items())[:3])
        print(f"  {record_id:<45} degree={len(neighbors)}  neighbors: {neighbor_str}")


if __name__ == "__main__":
    main()
