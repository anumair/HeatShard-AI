"""Render the dependency graph as a PNG -- the Stage 5 sanity check:
each connected component (expected: one product's product/review/order
triangle) is drawn as its own small cluster, confirming the graph
reflects exactly the relationships the simulator modeled, with no
unexpected cross-links between unrelated products.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from collector.storage import DEFAULT_DB_PATH  # noqa: E402
from planner.dependency_graph import DEFAULT_MIN_CO_ACCESS, DependencyGraph  # noqa: E402

DEFAULT_OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "dependency_graph.png"

NODE_COLORS = {"product": "#4C72B0", "review": "#DD8452", "reviews": "#DD8452", "order": "#55A868", "inventory": "#55A868"}


def short_label(record_id: str) -> str:
    """Just the id portion (color already encodes product/review/order)."""
    _, _, rest = record_id.partition(":")
    return rest[:8] if rest else record_id


def layout_component(component: set, center: tuple) -> dict:
    nodes = list(component)
    n = len(nodes)
    if n == 1:
        return {nodes[0]: center}
    radius = 0.4
    return {
        node: (center[0] + radius * math.cos(2 * math.pi * i / n), center[1] + radius * math.sin(2 * math.pi * i / n))
        for i, node in enumerate(nodes)
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize the dependency graph")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--min-co-access", type=int, default=DEFAULT_MIN_CO_ACCESS)
    parser.add_argument("--max-components", type=int, default=12, help="how many clusters to draw")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    args = parser.parse_args()

    graph = DependencyGraph.build(args.db, min_co_access=args.min_co_access)
    components = sorted(graph.connected_components(), key=len, reverse=True)[: args.max_components]

    if not components:
        print("no edges above the threshold -- nothing to plot")
        return

    cols = 4
    rows = math.ceil(len(components) / cols)
    fig, ax = plt.subplots(figsize=(cols * 2.8, rows * 2.8))

    positions = {}
    for idx, component in enumerate(components):
        center = ((idx % cols) * 1.3, -(idx // cols) * 1.3)
        positions.update(layout_component(component, center))

    for a, b, _weight in graph.all_edges():
        if a in positions and b in positions:
            x1, y1 = positions[a]
            x2, y2 = positions[b]
            ax.plot([x1, x2], [y1, y2], color="gray", linewidth=0.6, zorder=1)

    for node, (x, y) in positions.items():
        prefix = node.split(":")[0]
        color = NODE_COLORS.get(prefix, "#888888")
        ax.scatter([x], [y], s=420, color=color, zorder=2)
        ax.annotate(short_label(node), (x, y), fontsize=6, ha="center", va="center", color="white", zorder=3)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=label)
        for label, c in [("product", NODE_COLORS["product"]), ("review", NODE_COLORS["review"]), ("order", NODE_COLORS["order"])]
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_title(
        f"Dependency graph: {len(graph.records())} records, {graph.num_edges()} edges, "
        f"{len(components)}/{len(graph.connected_components())} components shown"
    )
    ax.axis("off")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
