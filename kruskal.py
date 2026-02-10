# kruskal.py
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

from functions import parse_tsplib_tsp, build_complete_graph_from_coords
from plotting import plot_points_interactive, plot_mst_interactive

MSTEdge = Tuple[int, int, int]  # (u, v, w)


class UnionFind:
    def __init__(self, nodes: List[int]) -> None:
        self.parent = {x: x for x in nodes}
        self.rank = {x: 0 for x in nodes}
        self.find_calls = 0
        self.union_calls = 0

    def find(self, x: int) -> int:
        self.find_calls += 1
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        self.union_calls += 1
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # union by rank
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def kruskal_mst(
    nodes: List[int],
    edges: List[MSTEdge],
    start_node: int = 1,
    return_steps: bool = False,
) -> Tuple[List[MSTEdge], int, Dict[str, int], Dict[int, int], int] | Tuple[List[MSTEdge], int, Dict[str, int], Dict[int, int], int, List[List[MSTEdge]]]:
    """
    Kruskal's algorithm.

    Returns:
      mst_edges, total_weight, op_metrics, order_added, n_it
    """
    uf = UnionFind(nodes)
    edges_sorted = sorted(edges, key=lambda e: e[2])

    mst_edges: List[MSTEdge] = []
    total_weight = 0
    n_it = 0
    steps: List[List[MSTEdge]] = []

    op_metrics: Dict[str, int] = {
        "find_calls": 0,
        "union_calls": 0,
    }

    order_added: Dict[int, int] = {}
    if start_node in set(nodes):
        order_added[start_node] = 0
    next_order = 1

    for u, v, w in edges_sorted:
        n_it += 1
        if len(mst_edges) == len(nodes) - 1:
            break

        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst_edges.append((u, v, w))
            total_weight += w

            # record this accepted edge as a single step
            steps.append([(u, v, w)])

            for node in (u, v):
                if node not in order_added:
                    order_added[node] = next_order
                    next_order += 1

    op_metrics["find_calls"] = uf.find_calls
    op_metrics["union_calls"] = uf.union_calls

    if return_steps:
        return mst_edges, total_weight, op_metrics, order_added, n_it, steps
    return mst_edges, total_weight, op_metrics, order_added, n_it


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kruskal on a TSPLIB .tsp instance.")
    parser.add_argument("--tsp", required=True, help="Path to TSPLIB .tsp file")
    parser.add_argument("--plot", action="store_true", help="Show plots (default behavior)")
    args = parser.parse_args()

    inst = parse_tsplib_tsp(Path(args.tsp))
    nodes, edges = build_complete_graph_from_coords(inst.coords)

    n = len(nodes)
    m = len(edges)
    start_node = 1 if 1 in set(nodes) else min(nodes)

    t0 = time.perf_counter()
    mst_edges, total_weight, op_metrics, order_added, n_it = kruskal_mst(nodes, edges, start_node=start_node)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    print(f"kruskal | {inst.name} | n={n} m={m} | W={total_weight} | n_it={n_it} | "
          f"find={op_metrics['find_calls']} union={op_metrics['union_calls']} | "
          f"time={elapsed_ms:.3f}ms")

    plot_points_interactive(
        inst.coords,
        title=f"{inst.name} | nodes | start={start_node}",
        start_node=start_node,
        show=True,
    )
    plot_mst_interactive(
        inst.coords,
        mst_edges,
        title=f"{inst.name} | Kruskal MST | W={total_weight} | time={elapsed_ms:.2f}ms",
        start_node=start_node,
        order_added=order_added,
        show=True,
    )


if __name__ == "__main__":
    main()
