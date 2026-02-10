# prim.py
from __future__ import annotations

import argparse
import heapq
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from functions import (
    parse_tsplib_tsp,
    build_complete_graph_from_coords,
    edges_to_adjacency,
)
from plotting import plot_points_interactive, plot_mst_interactive

MSTEdge = Tuple[int, int, int]  # (u, v, w)


def prim_mst(
    adj: Dict[int, List[Tuple[int, int]]],
    start: Optional[int] = None,
    return_steps: bool = False,
) -> Tuple[List[MSTEdge], int, Dict[str, int], Dict[int, int], int] | Tuple[List[MSTEdge], int, Dict[str, int], Dict[int, int], int, List[List[MSTEdge]]]:
    """
    Prim's algorithm using a lazy min-heap.

    Returns:
      mst_edges, total_weight, op_metrics, order_added, n_it
    """
    nodes = list(adj.keys())
    if not nodes:
        return [], 0, {"heap_pushes": 0, "heap_pops": 0}, {}, 0

    if start is None:
        start = min(nodes)
    if start not in adj:
        raise ValueError(f"Start node {start} is not in the graph.")

    n = len(nodes)
    in_mst = set([start])
    n_it = 0

    mst_edges: List[MSTEdge] = []
    total_weight = 0

    steps: List[List[MSTEdge]] = []

    order_added: Dict[int, int] = {start: 0}
    next_order = 1

    heap: List[Tuple[int, int, int]] = []
    op_metrics: Dict[str, int] = {"heap_pushes": 0, "heap_pops": 0}

    for v, w in adj[start]:
        heapq.heappush(heap, (w, start, v))
        op_metrics["heap_pushes"] += 1

    while heap and len(mst_edges) < n - 1:
        n_it += 1
        w, u, v = heapq.heappop(heap)
        op_metrics["heap_pops"] += 1

        if v in in_mst:
            continue

        in_mst.add(v)
        order_added[v] = next_order
        next_order += 1
        mst_edges.append((u, v, w))
        total_weight += w

        # record this accepted edge as a single step
        steps.append([(u, v, w)])

        for nxt, w2 in adj[v]:
            if nxt not in in_mst:
                heapq.heappush(heap, (w2, v, nxt))
                op_metrics["heap_pushes"] += 1

    if return_steps:
        return mst_edges, total_weight, op_metrics, order_added, n_it, steps
    return mst_edges, total_weight, op_metrics, order_added, n_it


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prim (lazy heap) on a TSPLIB .tsp instance.")
    parser.add_argument("--tsp", required=True, help="Path to TSPLIB .tsp file")
    parser.add_argument("--start", type=int, default=None, help="Optional start node id")
    parser.add_argument("--plot", action="store_true", help="Show plots (default behavior)")
    args = parser.parse_args()

    inst = parse_tsplib_tsp(Path(args.tsp))
    nodes, edges = build_complete_graph_from_coords(inst.coords)
    adj = edges_to_adjacency(nodes, edges)

    n = len(nodes)
    m = len(edges)
    start_node = args.start if args.start is not None else min(nodes)

    t0 = time.perf_counter()
    mst_edges, total_weight, op_metrics, order_added, n_it = prim_mst(adj, start=start_node)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    print(f"prim | {inst.name} | n={n} m={m} | W={total_weight} | n_it={n_it} | "
          f"heap_push={op_metrics['heap_pushes']} heap_pop={op_metrics['heap_pops']} | "
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
        title=f"{inst.name} | Prim MST | W={total_weight} | time={elapsed_ms:.2f}ms",
        start_node=start_node,
        order_added=order_added,
        show=True,
    )


if __name__ == "__main__":
    main()
