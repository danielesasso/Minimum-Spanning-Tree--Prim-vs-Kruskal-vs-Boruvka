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
) -> Tuple[List[MSTEdge], int, Dict[str, int], Dict[int, int]]:
    nodes = list(adj.keys())
    if not nodes:
        return [], 0, {"heap_pushes": 0, "heap_pops": 0, "edges_considered": 0}, {}

    if start is None:
        start = min(nodes)
    if start not in adj:
        raise ValueError(f"Start node {start} is not in the graph.")

    n = len(nodes)
    in_mst = set([start])

    mst_edges: List[MSTEdge] = []
    total_weight = 0

    # For tooltips: order node was added to MST
    order_added: Dict[int, int] = {start: 0}
    next_order = 1

    heap: List[Tuple[int, int, int]] = []  # (w, u, v)

    metrics: Dict[str, int] = {
        "heap_pushes": 0,
        "heap_pops": 0,
        "edges_considered": 0,
    }

    for v, w in adj[start]:
        heapq.heappush(heap, (w, start, v))
        metrics["heap_pushes"] += 1

    while heap and len(mst_edges) < n - 1:
        w, u, v = heapq.heappop(heap)
        metrics["heap_pops"] += 1

        if v in in_mst:
            continue

        in_mst.add(v)
        order_added[v] = next_order
        next_order += 1

        mst_edges.append((u, v, w))
        total_weight += w

        for nxt, w2 in adj[v]:
            metrics["edges_considered"] += 1
            if nxt not in in_mst:
                heapq.heappush(heap, (w2, v, nxt))
                metrics["heap_pushes"] += 1

    return mst_edges, total_weight, metrics, order_added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Prim (lazy heap) on a TSPLIB .tsp instance."
    )
    parser.add_argument("--tsp", required=True, help="Path to TSPLIB .tsp file")
    parser.add_argument("--start", type=int, default=None, help="Optional start node id")
    parser.add_argument("--plot", action="store_true", help="Show plots (nodes only + MST)")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to results/plots/")
    parser.add_argument("--no-show", action="store_true", help="Do not display plots")

    args = parser.parse_args()

    tsp_path = Path(args.tsp)
    inst = parse_tsplib_tsp(tsp_path)

    nodes, edges = build_complete_graph_from_coords(inst.coords)
    adj = edges_to_adjacency(nodes, edges)

    start_node = args.start if args.start is not None else min(nodes)

    t0 = time.perf_counter()
    mst_edges, total_weight, metrics, order_added = prim_mst(adj, start=start_node)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    expected_edges = inst.dimension - 1
    ok_size = (len(mst_edges) == expected_edges)

    print("=== Prim (lazy heap) ===")
    print(f"Instance: {inst.name}")
    print(f"EDGE_WEIGHT_TYPE: {inst.edge_weight_type}")
    print(f"Start node: {start_node}")
    print(f"n={inst.dimension}  m={len(edges)} (complete graph)")
    print(f"MST edges: {len(mst_edges)} (expected {expected_edges})  ok={ok_size}")
    print(f"MST total weight: {total_weight}")
    print(f"Time: {elapsed_ms:.3f} ms")
    print("Metrics:")
    for k in ["heap_pushes", "heap_pops", "edges_considered"]:
        print(f"  {k}: {metrics.get(k, 0)}")

    show = (not args.no_show)

    if args.plot or args.save_plots:
        save0 = Path("results/plots") / f"{inst.name}_nodes.png" if args.save_plots else None
        title0 = f"{inst.name} | nodes only | start={start_node}"
        plot_points_interactive(
            inst.coords,
            title=title0,
            start_node=start_node,
            save_path=save0,
            show=show if args.plot else False,
        )

        save1 = Path("results/plots") / f"{inst.name}_prim_mst.png" if args.save_plots else None
        title1 = f"{inst.name} | Prim MST | W={total_weight} | time={elapsed_ms:.2f}ms"
        plot_mst_interactive(
            inst.coords,
            mst_edges,
            title=title1,
            start_node=start_node,
            order_added=order_added,  # <-- qui invece serve
            save_path=save1,
            show=show if args.plot else False,
        )


if __name__ == "__main__":
    main()
