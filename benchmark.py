# benchmark.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import time

import matplotlib.pyplot as plt

try:
    import mplcursors
    _HAS_MPLCURSORS = True
except Exception:
    _HAS_MPLCURSORS = False

from utils.functions import parse_tsplib_tsp, build_complete_graph_from_coords, edges_to_adjacency
from kruskal import kruskal_mst
from prim import prim_mst
from boruvka import boruvka_mst
MSTEdge = Tuple[int, int, int]


class InteractiveMSTComparison:
    """Interactive step-by-step comparison of 3 MST algorithms side-by-side."""

    def __init__(
        self,
        coords: Dict[int, Tuple[float, float]],
        results: Dict[str, Tuple[List[MSTEdge], int, Dict, Dict, int, List[MSTEdge]]],
        dataset_name: str,
    ):
        """
        results = {
            'kruskal': (mst_edges, total_weight, op_metrics, order_added, n_it, edges_sequence),
            'prim': (...),
            'boruvka': (...)
        }
        """
        self.coords = coords
        self.results = results
        self.dataset_name = dataset_name
        self.node_ids = sorted(coords.keys())
        # Display order: Prim (left), Kruskal (center), Boruvka (right)
        self.algo_names = ['prim', 'kruskal', 'boruvka']

        # Extract data per algorithm (raw) and normalize into steps: List[List[edge]]
        raw_edges = {algo: results[algo][5] for algo in self.algo_names}
        self.order_added_per_algo = {algo: results[algo][3] for algo in self.algo_names}

        # Normalize so each step is a list of edges (edge = (u,v,w)).
        # Kruskal/Prim return: [ [(u,v,w)], [(u,v,w)], ... ] (each step has one edge)
        # Borůvka returns: [ [(u,v,w), (u,v,w), ...], [...], ... ] (each phase has multiple edges)
        # We keep them as-is since they're already in the right format.
        self.steps_per_algo: Dict[str, List[List[MSTEdge]]] = {}
        for algo, seq in raw_edges.items():
            # seq should already be List[List[MSTEdge]] from all three algorithms
            if seq and isinstance(seq[0], (list, tuple)):
                # Check if it's a list-of-lists or list-of-tuples
                if isinstance(seq[0], list):
                    # Already List[List[MSTEdge]] (Borůvka or others)
                    self.steps_per_algo[algo] = seq
                else:
                    # seq is List[MSTEdge] (shouldn't happen now, but fallback)
                    self.steps_per_algo[algo] = [[e] for e in seq]
            else:
                self.steps_per_algo[algo] = seq if seq else []

        # Global step unification (kept for info) and per-algo step counters
        self.max_steps = max((len(s) for s in self.steps_per_algo.values()), default=0)
        # Each algorithm advances independently by one of its own steps (a Boruvka phase
        # may contain multiple edges). `current_step_per_algo[algo]` is number of steps
        # applied for that algorithm.
        self.current_step_per_algo = {algo: 0 for algo in self.algo_names}

        # Create figure with 3 subplots
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.suptitle(f"{dataset_name} | MST Construction (Step-by-Step)", fontsize=14)

        # Track line artists and scatter per algorithm
        self.scatter_plots = {}
        self.line_artists = {algo: [] for algo in self.algo_names}  # For mplcursors
        self.line_meta = {algo: [] for algo in self.algo_names}     # For mplcursors metadata

        # Setup each subplot
        for idx, algo in enumerate(self.algo_names):
            ax = self.axes[idx]
            xs = [self.coords[i][0] for i in self.node_ids]
            ys = [self.coords[i][1] for i in self.node_ids]

            # Draw nodes
            self.scatter_plots[algo] = ax.scatter(xs, ys, s=50, c='blue', zorder=3)

            ax.set_title(self._get_title(algo))
            ax.axis('equal')
            ax.grid(True, alpha=0.3)

        # Info text
        self.info_text = self.fig.text(0.5, 0.01, "", ha='center', fontsize=10)

        # Keyboard navigation (left/right/space/r)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self.fig.tight_layout(rect=[0, 0.08, 1, 0.96])
        self._update_display()

    def _get_title(self, algo: str) -> str:
        """Get title with 'done' status if algo finished."""
        steps = self.steps_per_algo.get(algo, [])
        cur = self.current_step_per_algo.get(algo, 0)
        status = " \u2713 DONE" if cur >= len(steps) else ""
        return f"{algo.upper()}{status}"


    def _on_key(self, event):
        """Keyboard navigation handler."""
        if event.key in ("right", " "):
            # Advance each algorithm by one of its own steps (if available)
            changed = False
            for algo in self.algo_names:
                cur = self.current_step_per_algo.get(algo, 0)
                if cur < len(self.steps_per_algo.get(algo, [])):
                    self.current_step_per_algo[algo] = cur + 1
                    changed = True
            if changed:
                self._update_display()
        elif event.key == "left":
            changed = False
            for algo in self.algo_names:
                cur = self.current_step_per_algo.get(algo, 0)
                if cur > 0:
                    self.current_step_per_algo[algo] = cur - 1
                    changed = True
            if changed:
                self._update_display()
        elif event.key == "r":
            for algo in self.algo_names:
                self.current_step_per_algo[algo] = 0
            self._update_display()

    def _update_display(self):
        """Redraw all subplots for current step with mplcursors hover."""
        # Remove old lines and clear metadata
        for algo in self.algo_names:
            for line in self.line_artists[algo]:
                line.remove()
            self.line_artists[algo] = []
            self.line_meta[algo] = []

        # Draw edges up to each algorithm's own step count
        for algo_idx, algo in enumerate(self.algo_names):
            ax = self.axes[algo_idx]
            steps = self.steps_per_algo.get(algo, [])
            cur = self.current_step_per_algo.get(algo, 0)

            for step_idx in range(min(cur, len(steps))):
                step = steps[step_idx]
                # step can contain multiple edges (Boruvka) or a single edge
                for (u, v, w) in step:
                    x1, y1 = self.coords[u]
                    x2, y2 = self.coords[v]

                    # Color: red if current step (the most recent applied), darkred if past
                    color = 'red' if step_idx == cur - 1 else 'darkred'
                    linewidth = 2.5 if step_idx == cur - 1 else 1.5

                    (line,) = ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, zorder=1)
                    self.line_artists[algo].append(line)
                    self.line_meta[algo].append((u, v, w))

        # Update titles
        for idx, algo in enumerate(self.algo_names):
            self.axes[idx].set_title(self._get_title(algo))

        # Add mplcursors hover
        if _HAS_MPLCURSORS:
            # recreate single cursor to avoid stacking handlers
            if hasattr(self, 'cursor') and self.cursor is not None:
                try:
                    self.cursor.remove()
                except Exception:
                    try:
                        self.cursor.disconnect()
                    except Exception:
                        pass

            # build artists list (scatters + all current lines)
            artists_to_track = []
            for algo in self.algo_names:
                artists_to_track.append(self.scatter_plots[algo])
                artists_to_track.extend(self.line_artists[algo])

            self.cursor = mplcursors.cursor(artists_to_track, hover=True)

            @self.cursor.connect("add")
            def _on_add(sel):
                artist = sel.artist

                # Node hover: check if artist is one of the scatters
                for algo in self.algo_names:
                    sc = self.scatter_plots[algo]
                    if artist is sc:
                        idx = sel.index
                        node_id = self.node_ids[idx]
                        order_added = self.order_added_per_algo[algo]
                        ord_txt = ""
                        if node_id in order_added:
                            ord_txt = f"\norder_added: {order_added[node_id]}"
                        sel.annotation.set_text(f"node: {node_id}{ord_txt}")
                        return

                # Edge hover: find which algo's line list contains this artist
                for algo in self.algo_names:
                    if artist in self.line_artists[algo]:
                        i = self.line_artists[algo].index(artist)
                        u, v, w = self.line_meta[algo][i]
                        order_added = self.order_added_per_algo[algo]
                        step_txt = ""
                        # For Borůvka, edges are keyed as (u,v,w) tuples with step index
                        if algo == 'boruvka' and (u, v, w) in order_added:
                            step_txt = f"\nstep: {order_added[(u, v, w)]}"
                        sel.annotation.set_text(f"edge: ({u}, {v})\nweight: {w}{step_txt}")
                        return

                sel.annotation.set_text("edge")

        # Update per-algo step info
        parts = []
        for algo in self.algo_names:
            cur = self.current_step_per_algo.get(algo, 0)
            total = len(self.steps_per_algo.get(algo, []))
            parts.append(f"{algo}: {cur}/{total}")
        self.info_text.set_text(" | ".join(parts))

        self.fig.canvas.draw_idle()

    def show(self):
        """Display the interactive viewer."""
        plt.show()

def _has_cycle_in_undirected_graph(n_nodes: int, mst_edges: list[tuple[int, int, int]]) -> bool:
    """Return True if the edge set contains a cycle (Union-Find check)."""
    parent = {}
    rank = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False  # would form cycle
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    # init sets from nodes observed in mst_edges; fallback to 1..n if needed
    nodes = set()
    for u, v, _w in mst_edges:
        nodes.add(u); nodes.add(v)

    if not nodes:
        nodes = set(range(1, n_nodes + 1))

    for x in nodes:
        parent[x] = x
        rank[x] = 0

    for u, v, _w in mst_edges:
        if not union(u, v):
            return True
    return False


def show_validation_table(results: dict, dataset_name: str, n_nodes: int) -> None:
    """
    Show a table (matplotlib) with:
      rows: algorithms
      cols: Optimized, Cycle Presence
    Optimized: True if all algorithms have the same MST total weight.
    Cycle Presence: PASS/FAIL check (should be No).
    Also prints each weight in the Optimized cell.
    """
    algos = ["prim", "kruskal", "boruvka"]

    weights = {a: results[a][1] for a in algos}          # total_weight is index 1
    mst_edges_map = {a: results[a][0] for a in algos}    # mst_edges is index 0

    # optimized criterion: all weights equal
    all_equal = len(set(weights.values())) == 1

    # cycle check per algorithm
    has_cycle = {a: _has_cycle_in_undirected_graph(n_nodes, mst_edges_map[a]) for a in algos}

    # Build table content (English)
    col_labels = ["Optimized", "Cycle Presence"]
    row_labels = [a.capitalize() for a in algos]

    cell_text = []
    for a in algos:
        opt_text = f"{'YES' if all_equal else 'NO'}\n(weight={weights[a]})"
        cycle_text = "YES" if has_cycle[a] else "NO"
        cell_text.append([opt_text, cycle_text])

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axis("off")
    ax.set_title(f"{dataset_name} | Validation Summary", fontsize=13)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    plt.tight_layout()
    plt.show()


def show_comparison_barplots(results: Dict, exec_times: Dict[str, float], dataset_name: str, m_edges: int) -> None:
    """
    2x2 shared plots:
      (top-left) iterations/phases
      (top-right) total execution time (ms)
      (bottom-left) work per accepted edge
      (bottom-right) time per single accepted edge (ms / (n-1))
    """
    algo_names = ["prim", "kruskal", "boruvka"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # n_it values (your tuple format: index 4)
    n_its = {algo: results[algo][4] for algo in algo_names}

    # total time in ms
    exec_times_ms = {algo: exec_times[algo] * 1000 for algo in algo_names}

    # accepted edges (n-1) = len(mst_edges) (index 0)
    accepted = {algo: len(results[algo][0]) for algo in algo_names}

    # ---- Workload proxies (REAL counters you already have) ----
    # Prim workload: heap ops
    prim_op = results["prim"][2]
    prim_work = prim_op.get("heap_pushes", 0) + prim_op.get("heap_pops", 0)

    # Kruskal workload: scanned edges + UF ops
    kr_op = results["kruskal"][2]
    # your kruskal increments n_it once before breaking -> scanned ≈ n_it-1
    kr_edges_scanned = max(0, results["kruskal"][4] - 1)
    kr_work = kr_edges_scanned + kr_op.get("find_calls", 0) + kr_op.get("union_calls", 0)

    # Boruvka workload: estimated scans per phase + UF ops
    bo_op = results["boruvka"][2]
    phases = results["boruvka"][4]
    bo_edges_scanned_est = m_edges * phases
    bo_work = bo_edges_scanned_est + bo_op.get("find_calls", 0) + bo_op.get("union_calls", 0)

    work_total = {
        "prim": prim_work,
        "kruskal": kr_work,
        "boruvka": bo_work,
    }

    # Work per accepted edge
    work_per_edge = {a: (work_total[a] / accepted[a] if accepted[a] > 0 else 0.0) for a in algo_names}

    # Time per accepted edge (ms per MST edge)
    time_per_edge_ms = {a: (exec_times_ms[a] / accepted[a] if accepted[a] > 0 else 0.0) for a in algo_names}

    # ---- Plot 2x2 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{dataset_name} | Shared Comparison", fontsize=14)

    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    # (top-left) iterations/phases
    bars1 = ax1.bar(algo_names, [n_its[a] for a in algo_names], color=colors, alpha=0.7, edgecolor="black")
    ax1.set_ylabel("n_it (iterations / phases)")
    ax1.set_title("Iterations / Phases")
    ax1.grid(axis="y", alpha=0.3)
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h, f"{int(h)}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # (top-right) total time
    bars2 = ax2.bar(algo_names, [exec_times_ms[a] for a in algo_names], color=colors, alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Execution time (ms)")
    ax2.set_title("Total Time")
    ax2.grid(axis="y", alpha=0.3)
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h, f"{h:.2f}ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # (bottom-left) work per accepted edge
    bars3 = ax3.bar(algo_names, [work_per_edge[a] for a in algo_names], color=colors, alpha=0.7, edgecolor="black")
    ax3.set_ylabel("work / accepted edge")
    ax3.set_title("Work per Accepted Edge")
    ax3.grid(axis="y", alpha=0.3)
    for b in bars3:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2, h, f"{h:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # (bottom-right) time per accepted edge
    bars4 = ax4.bar(algo_names, [time_per_edge_ms[a] for a in algo_names], color=colors, alpha=0.7, edgecolor="black")
    ax4.set_ylabel("ms / accepted edge")
    ax4.set_title("Time per Single MST Edge")
    ax4.grid(axis="y", alpha=0.3)
    for b in bars4:
        h = b.get_height()
        ax4.text(b.get_x() + b.get_width()/2, h, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    plt.show()

def show_prim_exclusive_plots(results: Dict, dataset_name: str) -> None:
    """Prim-only: heap behavior (push/pop) + stale pops (lazy overhead)."""
    mst_edges, total_weight, op_metrics, order_added, n_it, steps = results["prim"]
    accepted_edges = len(mst_edges)
    heap_pushes = op_metrics.get("heap_pushes", 0)
    heap_pops = op_metrics.get("heap_pops", 0)
    stale_pops = max(0, heap_pops - accepted_edges)  # lazy pops that got discarded

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(f"{dataset_name} | Prim exclusive metrics", fontsize=13)

    labels = ["heap_pushes", "heap_pops", "stale_pops"]
    values = [heap_pushes, heap_pops, stale_pops]
    bars = ax.bar(labels, values, edgecolor="black", alpha=0.7)

    ax.set_ylabel("count")
    ax.set_title("Heap workload (lazy heap overhead)")
    ax.grid(axis="y", alpha=0.3)

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


def show_kruskal_exclusive_plots(results: Dict, dataset_name: str) -> None:
    """Kruskal-only: UF calls + accepted/rejected edges (cycle filtering)."""
    mst_edges, total_weight, op_metrics, order_added, n_it, steps = results["kruskal"]

    accepted_edges = len(mst_edges)  # should be n-1
    # In your kruskal.py: n_it increments before the 'done' break, so it can be +1.
    edges_scanned = max(0, n_it - 1)
    rejected_edges = max(0, edges_scanned - accepted_edges)

    find_calls = op_metrics.get("find_calls", 0)
    union_calls = op_metrics.get("union_calls", 0)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle(f"{dataset_name} | Kruskal exclusive metrics", fontsize=13)

    labels = ["edges_scanned", "accepted_edges", "rejected_edges", "find_calls", "union_calls"]
    values = [edges_scanned, accepted_edges, rejected_edges, find_calls, union_calls]
    bars = ax.bar(labels, values, edgecolor="black", alpha=0.7)

    ax.set_ylabel("count")
    ax.set_title("Cycle filtering + Union-Find workload")
    ax.grid(axis="y", alpha=0.3)

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom")

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()


def show_boruvka_exclusive_plots(results: Dict, dataset_name: str, m_edges: int) -> None:
    """Borůvka-only: phases + edges added per phase + estimated edge scans."""
    mst_edges, total_weight, op_metrics, order_added, n_it, edges_sequence = results["boruvka"]

    phases = n_it
    edges_added_per_phase = [len(phase) for phase in edges_sequence]
    # In your Boruvka implementation, each phase scans all edges once (for u,v,w in edges)
    edges_scanned_total_est = m_edges * phases

    find_calls = op_metrics.get("find_calls", 0)
    union_calls = op_metrics.get("union_calls", 0)

    # Figure 1: edges added per phase (line)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(f"{dataset_name} | Borůvka exclusive metrics", fontsize=13)

    ax.plot(range(1, len(edges_added_per_phase) + 1), edges_added_per_phase, marker="o")
    ax.set_xlabel("phase")
    ax.set_ylabel("edges added")
    ax.set_title("Edges added per phase (parallel additions)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Figure 2: summary bar
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = ["phases", "edges_scanned_est", "find_calls", "union_calls"]
    values = [phases, edges_scanned_total_est, find_calls, union_calls]
    bars = ax.bar(labels, values, edgecolor="black", alpha=0.7)

    ax.set_ylabel("count")
    ax.set_title("Phase count + estimated scan cost + UF workload")
    ax.grid(axis="y", alpha=0.3)

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="Interactive MST comparison benchmark.")
    parser.add_argument("--tsp", required=True, help="Path to TSPLIB .tsp file")
    args = parser.parse_args()

    # Parse and build graph
    print(f"Loading {args.tsp}...")
    inst = parse_tsplib_tsp(Path(args.tsp))
    nodes, edges = build_complete_graph_from_coords(inst.coords)
    adj = edges_to_adjacency(nodes, edges)

    start_node = 1 if 1 in set(nodes) else min(nodes)

    # Run all 3 algorithms
    print(f"Running 3 algorithms on {inst.name}...")
    results = {}
    exec_times = {}
    
    t0 = time.perf_counter()
    results['kruskal'] = kruskal_mst(nodes, edges, start_node=start_node, return_steps=True)
    exec_times['kruskal'] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    results['prim'] = prim_mst(adj, start=start_node, return_steps=True)
    exec_times['prim'] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    results['boruvka'] = boruvka_mst(nodes, edges, start_node=start_node, return_steps=True)
    exec_times['boruvka'] = time.perf_counter() - t0

    # Print summary
    for algo_name in ['kruskal', 'prim', 'boruvka']:
        result = results[algo_name]
        # result is expected to be (mst_edges, total_weight, op_metrics, order_added, n_it, steps)
        w = result[1]
        n_it = result[4]
        steps = result[5] if len(result) > 5 else []
        print(f"  {algo_name}: W={w} | n_it={n_it} | steps={len(steps)} | time={exec_times[algo_name]*1000:.3f}ms")

    print(f"\nLaunching interactive viewer...")

    # Launch interactive comparison
    visualizer = InteractiveMSTComparison(inst.coords, results, inst.name)
    visualizer.show()

    # After closing interactive graph viewer, show validation table
    show_validation_table(results, inst.name, n_nodes=inst.dimension)

    
    # After closing interactive viewer, show shared comparison barplots
    print("Showing shared comparison barplots...")
    show_comparison_barplots(results, exec_times, inst.name, m_edges=len(edges))


    # After closing shared plots, show exclusive plots per algorithm (one window sequence)
    print("Showing Prim-only plots...")
    show_prim_exclusive_plots(results, inst.name)

    print("Showing Kruskal-only plots...")
    show_kruskal_exclusive_plots(results, inst.name)

    print("Showing Borůvka-only plots...")
    show_boruvka_exclusive_plots(results, inst.name, m_edges=len(edges))


if __name__ == "__main__":
    main()
