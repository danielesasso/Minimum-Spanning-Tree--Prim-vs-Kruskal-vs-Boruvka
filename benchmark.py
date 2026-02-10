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


def show_comparison_barplots(results: Dict, exec_times: Dict[str, float], dataset_name: str) -> None:
    """Show two barplot figures comparing n_it and execution time."""
    algo_names = ['prim', 'kruskal', 'boruvka']
    
    # Extract n_it values
    n_its = {algo: results[algo][4] for algo in algo_names}
    
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{dataset_name} | Algorithm Comparison", fontsize=14)
    
    # Barplot 1: n_it
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars1 = ax1.bar(algo_names, [n_its[a] for a in algo_names], color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Iterations (n_it)', fontsize=11)
    ax1.set_title('Iterations per Algorithm', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Barplot 2: execution time (in milliseconds)
    exec_times_ms = {algo: exec_times[algo] * 1000 for algo in algo_names}
    bars2 = ax2.bar(algo_names, [exec_times_ms[a] for a in algo_names], color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Execution Time (ms)', fontsize=11)
    ax2.set_title('Execution Time per Algorithm', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    fig.tight_layout()
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
    
    # After closing interactive viewer, show comparison barplots
    print("Showing comparison barplots...")
    show_comparison_barplots(results, exec_times, inst.name)


if __name__ == "__main__":
    main()
