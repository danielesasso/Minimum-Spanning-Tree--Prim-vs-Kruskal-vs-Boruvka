from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

from utils.functions import parse_tsplib_tsp, build_complete_graph_from_coords
from utils.plotting import plot_points_interactive, plot_mst_interactive

MSTEdge = Tuple[int, int, int]


class UnionFind:
	def __init__(self, nodes: List[int]) -> None:
		self.parent = {x: x for x in nodes}
		self.rank = {x: 0 for x in nodes}
		self.find_calls = 0
		self.union_calls = 0

	def find(self, x: int) -> int:
		self.find_calls += 1
		if self.parent[x] != x:
			self.parent[x] = self.find(self.parent[x])
		return self.parent[x]

	def union(self, a: int, b: int) -> bool:
		self.union_calls += 1
		ra, rb = self.find(a), self.find(b)
		if ra == rb:
			return False
		if self.rank[ra] < self.rank[rb]:
			ra, rb = rb, ra
		self.parent[rb] = ra
		if self.rank[ra] == self.rank[rb]:
			self.rank[ra] += 1
		return True


def boruvka_mst(
	nodes: List[int],
	edges: List[MSTEdge],
	start_node: int = 1,
	return_steps: bool = False,
) -> Tuple[List[MSTEdge], int, Dict[str, int], Dict[int, int], int] | Tuple[List[MSTEdge], int, Dict[str, int], Dict[int, int], int, List[MSTEdge]]:
	"""
	Boruvka's algorithm for MST.

	Returns: (mst_edges, total_weight, op_metrics, order_added, n_it, edges_sequence)
	"""
	if not nodes:
		return [], 0, {"find_calls": 0, "union_calls": 0}, {}, 0, []

	uf = UnionFind(nodes)
	node_set = set(nodes)
	order_added: Dict[int, int] = {}
	if start_node in node_set:
		order_added[start_node] = 0

	mst_edges: List[MSTEdge] = []
	edges_sequence: List[List[MSTEdge]] = []  # Now list of phases, each phase is list of edges
	total_weight = 0
	n_it = 0

	# initial number of components
	comp_count = len(nodes)

	# iterate phases until single component
	while comp_count > 1:
		n_it += 1

		# best edge per component root
		best: Dict[int, MSTEdge] = {}

		for u, v, w in edges:
			ru = uf.find(u)
			rv = uf.find(v)
			if ru == rv:
				continue

			# update best for ru
			if ru not in best or w < best[ru][2]:
				best[ru] = (u, v, w)
			# update best for rv
			if rv not in best or w < best[rv][2]:
				best[rv] = (u, v, w)

		# apply selected edges (one per component) - may be duplicates
		phase_edges: List[MSTEdge] = []
		any_merged = False
		for e in set(best.values()):
			u, v, w = e
			if uf.find(u) != uf.find(v):
				merged = uf.union(u, v)
				if merged:
					any_merged = True
					mst_edges.append((u, v, w))
					phase_edges.append((u, v, w))
					total_weight += w
					# record step index for this edge and nodes
					order_added[(u, v, w)] = n_it
					for node in (u, v):
						if node not in order_added:
							order_added[node] = n_it
					comp_count -= 1

		# Append this phase's edges to sequence (all edges in one phase together)
		if phase_edges:
			edges_sequence.append(phase_edges)

		# safety: if no merges occurred (shouldn't happen on connected graph), break
		if not any_merged:
			break

	op_metrics = {"find_calls": uf.find_calls, "union_calls": uf.union_calls}
	if return_steps:
		return mst_edges, total_weight, op_metrics, order_added, n_it, edges_sequence
	return mst_edges, total_weight, op_metrics, order_added, n_it

def main() -> None:
	parser = argparse.ArgumentParser(description="Run Boruvka on a TSPLIB .tsp instance.")
	parser.add_argument("--tsp", required=True, help="Path to TSPLIB .tsp file")
	parser.add_argument("--plot", action="store_true", help="Show plots (default behavior)")
	args = parser.parse_args()

	inst = parse_tsplib_tsp(Path(args.tsp))
	nodes, edges = build_complete_graph_from_coords(inst.coords)

	n = len(nodes)
	m = len(edges)
	start_node = 1 if 1 in set(nodes) else min(nodes)

	t0 = time.perf_counter()
	mst_edges, total_weight, op_metrics, order_added, n_it, edges_sequence = boruvka_mst(nodes, edges, start_node=start_node, return_steps=True)
	t1 = time.perf_counter()
	elapsed_ms = (t1 - t0) * 1000.0

	print(f"boruvka | {inst.name} | n={n} m={m} | W={total_weight} | n_it={n_it} | "
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
		title=f"{inst.name} | Boruvka MST | W={total_weight} | time={elapsed_ms:.2f}ms",
		start_node=start_node,
		order_added=order_added,
		show=True,
	)


if __name__ == "__main__":
	main()

