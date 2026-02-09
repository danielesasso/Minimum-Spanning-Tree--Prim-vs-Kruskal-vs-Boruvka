# plotting.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
from collections import deque

import matplotlib.pyplot as plt

try:
    import mplcursors  # type: ignore
    _HAS_MPLCURSORS = True
except Exception:
    _HAS_MPLCURSORS = False


def _ensure_dir(save_path: Optional[str | Path]) -> Optional[Path]:
    if save_path is None:
        return None
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _mst_distances_from_start(
    mst_edges: Iterable[Tuple[int, int, int]],
    start_node: int,
) -> Dict[int, int]:
    """
    Given MST edges (tree), compute distance from start to every node along the tree.
    Distance = sum of weights on the unique path in the MST.
    """
    adj: Dict[int, List[Tuple[int, int]]] = {}
    for u, v, w in mst_edges:
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))

    dist: Dict[int, int] = {start_node: 0}
    q = deque([start_node])

    while q:
        u = q.popleft()
        for v, w in adj.get(u, []):
            if v in dist:
                continue
            dist[v] = dist[u] + w
            q.append(v)

    return dist


def plot_points_interactive(
    coords: Dict[int, Tuple[float, float]],
    title: str,
    start_node: Optional[int] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    node_ids = sorted(coords.keys())
    xs = [coords[i][0] for i in node_ids]
    ys = [coords[i][1] for i in node_ids]

    fig, ax = plt.subplots()
    sc = ax.scatter(xs, ys)

    if start_node is not None and start_node in coords:
        sx, sy = coords[start_node]
        ax.scatter([sx], [sy], marker="*", s=180)

    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()

    if _HAS_MPLCURSORS:
        cursor = mplcursors.cursor(sc, hover=True)

        @cursor.connect("add")
        def _on_add(sel):
            idx = sel.index
            node_id = node_ids[idx]
            start_txt = " (start)" if node_id == start_node else ""
            sel.annotation.set_text(f"node: {node_id}{start_txt}")

    p = _ensure_dir(save_path)
    if p is not None:
        fig.savefig(p, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_mst_interactive(
    coords: Dict[int, Tuple[float, float]],
    mst_edges: Iterable[Tuple[int, int, int]],
    title: str,
    start_node: Optional[int] = None,
    order_added: Optional[Dict[int, int]] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    node_ids = sorted(coords.keys())
    xs = [coords[i][0] for i in node_ids]
    ys = [coords[i][1] for i in node_ids]

    fig, ax = plt.subplots()
    sc = ax.scatter(xs, ys)

    # Draw MST edges and store meta
    line_artists = []
    line_meta: List[Tuple[int, int, int]] = []
    mst_edges_list = list(mst_edges)  # consume iterable once
    for u, v, w in mst_edges_list:
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        (ln,) = ax.plot([x1, x2], [y1, y2], color="red")
        line_artists.append(ln)
        line_meta.append((u, v, w))

    # Highlight start node
    dist_from_start: Optional[Dict[int, int]] = None
    if start_node is not None and start_node in coords:
        sx, sy = coords[start_node]
        ax.scatter([sx], [sy], marker="*", s=180)
        dist_from_start = _mst_distances_from_start(mst_edges_list, start_node)

    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()

    if _HAS_MPLCURSORS:
        cursor = mplcursors.cursor([sc, *line_artists], hover=True)

        @cursor.connect("add")
        def _on_add(sel):
            artist = sel.artist

            # Node hover
            if artist is sc:
                idx = sel.index
                node_id = node_ids[idx]
                start_txt = " (start)" if node_id == start_node else ""

                ord_txt = ""
                if order_added is not None and node_id in order_added:
                    ord_txt = f"\norder_added: {order_added[node_id]}"

                dist_txt = ""
                if dist_from_start is not None and node_id in dist_from_start:
                    dist_txt = f"\npath_weight_from_start: {dist_from_start[node_id]}"

                sel.annotation.set_text(f"node: {node_id}{start_txt}{ord_txt}{dist_txt}")
                return

            # Edge hover
            try:
                i = line_artists.index(artist)
                u, v, w = line_meta[i]
                sel.annotation.set_text(f"edge: ({u}, {v})\nweight: {w}")
            except Exception:
                sel.annotation.set_text("edge")

    p = _ensure_dir(save_path)
    if p is not None:
        fig.savefig(p, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)
