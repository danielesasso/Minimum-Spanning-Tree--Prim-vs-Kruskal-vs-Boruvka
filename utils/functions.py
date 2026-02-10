
# for parsing
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

# graph building
import math
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class TSPLIBInstance:
    """Parsed TSPLIB TSP instance (focused on NODE_COORD_SECTION)."""
    name: str
    dimension: int
    edge_weight_type: str
    coords: Dict[int, Tuple[float, float]]  # node_id -> (x, y)


def _clean_line(line: str) -> str:
    """Strip and drop inline carriage returns; keep content for parsing."""
    return line.strip().replace("\r", "")


def parse_tsplib_tsp(path: Union[str, Path]) -> TSPLIBInstance:
    """
    Parse a TSPLIB .tsp file that contains NODE_COORD_SECTION.

    Supports typical TSPLIB headers:
      NAME, TYPE, COMMENT, DIMENSION, EDGE_WEIGHT_TYPE, NODE_COORD_SECTION, EOF

    Returns:
      TSPLIBInstance(name, dimension, edge_weight_type, coords)

    Raises:
      ValueError: if required fields/sections are missing or malformed.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    name: Optional[str] = None
    dimension: Optional[int] = None
    edge_weight_type: Optional[str] = None

    coords: Dict[int, Tuple[float, float]] = {}
    in_node_coord_section = False

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = _clean_line(raw)
            if not line:
                continue

            # Section markers
            upper = line.upper()
            if upper == "NODE_COORD_SECTION":
                in_node_coord_section = True
                continue
            if upper == "EOF":
                break

            if in_node_coord_section:
                # Expected: "<id> <x> <y>"
                parts = line.split()
                if len(parts) < 3:
                    # treat as error to be safe
                    raise ValueError(f"Malformed NODE_COORD_SECTION line: {line}")

                try:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                except Exception as e:
                    raise ValueError(f"Invalid coordinate line: {line}") from e

                coords[node_id] = (x, y)
                continue

            # Header parsing: allow "KEY: value" or "KEY value"
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().upper()
                val = val.strip()
            else:
                # Split once: KEY value
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                key = parts[0].strip().upper()
                val = parts[1].strip()

            if key == "NAME":
                name = val
            elif key == "DIMENSION":
                try:
                    dimension = int(val)
                except Exception as e:
                    raise ValueError(f"Invalid DIMENSION value: {val}") from e
            elif key == "EDGE_WEIGHT_TYPE":
                edge_weight_type = val.upper()
            # We ignore TYPE/COMMENT/etc. here on purpose.

    # Basic validation
    if name is None:
        raise ValueError("Missing NAME in TSPLIB file.")
    if dimension is None:
        raise ValueError("Missing DIMENSION in TSPLIB file.")
    if edge_weight_type is None:
        # Many TSPLIB have it; for safety default to EUC_2D only if coords exist.
        edge_weight_type = "EUC_2D"

    if not coords:
        raise ValueError("Missing or empty NODE_COORD_SECTION.")
    if len(coords) != dimension:
        # TSPLIB node ids are often 1..dimension; but some files might have gaps.
        # We enforce exact count to avoid silent bugs.
        raise ValueError(
            f"NODE_COORD_SECTION has {len(coords)} nodes, but DIMENSION is {dimension}."
        )

    return TSPLIBInstance(
        name=name,
        dimension=dimension,
        edge_weight_type=edge_weight_type,
        coords=coords,
    )


def instance_to_lists(inst: TSPLIBInstance) -> Tuple[List[int], List[Tuple[float, float]]]:
    """
    Convert instance coords to ordered lists by node id (ascending).
    Returns:
      node_ids: [1..n]
      points:   [(x1,y1), (x2,y2), ...]
    """
    node_ids = sorted(inst.coords.keys())
    points = [inst.coords[i] for i in node_ids]
    return node_ids, points


def tsplib_euc_2d_weight(a: Tuple[float, float], b: Tuple[float, float]) -> int:
    """
    TSPLIB EUC_2D distance:
    d = round(sqrt((dx)^2 + (dy)^2)) with nearest integer (TSPLIB-style).
    Implemented as int(dist + 0.5).
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dist = math.sqrt(dx * dx + dy * dy)
    return int(dist + 0.5)


def build_complete_graph_from_coords(
    coords: Dict[int, Tuple[float, float]],
    weight_fn=tsplib_euc_2d_weight,
) -> Tuple[List[int], List[Tuple[int, int, int]]]:
    """
    Build a complete undirected weighted graph from node coordinates.

    Args:
      coords: node_id -> (x, y)
      weight_fn: function((x1,y1),(x2,y2)) -> weight (int)

    Returns:
      nodes: sorted node ids
      edges: list of (u, v, w) with u < v
    """
    nodes = sorted(coords.keys())
    edges: List[Tuple[int, int, int]] = []

    n = len(nodes)
    for i in range(n):
        u = nodes[i]
        for j in range(i + 1, n):
            v = nodes[j]
            w = weight_fn(coords[u], coords[v])
            edges.append((u, v, w))

    return nodes, edges


def edges_to_adjacency(
    nodes: List[int],
    edges: List[Tuple[int, int, int]],
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Convert undirected edge list to adjacency list.

    Returns:
      adj[u] = [(v, w), ...]
    """
    adj: Dict[int, List[Tuple[int, int]]] = {u: [] for u in nodes}
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj
