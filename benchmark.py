from functions import parse_tsplib_tsp, build_complete_graph_from_coords

INSTANCE_INFO = {
    "berlin52": "52 locations in Berlin (Groetschel)",
    "st70": "70-city problem (Smith/Thompson)",
    "pr76": "76-city problem (Padberg/Rinaldi)",
}

for fname in ["berlin52.tsp", "st70.tsp", "pr76.tsp"]:
    inst = parse_tsplib_tsp(f"dataset/{fname}")
    nodes, edges = build_complete_graph_from_coords(inst.coords)
    print(f"{inst.name}: {INSTANCE_INFO.get(inst.name, '')} | n={len(nodes)} | m={len(edges)} | w={inst.edge_weight_type}")
