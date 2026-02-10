# Minimum Spanning Tree – Prim vs Kruskal vs Borůvka

This project compares different **Minimum Spanning Tree (MST)** algorithms on **TSPLIB** instances, with a focus on **algorithmic behavior, metrics, and visualization**.

Currently implemented:
- **Prim’s algorithm** (lazy heap / priority queue)

---

## Requirements

Tested with **Python 3.10+**.

Install dependencies (Conda recommended):

```bash
conda create -n mst python=3.10
conda activate mst
conda install numpy matplotlib pandas networkx tqdm
conda install -c conda-forge mplcursors
```
---

Dataset

The project uses TSPLIB instances (EUC_2D, complete graphs):
```bash
dataset/
 ├─ berlin52.tsp
 ├─ st70.tsp
 └─ pr76.tsp
```

---
## Run Algorithm

Run with interactive plots (nodes + MST):
```bash
python -m method --tsp dataset/st70.tsp --plot
```
Save plots without showing them:
```bash
python -m method --tsp dataset/st70.tsp --save-plots --no-show
```

---
## Run Benchmark

Run per dataset:
```bash
python benchmark.py --tsp dataset/st70.tsp
```
---
## Output

Prim reports:
- Execution time (ms)
- MST total weight
- Number of MST edges
- Heap operations (push/pop)
- Number of edges examined
- Theoretical complexity

Interactive plots:
- **Initial graph**: nodes only, start node highlighted
- **Final graph**: MST in red  
  - Hover on nodes: node id, order added, path weight from start  
  - Hover on edges: edge weight

---

## Notes

- Graphs are treated as **complete graphs** derived from TSPLIB coordinates.
- Visualization uses `matplotlib` + `mplcursors` for interactive hover.
- Kruskal and Borůvka implementations will be added next.

