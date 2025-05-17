## DECO Overview
This repository provides several **task‑to‑engineer assignment baselines** plus our proposed **DECO** model.  
All methods simulate daily dispatch of CQC tickets, write their results to _Excel_ files, and are scored by a unified evaluator.

---



This file is required by `embedding.py` to construct the task graph for DeepWalk and knowledge graph embeddings.

---

## Execution Flow

### I. Embedding Generation

To generate node embeddings and knowledge graph triplets, run the following command from the project root:

```bash
python embedding.py
```



### II. Task Assignment Simulation

Once the embeddings have been generated, run the assignment simulation with:

```bash
python deco_main.py --structure both   --lambda_weight 0.4
```


---

### III. Evaluation

Evaluate every spreadsheet in a folder (default `results_xlsx/`):

```bash
python evaluation.py
```

The evaluator reports:
- Success Rate: % tasks completed on time
- Diversity: Std‑dev of fail‑mech distribution per engineer
- Workload Balance: Avg & std‑dev of workload ratio

Results are saved to `evaluatate_results/`.

---

## Models

- `deco.py`
Our full proposed model. Combines local (DeepWalk-based) and global (HolE-based) structural modeling on the defect-aware graph to encode test-fail relationships. Uses contrastive assignment refined by workload and expertise.

---

## Quick Start

```bash
pip3 install -r requirement.txt

# Optional embedding pre‑training
python embedding.py 

# Run the full DECO model
python deco.py  --structure both --lambda_weight 0.4
```