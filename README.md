# Learning to Assign Prediction Tasks to Agents with Capacity Constraints

Code, data and saved results for the experiments in Section 4 and Appendix H.

- **Camelyon17, MMLU, ImageNet16H, Civil**: this directory.
- **Bank, Credit, Coupon, Cardio** and the synthetic capacity validation: [`tabular/`](tabular/README.md), a self-contained package with its own environment.

## Setup

```bash
pip install -r requirements.txt               # Python 3.10
find data -name '*.gz' -exec gunzip -k {} +   # large data files are stored gzipped
```

For `tabular/`, use Python 3.11 and run `pip install -r tabular/requirements.txt`.

## Reproducing figures and tables

The commands below regenerate each item from the saved results: logs in `logs/` (the paper runs) and `tabular/<dataset>/results/`. Figures are written to `plots/`. `tabular` § *n* refers to section *n* of [`tabular/reproduce_figures_tables.ipynb`](tabular/reproduce_figures_tables.ipynb), which writes to `tabular/figures/` and `tabular/tables/` and runs in about 10 s.

| Paper item | Command |
|---|---|
| Fig. 1: MMLU accuracy vs. context | `python -m experiments.mmlu_pca_plot` |
| Fig. 2a: Camelyon17 accuracy vs. context | `python -m experiments.camelyon17_pca_plot` |
| Fig. 2b: Camelyon17 | `python -m experiments.camelyon17_results_plot --log-file logs/camelyon17_20260502_131702.jsonl` |
| Fig. 3a: Bank | `tabular` § 2 |
| Fig. 3b: MMLU | `python -m experiments.mmlu_results_plot --log-file logs/mmlu_20260502_025211.jsonl` |
| Fig. 3c: ImageNet16H | `python -m experiments.imagenet16h_results_plot --log-file logs/imagenet16h_ep10_vgg19_p48_20260502_023209.jsonl` |
| Fig. 4, Fig. 8: mini-batch (Bank) | `tabular` § 5 |
| Fig. 5a–c: Cardio, Coupon, Credit | `tabular` § 2 |
| Fig. 5d: Civil | `python -m experiments.civilcomments_results_plot --log-file logs/civilcomments_race_20260916_192050.jsonl` |
| Fig. 7: unconstrained free agent (Bank) | `tabular` § 6 |
| Fig. 9: effect of η (Camelyon17) | `python -m experiments.camelyon17_eta --plot --log-file logs/eta_sensitivity_camelyon17_20260504_142927.jsonl` |
| Fig. 10: synthetic capacity validation | `tabular` § 7 |
| Table 1: datasets and agent accuracies | `tabular` § 1; other datasets: `python summarize_log.py <log>` |
| Table 2: uncertainty across task streams | `tabular` § 3; other datasets: `python summarize_log.py <log>` |
| Table 3: tabular datasets across seeds | `tabular` § 4 |
| Table 4: Greedy vs. TS at η = 0 (Camelyon17) | `python -m experiments.camelyon17_baselines --plot logs/camelyon17_baselines_eta0_20261108_161454.jsonl` |
| Table 5: capacity enforcement (Bank) | `tabular` § 8 |
| Table 6: capacity enforcement (all datasets) | `tabular` § 9; other datasets: `trajectories/step_actions/` |
| Table 7: feature-split robustness (Bank) | `tabular` § 10 |
| Tables 8, 9: plug-in and DMD baselines (Bank) | `tabular` § 11 |
| Tables 10, 11: plug-in and DMD baselines (Camelyon17) | `python -m experiments.camelyon17_baselines --plot logs/camelyon17_baselines_baselines_20261109_142730.jsonl`; max final deviation: `results/camelyon17/` |
| Table 12: offline router (Bank) | `tabular` § 12 |
| Table 13: offline router (Camelyon17) | `python -m experiments.camelyon17_baselines --plot logs/camelyon17_baselines_split_20261108_161514.jsonl` |

## Rerunning the experiments

Defaults match the paper: 100 task streams, α₁ ∈ {0.2, 0.4, 0.5, 0.6, 0.8}, η = 0.5, seed 42. Each run writes a new `logs/<name>_<timestamp>.jsonl`. Pass that file to the plot command above with `--log-file`.

```bash
python -m experiments.camelyon17                                   # Fig. 2b
python -m experiments.camelyon17_eta                               # Fig. 9
python -m experiments.mmlu                                         # Fig. 3b
python -m experiments.imagenet16h --model vgg19 --participant 48   # Fig. 3c
OMP_NUM_THREADS=1 python -m experiments.civilcomments --dataset race --pooled --workers 40 \
    --policies "Logistic TS,Logistic Greedy,Tree TS,Tree Greedy,Random"     # Fig. 5d (~20 min with 40 workers)

# Tables 4, 10, 11, 13 (10 task streams; --save-actions writes the per-step assignments)
OMP_NUM_THREADS=1 python -m experiments.camelyon17_baselines --setup eta0 --workers 10        # Table 4
OMP_NUM_THREADS=1 python -m experiments.camelyon17_baselines --setup baselines --workers 10 \
    --save-actions results/camelyon17                                                         # Tables 10, 11
OMP_NUM_THREADS=1 python -m experiments.camelyon17_baselines --setup split --workers 10       # Table 13

# Table 6: per-step assignments at α₁ = 0.8, for d in camelyon17, mmlu, imagenet16h, civilcomments
python -m experiments.step_trajectories --dataset <d> --capacities 0.2 [--pooled]   # --pooled for civilcomments
```

The commands for the tabular experiments are listed in [`tabular/README.md`](tabular/README.md#scripts).

The Camelyon17, MMLU and η logs were produced before the tree policies' random state was seeded explicitly. Rerunning reproduces their tree-policy results statistically but not bit for bit. Logistic TS can also differ slightly across machines, because its posterior draws depend on the linear-algebra backend.

## Data

Processed copies of every dataset are included: `data/` for this directory and `tabular/<dataset>/data/` for the tabular datasets. For Camelyon17 only the 21 extracted patch features (`features_cache.npz`) and `metadata.csv` are included. The raw patches come from [WILDS](https://wilds.stanford.edu/datasets/) and are needed only to re-extract the features.
