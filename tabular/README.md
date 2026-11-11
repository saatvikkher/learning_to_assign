# Code and results for the tabular experiments

This folder contains the code, data and saved results behind every figure and table in the paper that uses the four tabular datasets (**Bank**, **Credit**, **Coupon**, **Cardio**), plus the synthetic capacity-validation simulation (Figure 10). Experiments on the other datasets (Camelyon17, MMLU, ImageNet16H, Civil) are not included here.

## Quick start

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate capacity_bandits
jupyter notebook reproduce_figures_tables.ipynb   # then "Run All" (about 10 seconds)
```

`reproduce_figures_tables.ipynb` is the single entry point. It reads only the saved results in `<dataset>/results/` (no model training), reruns the small synthetic simulation for Figure 10, and writes all figures to `figures/` and all tables to `tables/`. Both folders are already filled with its output.

## Folder layout

```
.
├── README.md                         this file
├── reproduce_figures_tables.ipynb    regenerates every tabular figure and table in the paper
├── environment.yml / requirements.txt
├── figures/                          notebook output (PDF/PNG), named fig<number>_<paper file name>
├── tables/                           notebook output (CSV), named table<number>_...
├── bank/
│   ├── data/bank_marketing_raw.csv
│   ├── code/                         experiment scripts (see table below)
│   └── results/                      saved outputs of those scripts
├── credit/
│   ├── data/default of credit card clients.xls
│   ├── code/                         main_100runs_error_rate.py, main_100runs_step_level.py, multi_seeds_10runs.py
│   └── results/                      main_100runs/, step_level_100runs/, multi_seeds_10runs/
├── coupon/                           same structure as credit/ (data: in-vehicle-coupon-recommendation.csv)
└── cardio/                           same structure as credit/ (data: cardio_train.csv)
```

## Paper item → notebook section → code → saved results

| Paper item | Notebook § | Script(s) | Saved results |
|---|---|---|---|
| Table 1 (tabular rows): features, T, agent accuracies | 1 | `<ds>/code/main_100runs_error_rate.py` | `<ds>/results/main_100runs/*_agents.json` |
| Fig. 3(a) Bank; Fig. 5(a) Cardio, 5(b) Coupon, 5(c) Credit | 2 | `<ds>/code/main_100runs_step_level.py` | `<ds>/results/step_level_100runs/*_runs100.json` |
| Table 2 (tabular rows): uncertainty over 100 streams | 3 | `<ds>/code/main_100runs_error_rate.py` | `<ds>/results/main_100runs/*_online_summary.csv` (per-run: `*_online_raw_runs.csv`) |
| Table 3: 5 seeds, 10 runs, α₁ = α₂ = 0.5 | 4 | `<ds>/code/multi_seeds_10runs.py` | `<ds>/results/multi_seeds_10runs/` |
| Fig. 4 (tree greedy), Fig. 8 (logistic greedy): mini-batch | 5 | `bank/code/minibatch_B100_capacity_sweep.py`, `bank/code/minibatch_vary_batch_size.py`, `bank/code/main_100runs_step_level.py` (online curve) | `bank/results/minibatch_B100_50runs/`, `bank/results/minibatch_vary_batch_size_10runs/`, `bank/results/minibatch_online_50runs/` |
| Fig. 7: extra unconstrained (free) agent | 6 | `bank/code/free_agent_50runs.py` | `bank/results/free_agent_50runs/` |
| Fig. 10 and text of App. H.10 (synthetic) | 7 | inside the notebook | none (rerun, seeds 0–49) |
| Table 5 and text of App. H.10 (Bank capacity convergence) | 8 | `bank/code/main_100runs_step_level.py` | `bank/results/step_level_100runs/*_step_actions_packed.npz` |
| Table 6 (tabular rows): final capacity deviation | 9 | `<ds>/code/main_100runs_step_level.py` | `<ds>/results/step_level_100runs/*_step_actions_packed.npz` |
| Table 7 and text of App. H.11: feature-split robustness | 10 | `bank/code/feature_split_sweep.py` | `bank/results/feature_split_sweep/` |
| Tables 8, 9: plug-in and DMD baselines (Bank) | 11 | `bank/code/main_100runs_*.py` (η = 0.5, first 10 streams), `bank/code/eta_sqrtAT_10runs.py`, `bank/code/plugin_10runs.py`, `bank/code/dmd_10runs.py` | `bank/results/{main_100runs, step_level_100runs, eta_sqrtAT_10runs, plugin_10runs, dmd_10runs}/` |
| Table 12: offline router, half split (Bank) | 12 | `bank/code/half_split_offline_router_10runs.py` | `bank/results/half_split_offline_router_10runs/` |

Numbering follows the current paper draft. All values produced by the notebook match the paper.

## Experimental setup (shared by all scripts)

- Two agents per dataset, built from a random feature split (the seed fixes which features each agent sees). The paper's **Agent 1** is an XGBoost classifier on the remaining features (200 trees, depth 4, learning rate 0.05, subsample and column subsample 0.8); **Agent 2** is a logistic regression with balanced class weights on a random subset of up to 20% of the features. This is the only agent configuration in the code. 70/30 train/test split; the T test points form the online task stream.
- Policies: Logistic TS, Logistic greedy, Tree TS (bootstrap), Tree greedy, each with the virtual-queue capacity penalty, η = 0.5 unless stated. Capacities of the logistic-regression agent: 0.2, 0.4, 0.5, 0.6, 0.8.
- Run r uses task-order permutation seed `seed + r`, shared by every policy and capacity, so comparisons across policies are paired. The 10-run baselines (Tables 8, 9) use the first 10 of these orders.
- Error rate of one run = 1 − mean step reward over the T tasks; reported value = mean over runs; SE = sample SD / √(number of runs).

**Agent numbering in code vs. paper.** The code numbers the agents the other way round: action `0`, `capacity_agent0`, "agent 1" in the console output and `agent1_logistic_regression` in `*_agents.json` refer to the logistic regression agent (paper Agent 2); action `1`, "agent 2" and `agent2_xgboost` refer to XGBoost (paper Agent 1). So paper α₂ = `capacity_agent0` and α₁ = 1 − `capacity_agent0`. In the feature-split results, `requested_fraction_agent1` is the share of features given to the logistic regression agent. The notebook converts everything to the paper's convention.

## Scripts

All scripts are standalone (no imports between them) and read the dataset from `../data/`, so run them from their own `code/` folder. Every script uses the same two agents (logistic regression and XGBoost; see above). The first argument is the dataset seed; the commands below use the seeds from the paper, which are also the defaults if the seed is omitted. Each script writes to a new subfolder of `code/`; the saved copies in `results/` were renamed as listed.

| Script | Command (paper setting) | Writes to | Saved copy |
|---|---|---|---|
| `<ds>/code/main_100runs_error_rate.py` | `python main_100runs_error_rate.py <seed>` with seed 2 (Bank), 2 (Credit), 9 (Coupon), 11 (Cardio) | `two_agents_100_run/` (Cardio: `two_agents_run_100/`) | `results/main_100runs/` |
| `<ds>/code/main_100runs_step_level.py` | `python main_100runs_step_level.py <seed>` with seed 2 (Bank), 2 (Credit), 9 (Coupon), 11 (Cardio) | `two_agents_100_run/` (Credit, Cardio: `two_agents_run_100/`) | `results/step_level_100runs/` |
| `<ds>/code/multi_seeds_10runs.py` | `python multi_seeds_10runs.py <s>` for s = 0..4 | `multi_seeds/` | `results/multi_seeds_10runs/` |
| `bank/code/eta_sqrtAT_10runs.py` | `python eta_sqrtAT_10runs.py 2` | `two_agents_10_run_eta_theory/` | `results/eta_sqrtAT_10runs/` |
| `bank/code/plugin_10runs.py` | `python plugin_10runs.py 2` | `plugin_10_run/` | `results/plugin_10runs/` |
| `bank/code/dmd_10runs.py` | `python dmd_10runs.py 2 10` (seed, runs) | `dmd_10_run/` | `results/dmd_10runs/` |
| `bank/code/half_split_offline_router_10runs.py` | `python half_split_offline_router_10runs.py 2` | `results_half_split_10_run/` | `results/half_split_offline_router_10runs/` |
| `bank/code/minibatch_B100_capacity_sweep.py` | `python minibatch_B100_capacity_sweep.py 2` | `batch_queue_results/` | `results/minibatch_B100_50runs/` |
| `bank/code/minibatch_vary_batch_size.py` | `python minibatch_vary_batch_size.py 2 <B>` (seed, batch size) for B in 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 3000, 5000, 12357 | `more_scale_batch_queue_results/` | `results/minibatch_vary_batch_size_10runs/` |
| `bank/code/free_agent_50runs.py` | `python free_agent_50runs.py 2` | `new_final_one_stage_free_agent/` | `results/free_agent_50runs/` |
| `bank/code/feature_split_sweep.py` | `python feature_split_sweep.py <f> <s> 10` (feature fraction, split seed, runs) for f in 0.1, 0.2, 0.5, 0.8, 0.9 and s = 0..4 | `feature_split_results/` | `results/feature_split_sweep/` |

What each script does:
- `main_100runs_error_rate.py`: the main capacity sweep (4 contextual policies × 5 capacities × 100 task orders) plus the offline benchmarks; saves per-run error rates (`*_online_raw_runs.csv`), their mean/SD/SE (`*_online_summary.csv`) and mean errors per policy (`*_runs100.json`).
- `main_100runs_step_level.py`: the same experiment, saving step-level rewards and actions and the mean errors used for the figures (`*_runs100.json`, including the offline unconstrained benchmarks `Offline * Un`). The online curve in Figs. 4 and 8 (`minibatch_online_50runs/`) is the 50-run checkpoint (`*_runs50.json`) of an earlier execution of this script.
- `multi_seeds_10runs.py`: the main experiment at α₁ = α₂ = 0.5 for a given dataset seed, 10 task orders.
- `eta_sqrtAT_10runs.py`: our policy with η = √(A/T) (≈ 0.0127 for Bank), 10 runs, with realized-capacity tracking.
- `plugin_10runs.py`: online plug-in policy that thresholds the estimated accuracy gap at the empirical capacity quantile.
- `dmd_10runs.py`: dual mirror descent (Euclidean and entropic reference functions) with hard per-agent budgets, using the same online reward models.
- `half_split_offline_router_10runs.py`: splits the held-out set in half; fits offline logistic / random-forest routers on one half with both agents' labels, then runs the frozen routers and our online policies (η = 0.5) on the other half with the same task orders.
- `minibatch_B100_capacity_sweep.py` / `minibatch_vary_batch_size.py`: mini-batch assignment with min-cost max-flow inside each batch and queue updates between batches (requires `ortools`).
- `free_agent_50runs.py`: two capacity-constrained agents plus one unconstrained agent, 50 runs.
- `feature_split_sweep.py`: one feature-split configuration (fraction of features given to the logistic regression agent, split seed) at α₁ = α₂ = 0.5, 10 task orders (stream seed 100000 + r); the random baseline is the analytical 50/50 error.

## Result file formats

- `*_online_raw_runs.csv`, `*_raw_runs.csv`: one row per (run, capacity, policy) with `error_rate`; baseline files also record realized-capacity statistics.
- `*_online_summary.csv`, `*_summary.csv`: mean, SD, SE and 95% CI of the error rate per (capacity, policy).
- `*_runs<N>.json`, `*_results.json`: mean error per policy; lists are ordered by `capacity_agent0` = 0.2, 0.4, 0.5, 0.6, 0.8.
- `*_agents.json`: written by the main, step-level, multi-seed and free-agent scripts. It records the seed, the number of online tasks T, each agent's feature list and its standalone accuracy on the T tasks (`agent1_logistic_regression` = paper Agent 2, `agent2_xgboost` = paper Agent 1, plus `free_agent` for the free-agent run). These accuracies give Table 1, the non-contextual baseline (Tables 2 and 3, dashed lines) and the endpoints of the capacity plots. For the saved results, these files were extracted from the console output of the original runs, which only printed these values.
- `*_step_actions_packed.npz`: the agent chosen at every step, for every policy, capacity and run (0 = logistic regression, 1 = XGBoost). The scripts save these as pickled dicts of int64 arrays (`*_step_actions.npy`); to keep the folder small they are stored here bit-packed, which is lossless. Key `"<policy>__<capacity_agent0>"` holds `np.packbits(actions, axis=1)` for an array of shape (runs, T), and key `"T"` holds T. The notebook's `load_actions()` restores the original `{policy: {capacity: array}}` dict with `np.unpackbits(..., axis=1, count=T)`. Step-level rewards are not included; they can be regenerated with `main_100runs_step_level.py`.

## Data

The four datasets are public: Bank Marketing, Default of Credit Card Clients and In-Vehicle Coupon Recommendation from the UCI Machine Learning Repository, and the Cardiovascular Disease dataset from Kaggle. All preprocessing (feature selection, encoding, train/test split) is done inside the scripts.

## Reproducibility notes

- Results were produced on CPU (Linux) with Python 3.11, numpy 2.4, pandas 3.0, scikit-learn 1.8, xgboost 3.2 and ortools 9.15. Rerunning the scripts reproduces the feature splits, agent accuracies and the greedy / tree-based policies; Logistic TS can differ slightly across machines because its multivariate-normal posterior draws depend on the linear-algebra backend.
- The saved results were produced by an earlier version of these scripts that also supported other agent pairs through a `model_mode` argument; the paper setting was `model_mode = 2`, which is exactly the logistic regression + XGBoost pair the scripts now hard-code. That is why some saved CSV columns and file names still contain `model_mode` or `mode_2`.
- Credit's `main_100runs/` and `step_level_100runs/` come from two separate executions of the same experiment. Logistic greedy agrees run for run; the randomized policies differ only through their random draws (100-run means within 0.0006). Table 2 uses `main_100runs/`, Fig. 5(c) uses `step_level_100runs/`.
