# Financial Fraud Detection under Extreme Class Imbalance

*A leakage-free comparative study of classical machine learning and tabular transformers, and a decision-level re-analysis of the same models.*
<img width="746" height="276" alt="Screenshot 2026-10-13 at 14 34 04" src="https://github.com/user-attachments/assets/b0a35273-b84d-44b4-809c-27581e2a611f" />

**Contents:** [Overview](#overview) · [Main findings](#main-findings) · [Quick start](#quick-start) · [Reproducing the results](#reproducing-the-results) · [Experimental protocol](#experimental-protocol) · [Limitations](#limitations) · [Repository contents](#repository-contents) · [Data and licences](#data-and-licences) · [Citation](#citation)


| Item | Detail |
|---|---|
| Benchmarks | ULB credit-card transactions (0.17% fraud) and BAF bank-account applications (1.1% fraud) |
| Models | Logistic regression, random forest, LightGBM, CatBoost, FT-Transformer; one-class SVM as anomaly reference |
| Imbalance strategies | None, RUS, ROS, SMOTE, SMOTE+Tomek, SMOTEENN, class weights |
| Unit of analysis | Ranking, validation-selected decision thresholds and alert workload, kept separate |

## Main findings

### Ranking (dissertation)

| Baseline model | ULB PR-AUC | BAF Base PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.6921 | 0.1432 |
| Random Forest | 0.7928 | 0.1588 |
| LightGBM | 0.8205 | 0.1768 |
| CatBoost | 0.8196 | 0.1796 |
| FT-Transformer | 0.8015 | 0.1769 |
| One-Class SVM | 0.2593 | 0.0190 |

PR-AUC is average precision, reported as a point estimate. On BAF Base, the paired bootstrap intervals for the differences among CatBoost, FT-Transformer and LightGBM all include zero (see the manuscript), so those three cannot be separated. No single model dominates every criterion.

### Thresholds, rebalancing and alerts (companion manuscript)

- **The decision rule matters more than the model on BAF.** For LightGBM, TEST F2 rises from 0.042 at threshold 0.5 to 0.316 with the DEV-selected max-F2 rule. Ranking scores do not change; recall and review workload do.
- **RUS, ROS and class weights mostly relocate the threshold.** They leave ranking and recall at a fixed review budget almost unchanged but worsen the Brier score 10-15-fold. A label-free prior-shift correction maps the selected thresholds (about 0.85) back to the unweighted range (0.05-0.065) in 11 of 12 model-strategy pairs.
- **SMOTE on one-hot inputs is different.** It lowers ranking quality for tree models (AP down 14-35%). The predefined SMOTE-NC control for the FT-Transformer reaches 0.1157 against a baseline of 0.1769, so categorical-aware oversampling did not recover it. Representation differences prevent an architecture-only causal interpretation.
- **Operating rules carry hidden costs.** Max-F2 behaves like a cost ratio of about 15-18 false alerts per missed fraud on BAF. Under the controlled BAF variants, a frozen threshold raises alert volume by 50-64% at unchanged prevalence, whereas a label-free capacity-preserving threshold stays within 0.011 F2 of a hindsight optimum.
- **Transfer is not determined by in-domain ranking.** After exact profile overlap removal, LightGBM leads Variant I AP and CatBoost leads Variants II-V, with models, preprocessors and Base thresholds frozen.
- **Interpretability has boundaries.** Compatible LR/LightGBM/CatBoost SHAP results concern feature-set agreement, not causality. Final-layer FT-Transformer attention is diffuse on average (normalised entropy 0.985) and is not a faithful attribution method.
- **SMOTE and SMOTE+Tomek can legitimately coincide.** The inspected sampler inputs have zero Tomek removals, so the two produce identical training sets.

The manuscript condenses these results into a five-step deployment protocol: keep scores probabilistic, apply the prior correction if rebalanced, set the threshold from a cost ratio or review capacity, report the rule with its workload and implied cost ratio, and monitor the alert rate.

Key tables: [ULB baseline](results/tables/ulb/baseline.tex), [BAF baseline](results/tables/baf/baseline.tex), [threshold sensitivity](results/tables/baf/threshold_f2.tex), [transfer](results/tables/baf/crossdomain_prauc.tex).

## Quick start

Requires Python 3.12. The recorded environment was Python 3.12.14 on Windows 11, an Intel Core i5-13600KF, 16 GB RAM and an NVIDIA RTX 3070 (8 GB). These are recorded conditions, not minimum specifications.

**Windows (PowerShell, tested):**

```powershell
git clone https://github.com/Applied-Intelligence-Hub/fraud-imbalance-classical-vs-transformers.git
cd fraud-imbalance-classical-vs-transformers
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

**Linux / macOS (bash, not yet tested):**

```bash
git clone https://github.com/Applied-Intelligence-Hub/fraud-imbalance-classical-vs-transformers.git
cd fraud-imbalance-classical-vs-transformers
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHONUTF8=1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` pins the CUDA 12.1 build of PyTorch 2.5.1. For a CPU-only install, keep the other pins and install the CPU wheel instead (choose one route; do not reinstall the CUDA requirements afterwards):

```powershell
$projectPackages = Get-Content requirements.txt | Where-Object { $_.Trim() -and $_ -notmatch '^(--|torch==)' }
python -m pip install $projectPackages
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

```bash
grep -Ev '^(--|torch==)|^\s*$' requirements.txt > requirements-cpu.txt
python -m pip install -r requirements-cpu.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

See the [official PyTorch versioned instructions](https://pytorch.org/get-started/previous-versions/#v251). The UTF-8 setting only prevents console encoding errors in progress messages; it does not affect any computation.

### Check the public evidence (no data, no training)

```bash
python src/export_public_evidence.py --verify
```

This checks publication hashes and internal numerical consistency. It is not an independent replication of training. <!-- TODO: add the expected success message, e.g. "All N checks passed". -->

### Verify the datasets

Download the CSVs from the original providers following [`datasets/README.md`](datasets/README.md), then check their exact hashes:

```bash
python src/verify_datasets.py
```

## Reproducing the results

### A. Decision-level analyses of the companion manuscript

These operate on the frozen DEV and TEST score arrays and need no training and no GPU.

<!-- TODO: confirm the script names, paths and the score-array location below, and remove this comment. -->

| Research question | Analysis | Script |
|---|---|---|
| RQ1 | Threshold rules, paired bootstrap | `src/threshold_study.py` |
| RQ2 | Prior-shift correction, Brier, ECE, reliability diagrams | `src/prior_correction.py` |
| RQ3 | Implied cost ratio, Bayes threshold, recall at a review budget | `src/cost_capacity.py` |
| RQ4 | Frozen vs capacity-preserving vs oracle thresholds on BAF Variants | `src/revision_transfer.py` |

```bash
python src/threshold_study.py --dataset baf_base --scores results/scores --output-dir derived/thresholds/baf_base
python src/prior_correction.py --dataset baf_base --scores results/scores --output-dir derived/prior
python src/cost_capacity.py --dataset baf_base --scores results/scores --output-dir derived/cost
python src/revision_transfer.py --scores results/scores --output-dir derived/transfer --bootstrap-iterations 1000
```

Every reported metric, threshold and confusion matrix is recomputed from the score arrays and compared with the archived values before any new analysis runs.

### B. Isolated fresh training (dissertation pipeline)

Always use a **new output directory and an explicit manifest**. Never train into the frozen public `results/` tree. This small pipeline check is not a thesis result:

```bash
python src/main.py --dataset ulb --models logreg --strategy none --sample 0.05 --n_trials 1 --bootstrap-iterations 0 --results-root runs/smoke --run-manifest runs/smoke/revision_manifest.json
```

For a full independent rerun, fit the baselines first, then the interventions, using the same new study manifest (full searches may take hours; <!-- TODO: add measured wall-clock per step -->):

```bash
python src/main.py --dataset ulb --models all --strategy none --n_trials 50 --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main.py --dataset baf_base --models all --strategy none --n_trials 50 --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main.py --dataset ulb --models logreg rf lgbm catboost --strategy all --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main.py --dataset baf_base --models logreg rf lgbm catboost --strategy all --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main_transformer.py --dataset ulb --strategy none --n_trials 50 --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main_transformer.py --dataset baf_base --strategy none --n_trials 50 --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main_transformer.py --dataset ulb --strategy all --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main_transformer.py --dataset baf_base --strategy all --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
```

`--models all` in the classical script includes the OCSVM but not the FT-Transformer. `--strategy all` means the six non-baseline strategies. The dissertation retained verified historical BAF classical models and selected hyperparameters, so fresh searches are an independent rerun, not an exact replay of those preserved runs.

Derive the baseline threshold study and run the frozen-model BAF transfer from your own run:

```bash
python src/threshold_study.py --dataset ulb --manifest runs/reproduction/revision_manifest.json --output-dir runs/reproduction/derived/thresholds/ulb_2013
python src/threshold_study.py --dataset baf_base --manifest runs/reproduction/revision_manifest.json --output-dir runs/reproduction/derived/thresholds/baf_base
python src/revision_transfer.py --manifest runs/reproduction/revision_manifest.json --models all --output-dir runs/reproduction/derived/transfer --bootstrap-iterations 1000 --device cpu
```

SHAP, attention and the predefined controls require explicitly pinned sources and assume the full author-side archive; they are not a generic replication recipe (see [`src/README.md`](src/README.md)). The reporting generator refuses incomplete primary, control or interpretability evidence and never invents absent results.

### C. Tests

Synthetic protocol tests (no full-data retraining):

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Experimental protocol

| Component | Procedure |
|---|---|
| ULB | Raw 284,807 transactions / 492 frauds; exact predictor-profile deduplication before splitting retains 283,726 profiles / 473 frauds. TEST contains 56,746 observations / 95 frauds. |
| BAF | Base and five controlled million-row variants; semantic numerical and nominal inputs. |
| Outer split | Stratified 80% DEV / 20% TEST, seed 42; transformations fitted within training partitions. |
| Baseline selection | 50 Optuna trials: five-fold inner validation for classical models; one internal DEV holdout for the FT-Transformer. ULB searches were repeated; the original BAF-selected parameters were retained. |
| Primary factorial | Five supervised models x seven strategies x two datasets, plus two separate OCSVM references: 72 primary cells. Baseline-selected parameters are reused across interventions. |
| Threshold study | Fixed 0.5, max-F1, max-F2 and the lowest feasible validation threshold with precision at least 0.5, applied to the same saved baseline TEST scores. |
| Prior-shift correction | Closed-form, label-free correction using the DEV prevalence (0.01103 for BAF, 0.00167 for ULB); strictly monotone, so ranking metrics are unchanged. |
| Capacity and cost | Top-k review budgets; cost ratio r with Bayes threshold 1/(1+r); DEV cost-optimal and TEST-oracle references. |
| BAF transfer | Common 30 predictors; Variant TEST profiles matching any Base DEV profile are removed. Retained cohorts have 177,526-178,828 rows. No Variant-specific fitting or threshold selection. Frozen, capacity-preserving and oracle thresholds are compared. |
| Absence codes | Documented numerical sentinels are preserved in the primary BAF pipeline; separate training-fitted NaN, mean-imputation and indicator controls are reported. |
| Categorical sampling | Predefined CatBoost and FT-Transformer SMOTE-NC controls complement the primary representation-dependent SMOTE path. |
| Uncertainty | 1,000 TEST-row bootstrap resamples conditional on fitted models and frozen thresholds; paired intervals for predefined controls. |



## Repository contents

```text
datasets/       Official-source instructions and exact raw-file hashes
notebooks/      Exploratory data analysis
src/            Scientific pipeline, diagnostics, decision-layer analyses and evidence checks
tests/          Synthetic protocol and integrity tests
results/        Final selected metrics, tables, figures, score arrays and provenance
docs/           Figures used in this README
```

