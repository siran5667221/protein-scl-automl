# Reference Homology-aware Auto-sklearn

This component adds the homology-aware Reference Auto-sklearn experiment to the existing repository. It uses distinct filenames from the Random component.

## Source and scope

Training is extracted from the first training cell of `homology_pipeline_autosklearn.ipynb`; test evaluation is extracted from its 100-workflow per-location evaluation cell. The earlier separate evaluation and later single-workflow inspection cells are not included in this component. The selected evaluation cell is self-contained and calculates overall and per-location metrics together.

Original model parameters, label mapping, split seeds, fitting and prediction calls, feature-order alignment, metric definitions and four-decimal per-location rounding are retained. Paths are configurable, execution uses a main guard, and training refuses a nonempty output directory. No refit, new model-selection step or additional experiment has been introduced. The evaluation loop retains the original logging and skipping of failed/missing workflows and exports `failed_pipelines.csv`; check successful counts before interpreting new exports as the full experiment.

## Inputs and recorded split

Supply the original `Train_Features_NoHomology.csv` and `Test_Features_NoHomology.csv`, with `ProteinID`, `label` and 44,158 ordered descriptor columns. Labels can be integer IDs or the original strings handled by the supplied mapping. These large input matrices are not bundled here.

The supplied Reference partition manifest `graphpart_result_7030.csv`, inspected separately, has 9,901 development and 3,993 test IDs; the notebook's recorded sizes match. The manifest and its preparation workflow will be integrated as a separate data-preparation component. The exact original GraphPart launch command was not retained in the notebook or recovered shell history; no threshold or launch command is inferred from the filename.

Default input locations relative to repository root:

- `data/reference/homology_aware/Train_Features_NoHomology.csv`
- `data/reference/homology_aware/Test_Features_NoHomology.csv`

## Original modelling settings

- 10 repeats × 10 stratified outer folds; split seed = repeat × 777.
- MinMaxScaler and RF-RFE fitted on each outer training fold only.
- RF: 50 trees, balanced class weights, n_jobs=-1, seed = repeat × 100 + fold.
- RFE: 100 features; step=0.1.
- Auto-sklearn: MCC scorer, SingleBest, ensemble_size=1, ensemble_nbest=1, time_left_for_this_task=7200, per_run_time_limit=600, n_jobs=3, memory_limit=60000, include=None, seed matching RF-RFE.
- No explicit `resampling_strategy` is supplied; the original environment's default applies.
- Fold-level scaling and RFE precede the Auto-sklearn search, rather than being refitted within each internal resample. The test set does not fit these stages.

Exact dependency versions remain to be recorded from the original environment. No guessed dependency pins are provided.

## Commands

Run from repository root in the original Auto-sklearn environment. Help commands do not train:

```bash
python scripts/autosklearn/train_reference_homology_aware.py --help
python scripts/autosklearn/evaluate_reference_homology_aware.py --help
```

From-scratch training, then evaluation:

```bash
python scripts/autosklearn/train_reference_homology_aware.py
python scripts/autosklearn/evaluate_reference_homology_aware.py
```

Training writes `runs/protein_rfe_autosklearn_nohomology/`, including `run_XX/fold_XX/complete_pipeline.pkl` with each workflow's scaler, RFE, selected features, model and validation metrics. Training has no resume mode. Model files are not bundled. Evaluation requires models from a completed training run or the existing server outputs and the matching original environment.

Evaluate existing server models without retraining:

```bash
python scripts/autosklearn/evaluate_reference_homology_aware.py \
  --model-dir /home/shuang/siran/protein_rfe_autosklearn_nohomology \
  --test-csv /home/shuang/siran/Test_Features_NoHomology.csv \
  --output-dir runs/autosklearn_reference_homology_aware_reevaluation
```

Default evaluation outputs are under `runs/`, separate from the archived CSVs in `results/`. The script exports overall metrics, per-location metrics, their summaries and the original failed-workflow report. This selected evaluation cell does not export ROC-AUC/PR-AUC; those metrics occur in a separate original evaluation cell, not integrated here.

## Archived results and verification

The three supplied CSVs are copied byte-for-byte. Only browser-added filename suffixes are removed. Validation and test tables each contain all 100 unique repeat-fold pairs. Validation seeds match repeat × 100 + fold. Per-location results contain 11 locations for each workflow, totalling 1,100 rows. No supplied table values are missing.

Per-location confusion counts total 3,993 test observations; positive counts equal support. Overall multiclass MCC reconstructed from these counts matches the overall test table. Per-location MCC matches the confusion counts at the original four-decimal precision.

| MCC statistic | Outer validation | Test |
| --- | ---: | ---: |
| Mean | 0.579427 | 0.450872 |
| Sample SD | 0.020154 | 0.013412 |
| Median | 0.580367 | 0.453964 |

Both scripts parse and their CLI help commands execute. Structural checks confirm preservation of Auto-sklearn settings and modelling calls. Archived CSV byte equality and result-table consistency checks pass. Full model training and inference have not been rerun here. This remains one component of the study release; data access, environment and remaining experiments will be integrated separately.
