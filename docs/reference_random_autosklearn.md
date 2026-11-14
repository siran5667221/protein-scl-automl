# Reference Random Auto-sklearn

This component adds the Reference Random Auto-sklearn experiment to the existing repository. It is not the complete study release.

## Source and permitted cleanup

Training is extracted from the first training cell of `final_ref_autosklearn.ipynb`. Evaluation is extracted from the final per-class evaluation cell, which already produces overall test metrics and per-class MCC. Duplicate audit cells, cross-partition exploratory cells, notebook outputs and label-debugging cells are not included in this component.

The obsolete printed text about selecting a best run and refitting on the full development pool has been removed: those steps were not executed by the supplied training loop. No refit, new model-selection step or extra experiment has been introduced. Original model parameters, fitting calls, predictions, metric formulas and the four-decimal per-class MCC rounding are preserved.

Only path handling and execution packaging have changed: command-line inputs, a main guard, an output directory separate from archived CSVs, and a check refusing to train into a nonempty output directory. The original evaluation loop logs and skips failed or missing workflows; check its successful count before interpreting any new evaluation as complete.

## Original modelling settings

- 10 repeats × 10 stratified outer folds. Split seed = repeat × 777.
- MinMaxScaler and RF-RFE fitted on each outer training fold only.
- RF: 50 trees, balanced class weights, n_jobs=-1, seed = repeat × 100 + fold.
- RFE: 100 features; step=0.1.
- Auto-sklearn: MCC scorer; SingleBest, ensemble_size=1, ensemble_nbest=1; time_left_for_this_task=7200; per_run_time_limit=600; n_jobs=3; memory_limit=60000; include=None; seed matching RF-RFE.
- No explicit `resampling_strategy` is supplied. The environment's Auto-sklearn default applies; this component does not add or change it.
- Fold-level scaling and RFE precede the Auto-sklearn search. They are not refitted separately within its internal resampling.
- The original training cell reads the fixed test CSV but never passes its features or labels to fitting or selection. That loading behaviour is retained.

Exact dependency versions still need to be recorded from the original Auto-sklearn environment. No guessed version pins are provided.

## Data and models

Supply the exact original train and test CSVs, with `ProteinID`, integer `label` and matching ordered descriptor columns. The recorded Random development/test sizes are 9,802/4,202; descriptor count is 44,158. Inputs, split provenance and environment files will be integrated separately.

Default input paths relative to repository root:

- `data/reference/random/train_numeric_labels.csv`
- `data/reference/random/test_numeric_labels.csv`

Training writes `runs/protein_rfe_autosklearn_longtime_method_final/`. Each workflow saves `run_XX/fold_XX/complete_pipeline.pkl` containing its scaler, RFE, selected features, fitted model and validation metrics. Models are not bundled here. Evaluation requires those existing models or a completed training run and the matching original environment.

## Commands

Run from repository root in the original Auto-sklearn environment. Full training is expensive. Help commands do not run training:

```bash
python scripts/autosklearn/train_reference_random.py --help
python scripts/autosklearn/evaluate_reference_random.py --help
```

From-scratch training, then evaluation:

```bash
python scripts/autosklearn/train_reference_random.py
python scripts/autosklearn/evaluate_reference_random.py
```

Evaluate existing server models without retraining:

```bash
python scripts/autosklearn/evaluate_reference_random.py \
  --model-dir /home/shuang/siran/protein_rfe_autosklearn_longtime_method_final \
  --test-csv /home/shuang/siran/test_numeric_labels.csv \
  --output-dir runs/autosklearn_reference_random_reevaluation
```

Evaluation exports `full_multi_metric_test_results.csv`, `per_class_mcc_all_pipelines.csv` and `per_class_mcc_summary.csv`. Archived results under `results/` are not the default output location.

## Archived results and verification

The three supplied CSVs are preserved byte-for-byte. The browser-added `(1)` suffix is removed only from the validation filename. Both overall tables contain the same complete set of 100 unique repeat-fold identifiers. Validation seeds match repeat × 100 + fold. Per-class results contain 1,100 rows (11 locations × 100 workflows) with no duplicate workflow/class pair or missing values.

Per-class confusion counts total 4,202 observations per class/workflow; positive counts equal class support. Reconstructing overall multiclass MCC from these counts matches the test table. Per-class MCC values match their confusion counts at the original four-decimal rounding. This checks the supplied tables; it is not a model rerun.

| MCC statistic | Outer validation | Test |
| --- | ---: | ---: |
| Mean | 0.552744 | 0.550683 |
| Sample SD | 0.020812 | 0.013008 |
| Median | 0.552577 | 0.553394 |

Training and evaluation scripts pass Python syntax checks and both CLI help commands run. Structural checks confirm preservation of the original Auto-sklearn configuration and modelling calls. Model fitting and pickle-based inference have not been rerun here. Data access, split manifests and dependency verification remain to be integrated with the rest of the study.
