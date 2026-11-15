# Reference Random AutoGluon SingleBest

## Source and results

The training/evaluation script comes from the supplied `autogluon_singlebest_random_internalholdout_10x10.py` (downloaded with suffix `(1)`). It performs both model fitting and evaluation. The supplied 100-workflow CSV is preserved byte for byte, with its downloaded filename suffix removed.

The CSV records the protocol `outer_validation_for_evaluation_only_internal_holdout_selection` in every row. It contains all 10 repetitions by 10 folds, 9,802 development rows, 4,202 test rows, 100 selected columns per workflow and a 7,200-second fit budget. Outer split seeds equal `run_id * 777`; RFE seeds equal `run_id * 100 + fold_id`. Validation MCC mean: 0.5770489508. Test MCC mean: 0.5739805271; median: 0.5752383293; sample SD: 0.0103770314.

## Model fitting

MinMax scaling and RF-RFE are fitted on each outer-training fold, then applied to outer-validation and test rows. RF-RFE uses a 50-tree random forest, balanced class weights, step 0.1 and 100 retained columns. AutoGluon receives only the selected outer-training data, with `tuning_data=None`; it creates its internal holdout for model selection. Outer-validation scores are evaluation outputs and are not passed to `fit`.

Original AutoGluon settings are retained: multiclass prediction, MCC optimisation, medium-quality preset, no bagging, no stacking, no auto-stack and no weighted ensemble. Holdout fraction and feature generation remain at library defaults. The script evaluates the selected model on both outer-validation and test data. It retains the original checks for completed and partially written workflow directories.

## Run from repository root

Use the original compatible AutoGluon environment; exact installed versions are not established by these supplied files. Input features, fitted models and prediction files are not included in this package. This command trains models; it is not needed to inspect the archived CSV.

```bash
REFERENCE_TRAIN_CSV=/path/to/train_numeric_labels.csv \
REFERENCE_TEST_CSV=/path/to/test_numeric_labels.csv \
AUTOGLUON_OUTPUT_DIR=/path/to/workflow_output \
python scripts/autogluon/autogluon_singlebest_random_internalholdout_10x10.py
```

## Changes and checks

Comments and docstrings are omitted; three input/output paths are configurable. The remaining script AST matches the supplied source. Syntax, workflow coverage, protocol fields, seeds, row counts and feature counts were checked. CSV values were not modified. No new training, model reevaluation, run-log comparison or manuscript comparison was performed.
