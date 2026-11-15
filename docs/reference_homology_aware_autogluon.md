# Reference homology-aware AutoGluon SingleBest

## Source and archived result

The combined training/evaluation script is extracted from `AUTOGLUON.ipynb`, code cell index 16 (zero-based), execution count 17. Historical input/output names and the result condition use `NoHomology`. The original inputs are `Train_Features_NoHomology.csv` and `Test_Features_NoHomology.csv`.

The supplied CSV is preserved byte for byte. All 100 workflow-level validation MCC values, test MCC values and selected-model names match the saved Notebook log to the printed four-decimal precision. It records 9,901 development rows, 3,993 test rows and 100 selected columns per workflow. Validation MCC mean: 0.6095873290. Test MCC mean: 0.4655193924; median: 0.4666402711; sample SD: 0.0105580032.

## Model fitting

Ten repetitions of stratified 10-fold outer evaluation yield 100 workflows. Outer split seeds are `run_id * 777`; RF-RFE seeds are `run_id * 100 + fold_id`. MinMax scaling and RF-RFE are fitted on each outer-training fold. RF-RFE uses a 50-tree random forest, balanced class weights, step 0.1 and 100 retained columns.

Only the selected outer-training data are passed to AutoGluon fit. The outer-validation data are evaluated separately and are not supplied as tuning data. Original fit settings are retained: multiclass prediction, MCC optimisation, 7,200-second fit budget, medium-quality preset, no bagging, no stacking, no auto-stack, no weighted ensemble and `keep_only_best=True`. The holdout fraction and feature-generation settings are left at library defaults. Original resume checks and output handling are retained.

## Run from repository root

Use the original compatible AutoGluon environment; exact dependency versions have not been established here. Matrices, fitted models, prediction files and preprocessing objects are not bundled. Running the script trains models and is not needed to inspect the archived CSV.

```bash
REFERENCE_TRAIN_CSV=/path/to/Train_Features_NoHomology.csv \
REFERENCE_TEST_CSV=/path/to/Test_Features_NoHomology.csv \
AUTOGLUON_OUTPUT_DIR=/path/to/workflow_output \
python scripts/autogluon/autogluon_singlebest_nohomology_10x10.py
```

## Changes and checks

Comments and docstrings are omitted. Three input/output path assignments are configurable; the remaining AST matches the original source cell. Syntax, workflow coverage, row counts, seeds and per-workflow logged model/MCC outputs were checked. No retraining, numerical model reevaluation or manuscript comparison was performed. Experimental settings and archived values are unchanged.
