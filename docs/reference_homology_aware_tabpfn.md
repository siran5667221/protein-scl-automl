# Reference homology-aware TabPFN

## Source and results

The combined training/evaluation script comes from `tabpfn.ipynb`, cell index 20 (zero-based), execution count 14. Historical input filenames, output filenames and condition labels use `NoHomology`.

The supplied result CSV is copied unchanged. It contains all 100 workflows (10 repetitions by 10 folds). Each workflow's validation and test MCC matches the corresponding Notebook log to four decimal places. Development fold sizes sum to 9,901 rows; every workflow evaluates 3,993 test rows and retains 100 feature columns. Test MCC mean: 0.4966835468; median: 0.4983735928; sample SD: 0.0076049761.

## Experimental settings

Outer split seeds are `run_id * 777`; RF-RFE seeds are `run_id * 100 + fold_id`. MinMax scaling and RF-RFE are fitted within each outer-training fold. RF-RFE uses a 50-tree random forest, balanced class weights, step 0.1 and 100 selected columns. The TabPFN constructor retains `device="cpu"` and `ignore_pretraining_limits=True`. The script enables `TABPFN_ALLOW_CPU_LARGE_DATASET`. No additional model seed or training option is introduced.

The original class mapping and probability alignment to IDs 0 through 10 are retained. Each workflow saves selected columns, validation/test predictions, metrics and metadata. Resume behavior and metric definitions are unchanged. This package contains overall workflow results; a localisation-specific result table for this condition has not been supplied.

## Run from repository root

Use the original compatible TabPFN environment and pretrained weights. Exact installed versions are not established by these files. Input matrices, weights and per-workflow predictions are not bundled. Running this command trains models and is not needed to inspect the archived CSV.

```bash
REFERENCE_TRAIN_CSV=/path/to/Train_Features_NoHomology.csv \
REFERENCE_TEST_CSV=/path/to/Test_Features_NoHomology.csv \
TABPFN_OUTPUT_DIR=/path/to/workflow_output \
python scripts/tabpfn/train_reference_homology_aware.py
```

## Changes and checks

Comments and docstrings are omitted; three input/output path assignments are configurable. The remaining script AST matches the original source cell and passes syntax checks. Workflow coverage, seeds, row counts and all 100 logged validation/test MCC pairs were verified. No model training, prediction regeneration or manuscript comparison was performed. Experimental parameters and archived results are unchanged.
