# SubCELL Random TPOT

## Source and design

Training and evaluation scripts are extracted from code cells 1 and 10 (zero-based) of the supplied `subcell_class11_TPOT.ipynb`. Comments and docstrings are omitted; only input/output path assignments are made configurable. Training parameters, feature processing, probability handling, metrics, resume logic and evaluation aggregation are retained.

The original training log reports 12,667 training rows and 44,158 feature columns. The saved Random split manifest contains 5,430 test IDs. Training uses 10 repetitions of stratified 10-fold outer evaluation, producing 100 workflows. The outer split seed is `run_id * 777`; each workflow seed is `100 + pipeline_id - 1` (100–199). Fold-specific MinMax scaling precedes RF-RFE (50 trees, balanced class weights, step 0.1, 100 retained columns). TPOT's internal CV is 10-fold, with 100 generations, population 50, 120-minute total budget, 10-minute candidate limit, two jobs and early stopping after five non-improving generations. All remaining constructor settings are retained from the Notebook.

## Archived results

The two CSVs are copied unchanged; only the downloaded validation filename suffix `(3)` is removed. Both have 100 rows with matching run/fold/workflow identifiers, seeds and pipeline paths. All 100 validation MCC values and all 100 test MCC values match the corresponding printed Notebook values to four decimal places. Validation MCC mean: 0.4857400791. Test MCC mean: 0.4805394010; median: 0.4807540599. The supplied partial test CSV equals the final CSV and is not bundled twice.

Paths inside archived result tables record historical output locations and are retained. Models, fitted scalers/RFE objects, label encoders, feature schemas and large input matrices are not included in this package.

## Run from repository root

Use the original compatible TPOT environment. Exact installed versions have not been established from these supplied files.

```bash
SUBCELL_TRAIN_CSV=/path/to/subcell_random18097_train_features_with_labels.csv \
TPOT_PIPELINE_DIR=/path/to/workflow_output \
python scripts/tpot/train_subcell_random.py
```

Training runs the experiment; it is not needed to inspect the supplied results. Evaluation requires the complete saved workflow directory, including its shared feature schema and label encoder:

```bash
SUBCELL_TEST_CSV=/path/to/subcell_random18097_test_features_with_labels.csv \
TPOT_PIPELINE_DIR=/path/to/workflow_output \
TPOT_EVALUATION_DIR=/path/to/new_evaluation_output \
python scripts/tpot/evaluate_subcell_random.py
```

The evaluation retains individual workflow metrics and the original additional voting outputs. Archived CSVs here report individual workflows. It does not select a workflow using test performance.

## Validation scope

Scripts pass syntax checks. Their ASTs match the supplied Notebook after excluding comments/docstrings and the documented path assignments. Archived table identifiers and logged MCC values were checked; models were not retrained or reevaluated, and no manuscript comparison was performed. The feature schema is read as in the original code, without deduplicating descriptor columns.
