# SubCELL homology-aware TPOT

## Sources and archived results

Training and evaluation scripts come from code cells 4 and 7 (zero-based) of the supplied `subcell_class11_TPOT.ipynb`. The historical output directory contains `nohomology` in its name; the actual training code uses the GraphPart-17,100 feature file and its saved train/test split. The original code validates 17,100 total rows, 12,026 training rows, 5,074 test rows and 44,158 feature columns, then retains training rows for fitting.

Both supplied result tables have 100 workflows. Their run/fold/workflow IDs, seeds, selected-feature counts and saved pipeline paths match. Every validation MCC and test MCC matches the corresponding Notebook log to four decimal places. Validation MCC mean: 0.5030639066; median: 0.5029727314. Test MCC mean: 0.3589182940; median: 0.3646225829; sample SD: 0.0273320462. These are individual-workflow test results. The partial test table equals the final table and is not bundled twice.

The result CSVs retain their original bytes; downloaded filename suffixes are removed. Historical model paths inside the tables are preserved. This package does not reconcile these results with a manuscript or establish that another reported value belongs to the same experiment.

## Training design

Ten repetitions of stratified 10-fold outer evaluation produce 100 workflows. Unlike the Random script, the outer split seed here is `100 + run_id - 1`. The workflow seed is `100 + pipeline_id - 1`, covering 100–199. MinMax scaling and RF-RFE are fitted on the outer-training fold only. RF-RFE uses 50 trees, balanced class weights, step 0.1 and 100 retained columns. All TPOT constructor settings, internal CV, preprocessing, probability handling, resume validation and evaluation aggregation are retained from the supplied Notebook.

## Run from repository root

Use the original compatible TPOT environment; its exact dependency versions are not established by these files. Training runs the experiment and is not required to inspect the archived tables.

```bash
SUBCELL_FEATURES_CSV=/path/to/subcell_graphpart17100_full_features_with_split.csv \
TPOT_PIPELINE_DIR=/path/to/workflow_output \
python scripts/tpot/train_subcell_homology_aware.py
```

Evaluation requires the full saved workflow directory, including feature schema, label encoder and fitted preprocessing/model objects:

```bash
SUBCELL_TEST_CSV=/path/to/subcell_graphpart17100_test_features_with_labels.csv \
TPOT_PIPELINE_DIR=/path/to/workflow_output \
TPOT_EVALUATION_DIR=/path/to/new_evaluation_output \
python scripts/tpot/evaluate_subcell_homology_aware.py
```

The scripts retain the original individual-workflow evaluation and additional voting outputs without selecting a workflow using test scores. Large feature matrices, models and shared fitted metadata are not bundled.

## Checks and changes

Comments and docstrings are omitted; five path assignments are configurable. The remaining script ASTs match the original Notebook. Python syntax checks pass. Workflow identifiers and logged MCC values were checked against the supplied CSVs. No training or model reevaluation was performed. Feature columns and experimental settings are not changed.
