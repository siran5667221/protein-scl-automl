# SubCELL Random Auto-sklearn

## Source and archived results

Training and evaluation scripts are extracted from code cells 1 and 3 (zero-based) of the supplied `subcell_class11_autosklearn.ipynb`. Comments and docstrings are omitted; five input/output path assignments are made configurable. Remaining code structure and experimental settings are retained.

Training logs report 12,667 rows and 44,158 feature columns. The original Random split manifest contains 5,430 test IDs. Both archived tables contain 100 workflows with matching run/fold/workflow IDs, seeds, selected-feature counts and historical pipeline paths. Every validation and test MCC matches the Notebook log to four decimal places. Validation MCC mean: 0.4911527953; median: 0.4923778521. Test MCC mean: 0.4823892620; median: 0.4834015259; sample SD: 0.0093739590.

Original CSV bytes are preserved; downloaded filename suffixes are removed. The supplied partial test table equals the final table and is not bundled twice. CSV model paths retain their historical locations.

## Training design

Ten repetitions of stratified 10-fold outer evaluation produce 100 workflows. Outer split seeds are `100 + run_id - 1`; workflow seeds are `100 + pipeline_id - 1` (100–199). MinMax scaling and RF-RFE are fitted on each outer-training fold. RF-RFE uses 50 trees, balanced class weights, two jobs, step 0.1 and 100 retained columns.

Auto-sklearn optimises MCC with a 7,200-second task budget, 600-second per-run limit, SingleBest configuration, ensemble size/nbest of one, unrestricted classifier inclusion, two jobs and a 60,000-MB memory limit. No resampling-strategy override is added; the original constructor behavior is retained. The supplied Notebook separately prints `holdout` and `None` for the current classifier's resampling settings. Training/evaluation logic, label encoding, invalid-value handling and feature order are unchanged.

## Run from repository root

Use the original compatible Auto-sklearn environment; exact dependency versions have not been established from these supplied files. Large matrices and fitted workflow objects are not included. Training runs the experiment and is not required to inspect the archived tables.

```bash
SUBCELL_TRAIN_CSV=/path/to/subcell_random18097_train_features_with_labels.csv \
AUTOSKLEARN_PIPELINE_DIR=/path/to/workflow_output \
python scripts/autosklearn/train_subcell_random.py
```

Evaluation requires the complete saved workflow directory, including the shared feature schema, label encoder and fitted preprocessing/model objects:

```bash
SUBCELL_TEST_CSV=/path/to/subcell_random18097_test_features_with_labels.csv \
AUTOSKLEARN_PIPELINE_DIR=/path/to/workflow_output \
AUTOSKLEARN_EVALUATION_DIR=/path/to/new_evaluation_output \
python scripts/autosklearn/evaluate_subcell_random.py
```

The original evaluation includes individual-workflow metrics and additional voting outputs. The archived CSV here reports individual workflows. Voting across saved outer-fold models is separate from the training SingleBest configuration. Evaluation does not select a model using test performance.

## Checks

Scripts pass syntax and no-comment/docstring checks. Their ASTs match the source cells except for the five documented path assignments. Workflow identifiers, seeds and all logged MCC values were verified against the supplied tables. Models were not retrained or reevaluated; no manuscript comparison was performed. Descriptor columns are not deduplicated.
