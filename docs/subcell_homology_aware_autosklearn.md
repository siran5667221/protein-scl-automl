# SubCELL homology-aware Auto-sklearn

## Sources and archived results

Training and evaluation scripts come from code cells 6 and 10 (zero-based) of the supplied `subcell_class11_autosklearn.ipynb`. The training script reads the GraphPart-17,100 matrix and validates 17,100 total rows, 12,026 training rows, 5,074 held-out test rows and 44,158 feature columns. Only training rows are used for fitting.

The two supplied CSVs contain 100 workflows with matching run/fold/workflow IDs, seeds, selected-feature counts and historical pipeline paths. All 100 test MCC values match the printed Notebook values to four decimal places. Training logs print 93 newly fitted validation MCC values, all matching their CSV workflow IDs to four decimal places. Workflows 2, 3, 5, 6, 7, 8 and 9 were skipped as already completed and are marked `resume_loaded=True` in the validation CSV; their individual validation values are not printed in this execution log.

Validation MCC mean: 0.4851441383; median: 0.4854178622. Test MCC mean: 0.4257916847; median: 0.4266689223; sample SD: 0.0104786783. CSV bytes and historical model paths are unchanged; downloaded filename suffixes are removed. The partial test table equals the final table and is not bundled twice. These results describe individual workflows.

## Training and label handling

Ten repetitions of stratified 10-fold outer evaluation yield 100 workflows. Outer split seeds are `100 + run_id - 1`; workflow seeds are `100 + pipeline_id - 1` (100–199). MinMax scaling and RF-RFE are fitted on each outer-training fold. RF-RFE uses 50 trees, balanced class weights, two jobs, step 0.1 and 100 retained columns.

Auto-sklearn optimises MCC with a 7,200-second task budget, 600-second per-run limit, SingleBest configuration, ensemble size/nbest of one, unrestricted classifier inclusion, two jobs and a 60,000-MB memory limit. No resampling-strategy override is added. Remaining constructor settings and fitting/evaluation code are preserved.

The original label encoder fits string representations of GraphPart label IDs. Its class order is `['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9']`. Model-encoded IDs therefore differ from original GraphPart IDs; the evaluation retains the saved encoder to transform test labels. This mapping must also be used when interpreting localisation-specific outputs.

## Run from repository root

Use the original compatible Auto-sklearn environment. Exact dependency versions are not established from the supplied files. Models, shared fitted metadata and large feature matrices are not bundled. Training runs the experiment and is not needed to inspect the archived CSVs.

```bash
SUBCELL_FEATURES_CSV=/path/to/subcell_graphpart17100_full_features_with_split.csv \
AUTOSKLEARN_PIPELINE_DIR=/path/to/workflow_output \
python scripts/autosklearn/train_subcell_homology_aware.py
```

Evaluation requires the complete saved workflow directory, including the shared feature schema, label encoder and fitted preprocessing/model objects:

```bash
SUBCELL_TEST_CSV=/path/to/subcell_graphpart17100_test_features_with_labels.csv \
AUTOSKLEARN_PIPELINE_DIR=/path/to/workflow_output \
AUTOSKLEARN_EVALUATION_DIR=/path/to/new_evaluation_output \
python scripts/autosklearn/evaluate_subcell_homology_aware.py
```

The original evaluation retains individual-workflow metrics and additional voting outputs without selecting a model using test scores. Voting across saved outer-fold models is separate from the training SingleBest configuration.

## Checks and changes

Comments and docstrings are omitted; five path assignments are configurable. Scripts pass Python syntax checks and otherwise match the original source-cell ASTs. Workflow metadata and available logged MCC values were checked as described above. Feature columns, numerical processing and experiment settings are retained. No model retraining, model reevaluation or manuscript comparison was performed.
