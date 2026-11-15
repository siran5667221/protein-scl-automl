# Reference Random TabPFN

## Sources and results

Training and prediction-based metric scripts are extracted from `tabpfn.ipynb`, cell indices 17 and 18 (zero-based), execution counts 9 and 10. All three supplied CSVs are retained byte for byte.

The training/evaluation table contains 100 workflows, covering 10 repetitions by 10 outer folds. All 100 validation MCC and test MCC values match the Notebook log to four decimal places. Test MCC mean: 0.6148273249; median: 0.6148536433; sample SD: 0.005078. The full-metric table agrees with the training/evaluation table on shared overall test metrics.

The localisation table contains 1,100 rows (100 workflows by 11 classes). Per-workflow supports sum to 4,202 test records; support-weighted recall matches overall accuracy, mean recall matches balanced accuracy and mean F1 matches macro F1. This checks consistency of supplied aggregates, not individual predictions. Development fold sizes sum to 9,802.

## Training design

Outer split seeds are `run_id * 777`; RF-RFE seeds are `run_id * 100 + fold_id`. MinMax scaling and RF-RFE are fitted within each outer-training fold. RF-RFE uses a 50-tree random forest, balanced class weights, step 0.1 and 100 selected columns. The original TabPFN constructor uses `device="cpu"` and `ignore_pretraining_limits=True`. The script sets `TABPFN_ALLOW_CPU_LARGE_DATASET=true`. No new model seed or other training option is introduced.

The training script writes per-workflow test predictions and summary tables. The metric script reads the saved test predictions, calculates full overall and class-level metrics, and writes confusion matrices and summaries. Class IDs are 0 through 10 in the original prediction column order.

## Run from repository root

Use the original compatible TabPFN environment and pretrained weights. Exact package versions are not established from these files. Input matrices, predictions, model weights and fitted objects are not included. Running the training script reruns experiments and is not needed to inspect the archived CSVs.

```bash
REFERENCE_TRAIN_CSV=/path/to/train_numeric_labels.csv \
REFERENCE_TEST_CSV=/path/to/test_numeric_labels.csv \
TABPFN_OUTPUT_DIR=/path/to/workflow_output \
python scripts/tabpfn/train_reference_random.py
```

To compute metrics from existing per-workflow `locked_test_predictions.csv` files:

```bash
TABPFN_OUTPUT_DIR=/path/to/workflow_output \
python scripts/tabpfn/evaluate_reference_random_predictions.py
```

This second command does not fit a model. It writes metric outputs to the specified workflow directory. Original t-based summary interval calculations are retained; the correlated outer-fold workflows should not be treated as independent replications when interpreting those intervals.

## Changes and validation

Comments and docstrings are omitted. Four path assignments are configurable; the two Notebook `display` calls become terminal `print` calls. Remaining ASTs match the original source cells. Python syntax, workflow coverage, logged MCC values, shared overall metrics and class-level consistency were checked. No model training, prediction regeneration or manuscript comparison was performed.
