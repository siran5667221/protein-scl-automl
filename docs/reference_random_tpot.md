# Reference Random TPOT

This is one component of the protein-scl-automl repository, not the complete study release.

## Source and scope

The training script is extracted from the first training cell of `final tpot_rfe.ipynb`; the evaluation script is extracted from its final per-location evaluation cell. Duplicate audit cells and notebook outputs are omitted. Original modelling settings and metric formulas are retained. Local paths are replaced by command-line options, a main guard is added, training refuses a nonempty output directory, and evaluation requires all 100 workflows to succeed before exporting full results.

## Inputs

Provide the exact original `train_numeric_labels.csv` and `test_numeric_labels.csv`. Both contain `ProteinID`, integer `label`, and the same ordered numeric descriptor columns. These inputs are not included in this component. Data acquisition, provenance and split manifests will be documented as the remaining study components are assembled. The supplied notebook records 4,202 test rows; the matching Random development pool contains 9,802 rows.

Default locations, relative to the repository root:

- `data/reference/random/train_numeric_labels.csv`
- `data/reference/random/test_numeric_labels.csv`

## Original modelling settings

- 10 repetitions of stratified 10-fold outer evaluation; split seed = repeat × 777.
- MinMaxScaler and RF-RFE are fitted on each outer training fold.
- RF: 50 trees, balanced class weights, n_jobs=-1, seed = repeat × 100 + fold.
- RF-RFE: 100 features; step=0.1.
- TPOT: 100 generations, population_size=50, offspring_size=None, MCC scoring, internal cv=10, max_time_mins=120, max_eval_time_mins=10, early_stop=5, n_jobs=-1, template=None, verbosity=0, seed matching RF-RFE.
- No `config_dict` is specified.
- Scaling and RF-RFE precede TPOT internal CV. They are not refitted separately within TPOT internal folds. The outer validation and test data do not fit these preprocessing steps.
- The notebook reports TPOT 0.12.2. Exact remaining dependency versions still need to be recovered from the original environment. No unverified dependency pins are provided here.

## Running

Run in the original TPOT environment, from the repository root. These are full, computationally expensive experiments; `--help` only displays options and does not train.

```bash
python scripts/tpot/train_reference_random.py --help
python scripts/tpot/evaluate_reference_random.py --help
```

From-scratch training, followed by test evaluation:

```bash
python scripts/tpot/train_reference_random.py
python scripts/tpot/evaluate_reference_random.py
```

To evaluate the existing server models without retraining:

```bash
python scripts/tpot/evaluate_reference_random.py \
  --model-dir /home/shuang/siran/protein_tpot_rfe_longtime_final \
  --test-csv /home/shuang/siran/test_numeric_labels.csv \
  --output-dir runs/tpot_reference_random_reevaluation
```

Paths can be supplied explicitly using `--train-csv`, `--test-csv`, `--model-dir` and `--output-dir` as applicable. Defaults write new runs under `runs/`; archived CSVs under `results/` are never the default evaluation output. Training has no resume mode. Evaluation reads all 100 `complete_pipeline.pkl` files using their saved scaler, RFE and fitted model. Models are not included in this component; test reevaluation requires existing models or completed training.

## Archived results

The two supplied CSVs are copied without modifying their bytes:

- `results/tpot/reference_random/nested_cv_results.csv`: outer-validation metrics for all 100 workflows.
- `results/tpot/reference_random/overall_pipeline_metrics.csv`: test metrics for the same 100 workflows. This export contains label-based metrics, not ROC-AUC or PR-AUC.

Both tables contain each repeat-fold pair once. Validation seeds match repeat × 100 + fold. Validation ROC-AUC and PR-AUC are missing for R5/F6 and R8/F4. The original code records missing AUCs when probability predictions are unavailable; the exact cause for these two fits has not been independently established. Do not replace missing AUCs with zero or fabricated scores. All validation MCC and test table entries are present.

| Metric | Outer validation | Test |
| --- | ---: | ---: |
| MCC mean | 0.556407 | 0.553255 |
| MCC sample SD | 0.016801 | 0.007526 |
| MCC median | 0.554434 | 0.553314 |

## Retrospective maximum

The original evaluation code additionally exports `best_pipeline_result/`, selecting the workflow with the highest observed test MCC. This export is preserved for traceability and is a retrospective descriptive maximum. It is not an independently selected deployable model or an unbiased estimate of model-selection performance. Primary summaries use all 100 workflows.

## Validation performed on this package

All original nonempty code cells and both extracted scripts parse successfully. Both CLI help commands execute without starting training. TPOT_CONFIG matches the original notebook exactly; archived result files match their uploaded bytes. Model training and pickle-based test evaluation have not been rerun here. Data, split manifests, dependency verification and the other study experiments remain to be integrated.
