# Reference Homology-aware TPOT

## Authoritative outputs and source

The user identifies `protein_tpot_rfe_NoHomology_Final.tar.gz` as the final Reference homology-aware TPOT archive. The two bundled result CSVs were supplied from its `tpot_test_per_location_results` subfolder. Their historical `pkl_path` values refer to `./protein_tpot_nohomology`, the directory named in `no_homology_tpot.ipynb`. Do not rewrite these historical CSV values.

The 100 archived `val_MCC` values match the notebook's 100 printed training validation MCCs to the four-decimal precision available in that log. This supports their workflow correspondence; fitting and inference have not been rerun here.

## Source cleanup

The uploaded training cell contains pasted debugging output between `row = val_metrics.copy()` and `cv_summary_records.append(row)`, breaking the `row.update(...)` statement and causing a syntax error. The release training script removes that pasted text and restores:

```python
row.update({'run_id': run_id, 'fold_id': fold_idx, 'seed': current_seed})
```

This is a repaired release copy of the supplied source, not a claim that the uploaded malformed cell can currently execute. No model parameters, feature-selection settings or archived result values are changed. The original run is represented by its saved outputs and notebook execution log.

The evaluation script comes from the executed comprehensive test/per-location cell. The unexecuted alternative evaluation cell and external figure-generation command are not bundled. CLI paths and a main guard are added. Training refuses a nonempty output directory to avoid overwriting fitted workflows. The original evaluation error logging and summary logic are retained.

## Data and modelling settings

Supply the original `Train_Features_NoHomology.csv` and `Test_Features_NoHomology.csv` (recorded development/test sizes 9,901/3,993), with `ProteinID`, `label` and 44,158 ordered descriptor columns. Their preparation and split provenance will be integrated separately. Large matrices and fitted models are not included.

- 10 repeats × 10 stratified outer folds; split seed = repeat × 777.
- Fold-specific MinMaxScaler followed by RF-RFE; 100 features; RFE step=0.1.
- RF: 50 trees, balanced class weights, n_jobs=-1, seed = repeat × 100 + fold.
- TPOT: 100 generations, population_size=50, offspring_size=None, MCC scoring, max_time_mins=120, max_eval_time_mins=10, cv=10, verbosity=0, n_jobs=-1, early_stop=5, template=None. No explicit config_dict is supplied.
- TPOT random_state matches the RF-RFE seed. Scaling/RFE precede TPOT internal CV and are not refitted within its internal folds. The outer validation and test data do not fit those steps.
- The recorded TPOT version is 0.12.2. Other exact dependency versions still need to be recovered; no guessed environment pins are provided.

## Commands

Run from repository root in the original TPOT environment. Help commands do not train:

```bash
python scripts/tpot/train_reference_homology_aware.py --help
python scripts/tpot/evaluate_reference_homology_aware.py --help
```

Default data paths are `data/reference/homology_aware/Train_Features_NoHomology.csv` and `data/reference/homology_aware/Test_Features_NoHomology.csv`.

From-scratch training, then evaluation:

```bash
python scripts/tpot/train_reference_homology_aware.py
python scripts/tpot/evaluate_reference_homology_aware.py
```

New runs default to `runs/protein_tpot_rfe_NoHomology_Final/`; this location reflects the final archive name and differs from the original historical working-directory name. Each workflow saves `run_XX/fold_XX/complete_pipeline.pkl`. Evaluation also needs the original training CSV to align descriptor columns. Pass explicit paths to use existing extracted models without retraining:

```bash
python scripts/tpot/evaluate_reference_homology_aware.py \
  --model-dir /path/to/extracted/protein_tpot_rfe_NoHomology_Final \
  --train-csv /home/shuang/siran/Train_Features_NoHomology.csv \
  --test-csv /home/shuang/siran/Test_Features_NoHomology.csv \
  --output-dir runs/tpot_reference_homology_aware_reevaluation
```

Replace `/path/to/extracted/...` with the actual extracted folder. Default generated results are under `runs/`, separate from archived CSVs under `results/`.

The original evaluation also exports predictions, validation-versus-test MCC and a test-MCC-maximum workflow. That maximum is a retrospective descriptive result, not independent model selection. Primary distribution summaries use all 100 workflows.

## Archived results and checks

The two supplied CSVs are preserved byte-for-byte. The overall table contains validation MCC together with full test metrics for all 100 unique repeat-fold pairs; all report 100 selected features. The per-location table contains 1,100 unique workflow/class rows, representing all 11 classes for each workflow. There are no missing table values.

Class support and predicted support both total 3,993 per workflow. Integer true-positive counts reconstructed from support and recall agree with precision, per-class MCC, and overall test MCC/accuracy. Validation MCC matches the notebook execution log to displayed precision. The full original validation CSV has not been supplied, so it is not fabricated; use the archived `val_MCC` column for the supplied validation MCC values.

| MCC statistic | Outer validation (archived val_MCC) | Test |
| --- | ---: | ---: |
| Mean | 0.582933 | 0.448515 |
| Sample SD | 0.016705 | 0.015985 |
| Median | 0.581690 | 0.451380 |

The repaired scripts pass syntax and CLI-help checks. TPOT_CONFIG and modelling calls match the supplied source after removing the pasted debugging text. CSV byte equality and cross-table numerical checks pass. These checks do not rerun the models or establish exact binary equivalence of all fitted objects. Exact environment records, data preparation and other study components remain to be integrated.
