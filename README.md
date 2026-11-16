# protein-scl-automl

Code and archived results for benchmarking AutoML workflows for protein subcellular localisation prediction.

## Research scope

The primary experiments compare **TPOT** and **Auto-sklearn** on the **Reference** and **SubCELL-derived** benchmarks under random and homology-aware partitioning. Supplementary Reference experiments use **AutoGluon SingleBest** and **TabPFN**.

The task is **11-class, single-label multiclass classification**. Cytoplasm-Nucleus is treated as one response class. Each condition uses 10 repetitions of stratified 10-fold outer evaluation, producing **100 fitted workflows**, each evaluated on the same fixed test set for that condition.

The study examines overall and localisation-specific prediction, validation-to-test correspondence, classifier choice and RF-RFE descriptor-selection consistency.

## Data and preprocessing

The following sizes correspond to the archived experiments:

| Benchmark | Partitioning | Development/train records | Test records |
| --- | --- | ---: | ---: |
| Reference | Random | 9,802 | 4,202 |
| Reference | Homology-aware | 9,901 | 3,993 |
| SubCELL-derived | Random | 12,667 | 5,430 |
| SubCELL-derived | Homology-aware | 12,026 | 5,074 |

SubCELL homology-aware test data combine GraphPart partitions 1, 2 and 4. The saved assignments are in [data/splits/subcell/](data/splits/subcell/); the splitting procedure is described in the [split guide](docs/subcell_final_splits.md).

Inputs are explicit sequence descriptors generated with iFeatureOmega-CLI. The archived matrices contain **44,158 descriptor columns**. Within each outer-training fold, MinMax scaling is fitted and RF-RFE retains **100 columns**. The fitted transformations are applied to outer-validation and test data.

Scaling and RF-RFE precede the learner's internal model-selection procedure and are not refitted within each internal search fold. See the individual guides for the original framework settings and seed schedules. The archived SubCELL header contains repeated names; the [feature-extraction guide](docs/subcell_feature_extraction.md) documents the preserved column structure.

Split manifests and the [descriptor parameter JSON](parameters/Protein_parameters_setting.json) are included. Full raw sequence collections, large feature matrices and trained model archives are not bundled.

## Independent test MCC

Each row summarises **100 individual workflows**. The mean and median are calculated from per-workflow multiclass test MCC, not validation scores or per-class MCC.

| Benchmark | Learner | Partitioning | Mean test MCC | Median test MCC | Source |
| --- | --- | --- | ---: | ---: | --- |
| Reference | TPOT | Random | 0.553255 | 0.553314 | [CSV](results/tpot/reference_random/overall_pipeline_metrics.csv) |
| Reference | TPOT | Homology-aware | 0.448515 | 0.451380 | [CSV](results/tpot/reference_homology_aware/tpot_test_all_pipelines_summary.csv) |
| Reference | Auto-sklearn | Random | 0.550683 | 0.553394 | [CSV](results/autosklearn/reference_random/full_multi_metric_test_results.csv) |
| Reference | Auto-sklearn | Homology-aware | 0.450872 | 0.453964 | [CSV](results/autosklearn/reference_homology_aware/overall_pipeline_metrics.csv) |
| SubCELL-derived | TPOT | Random | 0.480539 | 0.480754 | [CSV](results/tpot/subcell_random/per_pipeline_test_metrics.csv) |
| SubCELL-derived | TPOT | Homology-aware | 0.358918 | 0.364623 | [CSV](results/tpot/subcell_homology_aware/per_pipeline_test_metrics.csv) |
| SubCELL-derived | Auto-sklearn | Random | 0.482389 | 0.483402 | [CSV](results/autosklearn/subcell_random/per_pipeline_test_metrics.csv) |
| SubCELL-derived | Auto-sklearn | Homology-aware | 0.425792 | 0.426669 | [CSV](results/autosklearn/subcell_homology_aware/per_pipeline_test_metrics.csv) |
| Reference | AutoGluon SingleBest | Random, internal holdout | 0.573981 | 0.575238 | [CSV](results/autogluon/autogluon_singlebest_random_internalholdout_results_100workflows.csv) |
| Reference | AutoGluon SingleBest | Homology-aware | 0.465519 | 0.466640 | [CSV](results/autogluon/autogluon_singlebest_nohomology_results_100workflows.csv) |
| Reference | TabPFN | Random | 0.614827 | 0.614854 | [CSV](results/tabpfn/reference_random/tabpfn_locked_test_results_100workflows.csv) |
| Reference | TabPFN | Homology-aware | 0.496684 | 0.498374 | [CSV](results/tabpfn/reference_homology_aware/tabpfn_nohomology_locked_test_results_100workflows.csv) |

Mean and median test MCC are lower under homology-aware partitioning in every corresponding learner/benchmark comparison. TPOT and Auto-sklearn have similar Reference test medians; Auto-sklearn has the higher SubCELL homology-aware median. TabPFN has the highest test MCC among the four evaluated Reference learners.

These comparisons concern the archived descriptor-based experiments. Test membership and composition differ between partitioning regimes. The 100 workflows share data and a fixed test set, so their score distributions describe workflow variability rather than independent test cohorts.

The AutoGluon Random summary uses the final **internal-holdout** version: outer-validation data are not supplied as tuning data. Historical `NoHomology` filenames are retained for the corresponding homology-aware experiments.

## Repository contents

| Directory | Contents |
| --- | --- |
| [data/splits/subcell/](data/splits/subcell/) | Saved SubCELL split assignments. |
| [parameters/](parameters/) | Descriptor-generation settings. |
| [scripts/data/](scripts/data/) | SubCELL split-manifest preparation. |
| [scripts/features/](scripts/features/) | SubCELL descriptor extraction and helper. |
| [scripts/](scripts/) | TPOT, Auto-sklearn, AutoGluon and TabPFN training/evaluation scripts. |
| [results/](results/) | Archived validation, test and available localisation-specific metrics. |
| [docs/](docs/) | Experiment-specific settings, inputs and running instructions. |

## Running the scripts

Read existing CSV results without retraining. In the primary `nested_cv_results.csv` tables, `MCC` denotes outer-validation performance; independent test results are in the separate test tables linked above.

For new runs, work from the repository root and use the exact original input matrices specified in each guide. Input paths are configured through command-line arguments or environment variables, depending on the script.

| Learner | Reference Random | Reference Homology-aware | SubCELL Random | SubCELL Homology-aware |
| --- | --- | --- | --- | --- |
| TPOT | [Guide](docs/reference_random_tpot.md) | [Guide](docs/reference_homology_aware_tpot.md) | [Guide](docs/subcell_random_tpot.md) | [Guide](docs/subcell_homology_aware_tpot.md) |
| Auto-sklearn | [Guide](docs/reference_random_autosklearn.md) | [Guide](docs/reference_homology_aware_autosklearn.md) | [Guide](docs/subcell_random_autosklearn.md) | [Guide](docs/subcell_homology_aware_autosklearn.md) |
| AutoGluon | [Guide](docs/reference_random_autogluon.md) | [Guide](docs/reference_homology_aware_autogluon.md) | Not included | Not included |
| TabPFN | [Guide](docs/reference_random_tabpfn.md) | [Guide](docs/reference_homology_aware_tabpfn.md) | Not included | Not included |

For SubCELL, the documented order is split preparation, descriptor extraction, training and test evaluation. The [split scripts](scripts/data/) consume the original FASTA or saved GraphPart assignments; the homology split script does not rerun GraphPart. See the [feature-extraction guide](docs/subcell_feature_extraction.md) for input and output paths.

Model-based reevaluation requires the corresponding saved fitted components. The Reference Random TabPFN [metric script](scripts/tabpfn/evaluate_reference_random_predictions.py) instead reads existing per-workflow test predictions without fitting a model.

The experiments used separate software environments. The supplied TabPFN scripts run on CPU. Search/fit budgets are up to 120 minutes per workflow for TPOT, Auto-sklearn and AutoGluon; these budgets exclude other workflow costs and are not measured end-to-end runtimes. Per-experiment settings and available runtime fields are documented in the guides and result tables.

## Verification scope

The test MCC summaries above were recomputed from 100 non-missing records per condition and checked against the published source CSVs. The experiment guides describe the code, log and table checks performed. Full reproduction from raw sequences in a fresh environment has not been verified.

Project author: **Siran Huang**.
