# protein-scl-automl

Code and archived results for benchmarking AutoML workflows for protein subcellular localisation prediction.

## Research scope

This project examines predictive reliability and variation across complete fitted workflows, rather than reporting only the highest-scoring model. It compares random and homology-aware partitioning, overall and localisation-specific performance, validation-to-test ranking correspondence, and the consistency of classifier and descriptor selection.

- **Primary frameworks:** TPOT and Auto-sklearn, evaluated on the Reference and SubCELL-derived benchmarks under both partitioning regimes: eight conditions.
- **Supplementary learners:** AutoGluon SingleBest and TabPFN, evaluated on the Reference benchmark under both regimes: four conditions.
- **Task:** sequence-based, single-label multiclass prediction with 11 response classes. The Cytoplasm-Nucleus composite is treated as one class.
- **Evaluation unit:** one complete repeat-fold workflow. Each condition has 10 repetitions of stratified 10-fold outer evaluation, yielding 100 workflows evaluated on that condition's fixed test set.

The repository currently includes training/evaluation scripts, archived metric tables, SubCELL split manifests and SubCELL feature-extraction components. It is being assembled as a research release; full reproduction from raw sequences has not yet been verified in a fresh environment.

## Data and feature representation

The Reference benchmark is derived from DeepLoc data distributed through LEGO-CSM. The SubCELL-derived benchmark was prepared independently from SubCELL annotations and sequences.

The following sizes correspond to the archived experiments in this repository:

| Benchmark | Partitioning | Development/train records | Test records |
| --- | --- | ---: | ---: |
| Reference | Random | 9,802 | 4,202 |
| Reference | Homology-aware | 9,901 | 3,993 |
| SubCELL-derived | Random | 12,667 | 5,430 |
| SubCELL-derived | Homology-aware | 12,026 | 5,074 |

These are condition-specific populations; comparisons between regimes do not hold test membership and class composition constant. The retained SubCELL homology-aware set contains 17,100 proteins, with GraphPart partitions 1, 2 and 4 assigned to the test set. The Random SubCELL manifest contains 18,097 proteins. See the [SubCELL split documentation](docs/subcell_final_splits.md).

Inputs are explicit amino-acid sequence descriptors generated using iFeatureOmega-CLI, not protein-language-model embeddings. Archived feature matrices contain **44,158 descriptor columns**. Within each outer-training fold, MinMax scaling is fitted and random-forest recursive feature elimination (RF-RFE) retains 100 columns; the fitted transformations are then applied to validation and test data.

The archived feature schema includes repeated descriptor names and retained column-selection behaviour. The column count must not be interpreted as 44,158 distinct descriptor names or independent quantities. See the [feature-extraction documentation](docs/subcell_feature_extraction.md) before changing column names, order or duplicate handling.

Scaling and RF-RFE precede the downstream learner's internal model-selection procedure; they are not fitted on the outer-validation or test data. They are not separately refitted within each internal search fold. Framework-specific parameters and random-state schedules are recorded in the individual experiment guides.

### Data availability

The [SubCELL split manifests](data/splits/subcell/) and [descriptor parameter JSON](parameters/Protein_parameters_setting.json) are included. The full raw sequence collections, original large feature matrices and trained model archives are **not bundled**. A complete data-acquisition/download manifest and the full upstream Reference/SubCELL preparation workflow remain to be added.

Until those materials are available, use the exact original inputs described in each experiment guide. A split manifest is an assignment table, not a feature matrix. Do not substitute a similarly named file without checking IDs, labels, split membership and feature-column order.

## Archived test MCC

Each row summarises **100 individual workflows**. Mean and median are computed from the per-workflow multiclass test MCC values. They are not outer-validation scores, class-level averages, pooled-prediction MCC or voting-ensemble scores.

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

The archived summaries show lower mean and median test MCC under homology-aware partitioning in every corresponding learner/benchmark comparison. TPOT and Auto-sklearn have similar Reference test medians; Auto-sklearn has the higher SubCELL homology-aware test median. In the supplementary Reference comparisons, TabPFN has the highest test MCC among the four displayed learners, but the homology-aware reduction remains. These are results for the specified descriptor-based workflows and evaluation conditions, not a claim of superiority over all localisation predictors.

Repeated workflows share data and a fixed test set. Their score distributions describe workflow variability, not 100 independent test cohorts. Differences between partitioning regimes do not isolate a causal effect of sequence similarity.

The AutoGluon Random row uses the final **internal-holdout** results. Outer-validation data are not supplied as tuning data. Historical filenames containing `NoHomology` or `nohomology` are retained for traceability; the corresponding rows above identify the homology-aware condition.

## Repository layout

| Location | Contents |
| --- | --- |
| [data/splits/subcell/](data/splits/subcell/) | Archived protein-level Random and GraphPart-based split assignments. |
| [parameters/](parameters/) | Descriptor-generation parameter JSON. |
| [scripts/data/](scripts/data/) | SubCELL Random splitting and selection of test partitions from saved GraphPart output. |
| [scripts/features/](scripts/features/) | SubCELL feature extraction and its helper. |
| [scripts/tpot/](scripts/tpot/) | Reference/SubCELL TPOT training and evaluation. |
| [scripts/autosklearn/](scripts/autosklearn/) | Reference/SubCELL Auto-sklearn training and evaluation. |
| [scripts/autogluon/](scripts/autogluon/) | Reference AutoGluon SingleBest training and evaluation. |
| [scripts/tabpfn/](scripts/tabpfn/) | Reference TabPFN training/evaluation and Random prediction-based metric export. |
| [results/](results/) | Archived validation, overall test and available localisation-specific metric tables. |
| [docs/](docs/) | Experiment-specific inputs, settings, commands and verification scope. |
| `runs/` | Default location for newly generated outputs; not an archived-result source. |

## How to use the repository

### 1. Inspect the existing results

No model training is needed to read the CSVs in [results/](results/). Start with the MCC source links above and the matching experiment guide below. In primary `nested_cv_results.csv` tables, `MCC` denotes outer-validation performance; use the corresponding test table for independent test summaries.

### 2. Prepare data and descriptors

Run commands from the repository root in the appropriate software environment.

For SubCELL, the available preparation order is:

1. Obtain the original new-only FASTA described in the [split guide](docs/subcell_final_splits.md).
2. Use the saved split manifests, or reproduce the documented split-manifest stage.
3. Extract descriptors using the original FASTA headers and [parameter JSON](parameters/Protein_parameters_setting.json).
4. Supply the resulting matrices to the selected training script using its documented path options.

The homology split script consumes the **saved GraphPart assignment table**; it does not rerun the sequence-similarity/GraphPart calculation. The extraction helper is a reconstruction with repaired input paths; numerical reproduction of the full historical matrix has not yet been established.

The split CLI help commands do not train models:

```bash
python scripts/data/prepare_subcell_random_split.py --help
python scripts/data/prepare_subcell_homology_split.py --help
```

See [feature extraction](docs/subcell_feature_extraction.md) for the separate Random train/test FASTA commands and the homology-aware extraction command. Reference experiments require the exact original Reference matrices described in their individual guides.

### 3. Train or reevaluate a condition

Use the guide for the intended dataset, partition and learner. Input defaults are not uniform across components: some scripts use command-line arguments and others use environment variables.

| Learner | Reference Random | Reference Homology-aware | SubCELL Random | SubCELL Homology-aware |
| --- | --- | --- | --- | --- |
| TPOT | [Guide](docs/reference_random_tpot.md) | [Guide](docs/reference_homology_aware_tpot.md) | [Guide](docs/subcell_random_tpot.md) | [Guide](docs/subcell_homology_aware_tpot.md) |
| Auto-sklearn | [Guide](docs/reference_random_autosklearn.md) | [Guide](docs/reference_homology_aware_autosklearn.md) | [Guide](docs/subcell_random_autosklearn.md) | [Guide](docs/subcell_homology_aware_autosklearn.md) |
| AutoGluon | [Guide](docs/reference_random_autogluon.md) | [Guide](docs/reference_homology_aware_autogluon.md) | Not included | Not included |
| TabPFN | [Guide](docs/reference_random_tabpfn.md) | [Guide](docs/reference_homology_aware_tabpfn.md) | Not included | Not included |

For example, after supplying the exact Reference Random inputs and activating a compatible TPOT environment:

```bash
python scripts/tpot/train_reference_random.py --help
python scripts/tpot/evaluate_reference_random.py --help
```

Training launches the experiments. Reevaluation requires the saved fitted components or prediction files specified in the guide; it is not equivalent to retraining. Supplementary scripts may perform fitting and test evaluation in the same program.

### 4. Recalculate metrics from saved predictions

For Reference Random TabPFN, the provided metric script reads existing per-workflow `locked_test_predictions.csv` files:

```bash
TABPFN_OUTPUT_DIR=/path/to/existing/tabpfn_10x10_100workflows \
python scripts/tabpfn/evaluate_reference_random_predictions.py
```

This command computes metrics without fitting a new model. The prediction files themselves are not bundled. Other conditions have their own model-loading/evaluation requirements; consult their guides rather than substituting this command.

## Environments and computational requirements

The original work used separate environments for TPOT, Auto-sklearn, AutoGluon, TabPFN and descriptor generation. Verified, complete per-environment dependency exports are not yet included; there is no validated single `requirements.txt` or `environment.yml` for all components. Use the original compatible environments where available, especially when loading serialised estimators.

The supplied TabPFN scripts explicitly use CPU execution and require compatible pretrained weights. The provided SubCELL extraction scripts default to eight worker processes. Learner parallelism and memory settings differ by experiment and are listed in the individual guides.

TPOT and Auto-sklearn scripts use up to 120 minutes of search time per workflow; the AutoGluon scripts set a 7,200-second fit budget. These are configured search/fit budgets, not measured end-to-end runtimes: descriptor extraction, RF-RFE, evaluation and I/O add work. TabPFN runtime is recorded in its archived tables. Each full condition involves 100 workflows, so full training can require substantial CPU time and memory. This repository does not establish a controlled hardware-normalised speed comparison.

## Verification and release status

Archived result tables retain their supplied values. The README MCC summary was recalculated from 100 non-missing test MCC records per condition, with unique repeat-fold identifiers; source files were checked against the published repository versions. Component guides describe their syntax checks, structural comparisons, log checks and table-consistency checks.

These checks do not establish that the whole project has been rerun successfully from raw data in a clean environment. Remaining release materials include complete data-access/provenance instructions, verified environment exports, the full upstream preparation workflow, and additional figure-generation, workflow-composition and descriptor-stability analysis scripts.

## Citation and licence

Project author: **Siran Huang**.

Manuscript citation details, a formal citation file and a repository licence have not yet been added. They will be provided after the publication metadata and appropriate code/data permissions are confirmed. The upstream datasets and third-party packages retain their own terms.
