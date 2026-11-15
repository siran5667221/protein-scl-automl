# SubCELL feature extraction

## Files and source

- `scripts/features/extract_subcell_homology_aware.py` is adapted from `run_parallel_graphpart17100_all.py`.
- `scripts/features/extract_subcell_random_split.py` is adapted from `run_parallel_ifeature_one_split.py`.
- `scripts/features/util_features.py` is reconstructed from the supplied `util_features-checkpoint.py`, with configurable input/parameter paths.
- `parameters/Protein_parameters_setting.json` is copied unchanged. It matches the settings printed in `ifeature information.ipynb`.
- The original homology failed-descriptor report is preserved under `results/feature_extraction/subcell_homology_aware/`.

Both supplied entry scripts request the same 70 descriptor types in the same order. Those lists and the existing feature-to-metadata merge functions are retained. Scripts contain no comments or explanatory docstrings.

## Path repair and reproducibility status

The recovered helper hard-codes the Reference test FASTA. The Random entry point instead expects an environment-specified FASTA, and the homology entry point tries to set helper globals. Neither changes a hard-coded literal inside that recovered helper. That backup therefore cannot establish the exact helper used for the final SubCELL runs.

This package repairs input transport: the helper reads `IFEATURE_FASTA`, and `IFEATURE_PARAMETERS` optionally specifies the parameter JSON. The homology entry point sets the input environment in both parent and spawned workers. Imports use the colocated helper rather than an external Downloads directory. This is a path-repaired release reconstruction, not a verified byte-identical copy of the historical final extraction environment.

The descriptor API calls remain `iProtein(input)`, `import_parameters(parameters)`, `get_descriptor(type)`, with the same dataframe/error collection and successful-descriptor concatenation. Existing biological sequences and feature values have not been modified or regenerated. A successful historical-matrix reproduction has not been established. Exact library/dependency versions remain unverified. The original homology-aware CSV header is included as a schema reference; numerical reproduction remains unverified.

## Inputs

Use original FASTAs with their original header IDs and record order. The Random manifest covers 18,097 IDs, split into 12,667 train / 5,430 test. Run Random extraction separately on the original train and test FASTAs.

The supplied homology entry point defaults to the 18,097-sequence new-only FASTA and merges extracted descriptors with the saved 17,100-ID GraphPart split. Consequently the joined output retains 12,026 train / 5,074 test rows. That input/inner-join design is retained; the file name alone does not imply only 17,100 input FASTA records were processed.

## Homology-aware command

From repository root, in the original iFeature environment:

```bash
IFEATURE_FASTA=/path/to/subcell_lego11_unique_sequences_vs_old.fasta \
IFEATURE_PARAMETERS=parameters/Protein_parameters_setting.json \
IFEATURE_OUTPUT_DIR=runs/ifeature_outputs_subcell_graphpart17100 \
python scripts/features/extract_subcell_homology_aware.py
```

The saved homology manifest under `data/splits/subcell/` is the default split input. Override it with `IFEATURE_SPLIT_CSV=/path/to/manifest.csv` when needed. The script retains eight spawned workers, full CSV/PKL exports and separate train/test CSV exports.

## Random commands

```bash
IFEATURE_FASTA=/path/to/subcell_lego11_unique_random70_train.fasta \
IFEATURE_PARAMETERS=parameters/Protein_parameters_setting.json \
python scripts/features/extract_subcell_random_split.py --split train

IFEATURE_FASTA=/path/to/subcell_lego11_unique_random30_test.fasta \
IFEATURE_PARAMETERS=parameters/Protein_parameters_setting.json \
python scripts/features/extract_subcell_random_split.py --split test
```

Options `--split_csv`, `--out_dir`, `--prefix` and `--max_workers` retain their existing names. Defaults use the archived Random manifest under `data/splits/subcell/` and write new outputs under `runs/ifeature_outputs_subcell_random18097_separate/`. Large matrices and input sequences are not bundled.

## Recorded unsupported descriptors

The supplied homology report lists 20 failed descriptor types with equal-sequence-length requirements. It includes binary encodings, AESNN3, OPF encodings, AAIndex, BLOSUM62, ZScale and KNN. The original pipeline retains successful descriptor tables and records failures; it does not pad, truncate or impute failed types in these scripts. The supplied original matrix header confirms 44,158 descriptor columns, following four metadata columns. The failure report alone does not establish numerical reproduction.

## Archived homology-aware schema

`results/feature_extraction/subcell_homology_aware/subcell_graphpart17100_columns.csv` preserves the supplied header inventory unchanged, including positions and repeated names. The archived CSV contains 44,162 columns: `protein_id`, `label`, `partition`, `split`, followed by 44,158 descriptor columns. There are 13,408 distinct descriptor names: 3,158 occur once and 10,250 occur four times.

Overlapping names from descriptor tables are retained during concatenation. The supplied entry points then select columns using a list that itself contains repeated names. In pandas, selecting a name twice when it labels two columns returns four columns, repeating both original columns. The release retains this selection behavior to preserve the archived representation. No columns are removed, renamed or deduplicated here. Header inspection does not compare numeric values or establish which columns a downstream training script used. The Random matrix header has not been supplied.

## Checks

All scripts pass Python syntax and comment-removal checks. Descriptor lists/order and metadata-merge functions match the supplied entry points; parameter JSON and the original failure report retain their bytes. End-to-end multiprocessing tests with a small fake descriptor provider confirm path propagation and metadata joins for both entry points, including spawned workers. These tests do not execute iFeatureOmega, compute biological descriptors, verify numerical equality with the archived matrices or rerun the experiments. The homology-aware column count and order are documented from the supplied header, rather than from a new extraction.
