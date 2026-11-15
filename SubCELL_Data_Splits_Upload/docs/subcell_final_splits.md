# SubCELL final random and homology-aware split manifests

## Scope and sources

This component packages the final split-manifest stage from `subcell dataset.ipynb`. It does not yet package the entire upstream SubCELL download, annotation mapping, sequence QC or feature-extraction workflow. Early nine-class and intermediate 28,324-sequence partition experiments are excluded from this final-split component.

The scripts contain no code comments or explanatory docstrings, as requested. Random splitting combines the original random-FASTA splitting and final protein-level CSV export cells. Homology splitting combines the original GraphPart-table preparation, partition-combination scoring and final split exports. Paths are exposed through CLI arguments; notebook display calls are replaced by print calls. Splitting algorithms and parameters are retained.

The three original manifests are copied byte-for-byte into `data/splits/subcell/`:

- `subcell_lego11_random18097_train_test_split.csv`
- `subcell_lego11_unique_vs_old_graphpart_30id_10partitions.csv`
- `subcell_lego11_unique_vs_old_graphpart_train_test_split.csv`

## Final data sizes

| Condition | Total | Development/train | Test |
| --- | ---: | ---: | ---: |
| Random | 18,097 | 12,667 | 5,430 |
| Homology-aware retained set | 17,100 | 12,026 | 5,074 |

The random manifest contains protein_id, string label, partition=-1, split and header. The original numeric GraphPart label codes map consistently to the random string labels. The homology split uses test partitions 1, 2 and 4. The retained 17,100 protein IDs are a subset of the 18,097 IDs in the Random manifest. Each manifest has unique IDs, no missing fields and no train/test ID overlap.

## Random procedure

Input: the original `subcell_lego11_unique_sequences_vs_old.fasta`. Preserve its original headers and record order: identifiers such as `unique_new_1` were assigned during preceding sequence filtering. Original sequence content is required for the exported FASTAs. That upstream FASTA is not bundled in this component.

The script parses label groups from headers, shuffles within each label using pandas sample(random_state=42), rounds class size × 0.7 for the training allocation, retains the original minimum-allocation rule, concatenates groups and shuffles each final split with random_state=42. The split CSV removes `label=` from string labels and uses the first pipe-delimited header field as protein_id. It exports train/test FASTAs, a class summary and the protein-level split CSV.

```bash
python scripts/data/prepare_subcell_random_split.py --help
python scripts/data/prepare_subcell_random_split.py \
  --input-fasta /path/to/subcell_lego11_unique_sequences_vs_old.fasta \
  --output-dir runs/subcell_random_split
```

## Homology partition selection

Input: the bundled saved GraphPart output. This script starts after GraphPart has run; it does not recreate pairwise similarities or generate new GraphPart partitions.

It enumerates every combination of three of the ten partitions. The score is the sum of absolute test-versus-overall class-frequency differences plus the absolute deviation of the test fraction from 0.30. The lowest-scoring combination is selected, reproducing partitions 1, 2 and 4. It writes the combined split CSV and separate train/test CSVs. These are manifests, not feature matrices.

```bash
python scripts/data/prepare_subcell_homology_split.py --help
python scripts/data/prepare_subcell_homology_split.py
```

The default GraphPart input is the bundled CSV under `data/splits/subcell/`; new outputs go to `runs/subcell_homology_split/`, separate from archived manifests.

The original GraphPart invocation is not present in this notebook. The filename identifies the saved 30id/10partitions result, but the exact executable invocation, alignment settings and environment must be documented from original records rather than guessed. The saved partition assignments remain available here.

## Checks performed

All 18,097 Random IDs and all 17,100 homology IDs are unique; each split is disjoint. The GraphPart AC, label-val and cluster values agree with protein_id, label and partition in the final homology manifest. Numeric/string labels map one-to-one by retained protein ID.

Both scripts pass syntax and CLI-help checks. The homology script was executed on the saved GraphPart CSV and reproduced the full final manifest, including order. The Random script was exercised using archived headers in their original numeric ID order and synthetic placeholder sequences: it reproduced the entire protein-level manifest, including split membership and row order. That test validates the splitting code only; it does not validate actual protein sequences or the upstream QC/exclusion stages. Synthetic sequences are not included in the delivered package.

No GraphPart rerun, feature extraction or model training was performed. Archive CSVs retain their original bytes. Actual sequence provenance, upstream raw-data preparation, feature extraction and dependency versions will be integrated separately.
