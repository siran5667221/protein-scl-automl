import argparse
import concurrent.futures
from multiprocessing import get_context
from pathlib import Path
import pandas as pd
import pickle
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTIL_DIR = str(Path(__file__).resolve().parent)
if UTIL_DIR not in sys.path:
    sys.path.insert(0, UTIL_DIR)

import util_features


feature_types = [
    "AAC", "CKSAAP type 1", "CKSAAP type 2", "DPC type 1", "DPC type 2",
    "TPC type 1", "TPC type 2",
    "CTDC", "CTDT", "CTDD", "CTriad", "KSCTriad", "ASDC", "DistancePair", "GAAC",
    "CKSAAGP type 1", "CKSAAGP type 2", "GDPC type 1", "GDPC type 2",
    "GTPC type 1", "GTPC type 2",
    "Moran", "Geary", "NMBroto", "AC", "CC", "ACC", "SOCNumber", "QSOrder",
    "PAAC", "APAAC",
    "PseKRAAC type 1", "PseKRAAC type 2", "PseKRAAC type 3A", "PseKRAAC type 3B",
    "PseKRAAC type 4", "PseKRAAC type 5",
    "PseKRAAC type 6A", "PseKRAAC type 6B", "PseKRAAC type 6C",
    "PseKRAAC type 7", "PseKRAAC type 8", "PseKRAAC type 9",
    "PseKRAAC type 10", "PseKRAAC type 11", "PseKRAAC type 12",
    "PseKRAAC type 13", "PseKRAAC type 14", "PseKRAAC type 15", "PseKRAAC type 16",
    "binary", "binary_6bit",
    "binary_5bit type 1", "binary_5bit type 2",
    "binary_3bit type 1", "binary_3bit type 2", "binary_3bit type 3",
    "binary_3bit type 4", "binary_3bit type 5", "binary_3bit type 6",
    "binary_3bit type 7",
    "AESNN3",
    "OPF_10bit",
    "OPF_7bit type 1", "OPF_7bit type 2", "OPF_7bit type 3",
    "AAIndex", "BLOSUM62", "ZScale", "KNN"
]


def clean_feature_index_to_protein_id(index_value):
    s = str(index_value).strip()

    if s.startswith(">"):
        s = s[1:]


    s = s.split()[0]
    s = s.split("|")[0]

    return s.strip()


def load_split_metadata(split_csv, split_name):
    split_df = pd.read_csv(split_csv)

    required = ["protein_id", "label", "split"]
    missing = [c for c in required if c not in split_df.columns]
    if missing:
        raise ValueError(f"Split CSV missing columns: {missing}")

    split_df["protein_id"] = split_df["protein_id"].astype(str)
    split_df["label"] = split_df["label"].astype(str)
    split_df["split"] = split_df["split"].astype(str)

    split_part = split_df[split_df["split"] == split_name].copy()

    if split_part.empty:
        raise ValueError(f"No rows found for split = {split_name}")

    print("\n[Split metadata]")
    print("Split name:", split_name)
    print("Rows:", split_part.shape[0])
    print(split_part["label"].value_counts().sort_index())

    keep_cols = ["protein_id", "label", "split"]
    if "original_id" in split_part.columns:
        keep_cols.append("original_id")
    if "original_header" in split_part.columns:
        keep_cols.append("original_header")

    return split_part[keep_cols]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument(
        "--split_csv",
        default=str(PROJECT_ROOT / "data/splits/subcell/subcell_lego11_random18097_train_test_split.csv")
    )
    parser.add_argument(
        "--out_dir",
        default=str(PROJECT_ROOT / "runs/ifeature_outputs_subcell_random18097_separate")
    )
    parser.add_argument(
        "--prefix",
        default=None
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8
    )

    args = parser.parse_args()

    fasta_path = os.environ.get("IFEATURE_FASTA", None)

    if fasta_path is None:
        raise ValueError("Please set IFEATURE_FASTA before running this script.")

    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    split_name = args.split

    prefix = args.prefix
    if prefix is None:
        prefix = f"subcell_random18097_{split_name}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"{prefix}_features_with_labels.csv"
    out_pkl = out_dir / f"{prefix}_features_with_labels.pkl"
    error_csv = out_dir / f"{prefix}_failed_features.csv"

    print("=" * 80)
    print("iFeature extraction started")
    print("=" * 80)
    print("Split:", split_name)
    print("FASTA:", fasta_path)
    print("Split CSV:", args.split_csv)
    print("Output CSV:", out_csv)

    split_meta = load_split_metadata(args.split_csv, split_name)

    ctx = get_context("spawn")

    results_dict = {}
    errors = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers,
        mp_context=ctx
    ) as executor:

        for ft, feature_df, err in executor.map(util_features.process_feature_type, feature_types):
            results_dict[ft] = feature_df

            if err:
                errors.append({
                    "feature_type": ft,
                    "error": str(err)
                })

    ok_cnt = sum(1 for value in results_dict.values() if hasattr(value, "shape"))
    fail_cnt = len(feature_types) - ok_cnt

    print("\n" + "=" * 80)
    print("[Feature extraction summary]")
    print("=" * 80)
    print("Success:", ok_cnt)
    print("Failed:", fail_cnt)

    if errors:
        error_df = pd.DataFrame(errors)
        error_df.to_csv(error_csv, index=False)

        print("\nFailed feature types saved to:")
        print(error_csv)

        for item in errors:
            print(f"- {item['feature_type']}: {item['error'].splitlines()[0]}")

    valid_dfs = []

    for ft, feature_df in results_dict.items():
        if hasattr(feature_df, "shape"):
            print(f"[OK] {ft}: {feature_df.shape}")
            valid_dfs.append(feature_df)
        else:
            print(f"[FAILED] {ft}")

    if not valid_dfs:
        raise ValueError("No valid feature tables generated.")

    merged_features = pd.concat(valid_dfs, axis=1)

    print("\n[Merged features]")
    print("Shape:", merged_features.shape)
    print("Index examples:")
    print(list(merged_features.index[:10]))

    merged_features = merged_features.copy()
    merged_features.insert(
        0,
        "protein_id",
        [clean_feature_index_to_protein_id(idx) for idx in merged_features.index]
    )

    final_df = merged_features.merge(
        split_meta,
        on="protein_id",
        how="inner"
    )

    print("\n[Merge check]")
    print("Feature rows:", merged_features.shape[0])
    print("Metadata rows:", split_meta.shape[0])
    print("Merged rows:", final_df.shape[0])

    if final_df.shape[0] != split_meta.shape[0]:
        missing_in_features = set(split_meta["protein_id"]) - set(merged_features["protein_id"])
        missing_in_meta = set(merged_features["protein_id"]) - set(split_meta["protein_id"])

        print("IDs in metadata but not features:", len(missing_in_features))
        print("IDs in features but not metadata:", len(missing_in_meta))
        print("Example missing in features:", list(missing_in_features)[:20])
        print("Example missing in metadata:", list(missing_in_meta)[:20])

        raise ValueError("ID merge failed. Stop here.")

    meta_cols = ["protein_id", "label", "split"]
    for extra_col in ["original_id", "original_header"]:
        if extra_col in final_df.columns:
            meta_cols.append(extra_col)

    feature_cols = [c for c in final_df.columns if c not in meta_cols]
    final_df = final_df[meta_cols + feature_cols]

    print("\n[Final data]")
    print(final_df.shape)
    print(final_df["label"].value_counts().sort_index())

    final_df.to_csv(out_csv, index=False)

    with open(out_pkl, "wb") as f:
        pickle.dump(final_df, f)

    print("\nSaved CSV:")
    print(out_csv)

    print("Saved PKL:")
    print(out_pkl)

    print("\nDone.")


if __name__ == "__main__":
    main()
