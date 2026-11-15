import concurrent.futures
from multiprocessing import get_context
from pathlib import Path
import pandas as pd
import pickle
import sys
import os


UTIL_DIR = str(Path(__file__).resolve().parent)

if UTIL_DIR not in sys.path:
    sys.path.insert(0, UTIL_DIR)

print("Current working directory:", os.getcwd())
print("util_features.py exists:", os.path.exists(str(Path(UTIL_DIR) / "util_features.py")))

import util_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FASTA_FILE_PATH = os.environ.get("IFEATURE_FASTA", str(PROJECT_ROOT / "data/subcell/new_only/subcell_lego11_unique_sequences_vs_old.fasta"))

SPLIT_CSV_PATH = os.environ.get("IFEATURE_SPLIT_CSV", str(PROJECT_ROOT / "data/splits/subcell/subcell_lego11_unique_vs_old_graphpart_train_test_split.csv"))


OUTPUT_DIR = Path(os.environ.get("IFEATURE_OUTPUT_DIR", str(PROJECT_ROOT / "runs/ifeature_outputs_subcell_graphpart17100")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_PKL = OUTPUT_DIR / "subcell_graphpart17100_full_features_with_split.pkl"
FULL_CSV = OUTPUT_DIR / "subcell_graphpart17100_full_features_with_split.csv"

TRAIN_CSV = OUTPUT_DIR / "subcell_graphpart17100_train_features_with_labels.csv"
TEST_CSV = OUTPUT_DIR / "subcell_graphpart17100_test_features_with_labels.csv"

ERROR_CSV = OUTPUT_DIR / "subcell_graphpart17100_failed_features.csv"


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


def set_util_fasta_path(fasta_path):
    os.environ["IFEATURE_FASTA"] = str(fasta_path)


def worker_init(fasta_path):
    os.environ["IFEATURE_FASTA"] = str(fasta_path)


def load_split_table(split_csv_path):
    split_df = pd.read_csv(split_csv_path)

    print("\n" + "=" * 80)
    print("[Split CSV loaded]")
    print("=" * 80)
    print(split_df.head())
    print("Columns:", split_df.columns.tolist())
    print("Shape:", split_df.shape)

    rename_dict = {}

    if "AC" in split_df.columns:
        rename_dict["AC"] = "protein_id"

    if "label-val" in split_df.columns:
        rename_dict["label-val"] = "label"

    if "cluster" in split_df.columns:
        rename_dict["cluster"] = "partition"

    split_df = split_df.rename(columns=rename_dict)

    required_cols = ["protein_id", "label", "partition", "split"]

    missing_cols = [col for col in required_cols if col not in split_df.columns]

    if missing_cols:
        raise ValueError(f"Split CSV 缺少这些列: {missing_cols}")

    split_df = split_df[required_cols].copy()

    split_df["protein_id"] = split_df["protein_id"].astype(str)
    split_df["label"] = split_df["label"].astype(str)
    split_df["partition"] = split_df["partition"].astype(int)
    split_df["split"] = split_df["split"].astype(str)

    print("\n[Split size]")
    print(split_df["split"].value_counts())

    print("\n[Label count by split]")
    print(pd.crosstab(split_df["split"], split_df["label"]))

    return split_df


def clean_feature_index_to_protein_id(index_value):


    s = str(index_value).strip()

    if s.startswith(">"):
        s = s[1:]

    s = s.split()[0]
    s = s.split("|")[0]

    return s.strip()


def add_split_info_to_features(merged_features, split_df):
    merged_features = merged_features.copy()

    protein_ids = [
        clean_feature_index_to_protein_id(idx)
        for idx in merged_features.index
    ]

    merged_features.insert(0, "protein_id", protein_ids)

    print("\n" + "=" * 80)
    print("[ID check]")
    print("=" * 80)
    print("Feature ID examples:")
    print(merged_features["protein_id"].head(10).tolist())

    print("Split ID examples:")
    print(split_df["protein_id"].head(10).tolist())

    full_df = merged_features.merge(
        split_df,
        on="protein_id",
        how="inner"
    )

    print("\n" + "=" * 80)
    print("[Merge check]")
    print("=" * 80)
    print("Feature rows:", merged_features.shape[0])
    print("Split rows:", split_df.shape[0])
    print("Merged rows:", full_df.shape[0])

    if full_df.shape[0] != split_df.shape[0]:
        missing_in_features = set(split_df["protein_id"]) - set(merged_features["protein_id"])
        missing_in_split = set(merged_features["protein_id"]) - set(split_df["protein_id"])

        print("\n[WARNING] merge 后行数不等于 split 行数，说明 protein_id 没完全对上。")
        print("IDs in split but not in features:", len(missing_in_features))
        print("IDs in features but not in split:", len(missing_in_split))

        print("\nExample IDs in split but not in features:")
        print(list(missing_in_features)[:20])

        print("\nExample IDs in features but not in split:")
        print(list(missing_in_split)[:20])

        raise ValueError("ID merge failed. Stop here. Do not train models.")

    print("\n[Final split count]")
    print(full_df["split"].value_counts())

    print("\n[Final label count by split]")
    print(pd.crosstab(full_df["split"], full_df["label"]))

    meta_cols = ["protein_id", "label", "partition", "split"]
    feature_cols = [col for col in full_df.columns if col not in meta_cols]

    full_df = full_df[meta_cols + feature_cols]

    return full_df


def main(max_workers=8):
    print("=" * 80)
    print("GraphPart 17100 full iFeature extraction started")
    print("=" * 80)
    print("FASTA file:", FASTA_FILE_PATH)
    print("Split CSV:", SPLIT_CSV_PATH)
    print("Output dir:", OUTPUT_DIR)

    if not os.path.exists(FASTA_FILE_PATH):
        raise FileNotFoundError(f"FASTA file not found: {FASTA_FILE_PATH}")

    if not os.path.exists(SPLIT_CSV_PATH):
        raise FileNotFoundError(f"Split CSV not found: {SPLIT_CSV_PATH}")

    set_util_fasta_path(FASTA_FILE_PATH)

    split_df = load_split_table(SPLIT_CSV_PATH)

    print("\n" + "=" * 80)
    print("[Feature extraction]")
    print("=" * 80)
    print("开始并行计算 features...")
    print("请确认 util_features.py 实际读取的是这个 FASTA:")
    print(FASTA_FILE_PATH)

    ctx = get_context("spawn")

    results_dict = {}
    errors = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=worker_init,
        initargs=(FASTA_FILE_PATH,)
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
        error_df.to_csv(ERROR_CSV, index=False)

        print("\n[Failed feature types]")
        for item in errors:
            print(f"- {item['feature_type']}: {item['error'].splitlines()[0]}")

        print("\nFailed feature list saved to:")
        print(ERROR_CSV)

    valid_dfs = []

    for ft, feature_df in results_dict.items():
        if hasattr(feature_df, "shape"):
            print(f"[OK] {ft}: {feature_df.shape}")
            valid_dfs.append(feature_df)
        else:
            print(f"[FAILED] {ft}")

    if not valid_dfs:
        raise ValueError("没有成功计算的特征，无法生成最终文件。")

    merged_features = pd.concat(valid_dfs, axis=1)

    print("\n" + "=" * 80)
    print("[Merged features]")
    print("=" * 80)
    print("Merged feature shape:", merged_features.shape)
    print("Index examples:")
    print(list(merged_features.index[:10]))

    final_df = add_split_info_to_features(merged_features, split_df)

    return final_df


if __name__ == "__main__":

    final_df = main(max_workers=8)

    print("\n" + "=" * 80)
    print("[Saving files]")
    print("=" * 80)


    with open(FULL_PKL, "wb") as file:
        pickle.dump(final_df, file)


    final_df.to_csv(FULL_CSV, index=False)

    print("Saved full PKL:")
    print(FULL_PKL)

    print("Saved full CSV:")
    print(FULL_CSV)

    print("Full shape:")
    print(final_df.shape)


    train_df = final_df[final_df["split"] == "train"].copy()
    test_df = final_df[final_df["split"] == "test"].copy()

    print("\nTrain shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts().sort_index())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts().sort_index())


    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print("\nSaved train CSV:")
    print(TRAIN_CSV)

    print("Saved test CSV:")
    print(TEST_CSV)

    print("\nDone.")
