import argparse
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Reproduce the original SubCELL test-partition selection from saved GraphPart output.')
    parser.add_argument("--graphpart-csv", type=Path, default=project_root / "data/splits/subcell/subcell_lego11_unique_vs_old_graphpart_30id_10partitions.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "runs/subcell_homology_split")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import numpy as np
    import itertools

    graphpart_csv = args.graphpart_csv

    df = pd.read_csv(graphpart_csv)

    df = df.rename(columns={
        "AC": "protein_id",
        "label-val": "label",
        "cluster": "partition"
    })

    df["protein_id"] = df["protein_id"].astype(str)
    df["label"] = df["label"].astype(str)
    df["partition"] = df["partition"].astype(int)

    print(df.head())
    print(df.shape)
    print(df["partition"].value_counts().sort_index())
    print(df["label"].value_counts().sort_index())

    partitions = sorted(df["partition"].unique())

    overall_dist = df["label"].value_counts(normalize=True).sort_index()
    total_n = len(df)

    results = []

    for test_parts in itertools.combinations(partitions, 3):
        test_tmp = df[df["partition"].isin(test_parts)]
        train_tmp = df[~df["partition"].isin(test_parts)]

        test_ratio = len(test_tmp) / total_n

        test_dist = (
            test_tmp["label"]
            .value_counts(normalize=True)
            .reindex(overall_dist.index, fill_value=0)
            .sort_index()
        )

        label_diff = np.abs(test_dist - overall_dist).sum()
        size_diff = abs(test_ratio - 0.30)

        score = label_diff + size_diff

        results.append({
            "test_partitions": test_parts,
            "train_size": len(train_tmp),
            "test_size": len(test_tmp),
            "test_ratio": test_ratio,
            "label_distribution_diff": label_diff,
            "size_diff": size_diff,
            "score": score
        })

    split_candidates = pd.DataFrame(results).sort_values("score").reset_index(drop=True)

    print(split_candidates.head(10))

    best_test_partitions = list(split_candidates.loc[0, "test_partitions"])

    print("Selected test partitions:", best_test_partitions)

    df["split"] = np.where(
        df["partition"].isin(best_test_partitions),
        "test",
        "train"
    )

    print(df["split"].value_counts())
    print(df["split"].value_counts(normalize=True))

    print("Label count by split:")
    print(pd.crosstab(df["split"], df["label"]))

    print("Label percentage by split:")
    print(pd.crosstab(df["split"], df["label"], normalize="index"))

    output_split_csv = str(args.output_dir / "subcell_lego11_unique_vs_old_graphpart_train_test_split.csv")

    df.to_csv(output_split_csv, index=False)

    print("Saved split file:")
    print(output_split_csv)

    train_split_df = df[df["split"] == "train"].copy()
    test_split_df = df[df["split"] == "test"].copy()

    train_split_df.to_csv(
        str(args.output_dir / "subcell_lego11_unique_vs_old_graphpart_train_split.csv"),
        index=False
    )

    test_split_df.to_csv(
        str(args.output_dir / "subcell_lego11_unique_vs_old_graphpart_test_split.csv"),
        index=False
    )

    print("Saved train:", train_split_df.shape)
    print("Saved test:", test_split_df.shape)

if __name__ == "__main__":
    main()
