import argparse
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Reproduce the original SubCELL random split.')
    parser.add_argument("--input-fasta", type=Path, default=project_root / "data/subcell/new_only/subcell_lego11_unique_sequences_vs_old.fasta")
    parser.add_argument("--output-dir", type=Path, default=project_root / "runs/subcell_random_split")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import random


    input_fasta = args.input_fasta


    print("Input FASTA exists:", input_fasta.exists())
    print("Input FASTA path:", input_fasta.resolve())

    if not input_fasta.exists():
        raise FileNotFoundError(f"Cannot find input FASTA: {input_fasta}")


    def clean_seq(seq):
        return (
            str(seq)
            .replace("*", "")
            .replace("-", "")
            .replace(" ", "")
            .replace("\n", "")
            .upper()
            .strip()
        )


    def read_fasta(fasta_path):
        records = []
        header = None
        seq_lines = []

        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.startswith(">"):
                    if header is not None:
                        records.append({
                            "header": header,
                            "sequence": clean_seq("".join(seq_lines))
                        })

                    header = line[1:].strip()
                    seq_lines = []
                else:
                    seq_lines.append(line)

        if header is not None:
            records.append({
                "header": header,
                "sequence": clean_seq("".join(seq_lines))
            })

        return pd.DataFrame(records)


    def wrap_sequence(seq, width=60):
        return "\n".join(seq[i:i+width] for i in range(0, len(seq), width))


    def write_fasta(df, out_path):
        with open(out_path, "w") as f:
            for _, row in df.iterrows():
                f.write(f">{row['header']}\n")
                f.write(wrap_sequence(row["sequence"]) + "\n")


    def parse_label_from_header(header):


        header = str(header).strip()


        if "|" in header:
            return header.split("|")[-1].strip()


        parts = header.split()
        if len(parts) >= 2:
            return " ".join(parts[1:]).strip()

        return None


    df = read_fasta(input_fasta)
    df["label"] = df["header"].apply(parse_label_from_header)

    print("Total records:", len(df))
    print("Unique sequences:", df["sequence"].nunique())
    print("Unique labels:", df["label"].nunique())

    print(df.head())

    label_counts = (
        df["label"]
        .value_counts()
        .reset_index()
    )

    label_counts.columns = ["label", "total_count"]

    print(label_counts)


    random_seed = 42
    train_ratio = 0.7

    train_parts = []
    test_parts = []

    for label, group in df.groupby("label"):
        group = group.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        n = len(group)


        n_train = int(round(n * train_ratio))


        if n >= 2:
            n_train = max(1, min(n_train, n - 1))
        else:

            n_train = 1

        train_parts.append(group.iloc[:n_train])
        test_parts.append(group.iloc[n_train:])

    df_train = pd.concat(train_parts, ignore_index=True)
    df_test = pd.concat(test_parts, ignore_index=True)


    df_train = df_train.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print("Train records:", len(df_train))
    print("Test records:", len(df_test))
    print("Train ratio:", len(df_train) / len(df))
    print("Test ratio:", len(df_test) / len(df))


    train_counts = df_train["label"].value_counts().rename("train_count")
    test_counts = df_test["label"].value_counts().rename("test_count")
    total_counts = df["label"].value_counts().rename("total_count")

    split_summary = pd.concat([total_counts, train_counts, test_counts], axis=1).fillna(0)
    split_summary = split_summary.astype(int)
    split_summary["train_ratio"] = split_summary["train_count"] / split_summary["total_count"]
    split_summary["test_ratio"] = split_summary["test_count"] / split_summary["total_count"]

    split_summary = split_summary.reset_index().rename(columns={"index": "label"})

    print(split_summary)


    out_train = (args.output_dir / "subcell_lego11_unique_random70_train.fasta")
    out_test = (args.output_dir / "subcell_lego11_unique_random30_test.fasta")
    out_summary = (args.output_dir / "subcell_lego11_unique_random70_30_split_summary.csv")

    write_fasta(df_train, out_train)
    write_fasta(df_test, out_test)

    split_summary.to_csv(out_summary, index=False)

    print("Saved train FASTA:", out_train.resolve())
    print("Saved test FASTA:", out_test.resolve())
    print("Saved summary CSV:", out_summary.resolve())


    def extract_protein_id(header):


        return str(header).split("|")[0].strip()


    def clean_label(label):


        label = str(label).strip()
        label = label.replace("label=", "")
        return label


    df_train_split = df_train.copy()
    df_test_split = df_test.copy()

    df_train_split["protein_id"] = df_train_split["header"].apply(extract_protein_id)
    df_test_split["protein_id"] = df_test_split["header"].apply(extract_protein_id)

    df_train_split["label"] = df_train_split["label"].apply(clean_label)
    df_test_split["label"] = df_test_split["label"].apply(clean_label)

    df_train_split["split"] = "train"
    df_test_split["split"] = "test"


    df_train_split["partition"] = -1
    df_test_split["partition"] = -1

    random_split_df = pd.concat(
        [df_train_split, df_test_split],
        ignore_index=True
    )

    random_split_df = random_split_df[
        ["protein_id", "label", "partition", "split", "header"]
    ].copy()

    out_random_split = (args.output_dir / "subcell_lego11_random18097_train_test_split.csv")

    random_split_df.to_csv(out_random_split, index=False)

    print("Saved protein-level random split CSV:", out_random_split.resolve())
    print(random_split_df["split"].value_counts())
    print("Total:", len(random_split_df))
    print("Unique protein_id:", random_split_df["protein_id"].nunique())
    print("Duplicated protein_id:", random_split_df["protein_id"].duplicated().sum())

    train_ids = set(random_split_df.loc[random_split_df["split"] == "train", "protein_id"])
    test_ids = set(random_split_df.loc[random_split_df["split"] == "test", "protein_id"])

    print("Train/test overlap:", len(train_ids & test_ids))

    print(pd.crosstab(random_split_df["split"], random_split_df["label"]))

if __name__ == "__main__":
    main()
