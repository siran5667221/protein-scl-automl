import argparse
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Evaluate Reference homology-aware Auto-sklearn workflows.')
    parser.add_argument("--test-csv", type=Path, default=project_root / "data/reference/homology_aware/Test_Features_NoHomology.csv")
    parser.add_argument("--model-dir", type=Path, default=project_root / "runs/protein_rfe_autosklearn_nohomology")
    parser.add_argument("--output-dir", type=Path, default=project_root / "runs/protein_rfe_autosklearn_nohomology/per_location_evaluation")
    args = parser.parse_args()
    if not args.test_csv.is_file():
        parser.error(f"Test CSV not found: {args.test_csv}")


    import os
    import pickle
    import numpy as np
    import pandas as pd
    from scipy import stats

    from sklearn.metrics import (
        confusion_matrix,
        matthews_corrcoef,
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
        balanced_accuracy_score
    )


    ROOT_SAVE_DIR = str(args.model_dir)


    TEST_FILE = str(args.test_csv)

    REPEATS     = 10
    OUTER_FOLDS = 10

    ID_COL    = "ProteinID"
    LABEL_COL = "label"

    LABEL_MAPPING = {
        'Cell.membrane': 0,
        'Cytoplasm': 1,
        'Cytoplasm-Nucleus': 2,
        'Endoplasmic.reticulum': 3,
        'Extracellular': 4,
        'Golgi.apparatus': 5,
        'Lysosome/Vacuole': 6,
        'Mitochondrion': 7,
        'Nucleus': 8,
        'Peroxisome': 9,
        'Plastid': 10
    }

    LABEL_NAMES = {
        0: 'Cell.membrane',
        1: 'Cytoplasm',
        2: 'Cytoplasm-Nucleus',
        3: 'Endoplasmic.reticulum',
        4: 'Extracellular',
        5: 'Golgi.apparatus',
        6: 'Lysosome/Vacuole',
        7: 'Mitochondrion',
        8: 'Nucleus',
        9: 'Peroxisome',
        10: 'Plastid'
    }

    CLASSES = list(LABEL_NAMES.keys())


    OUTPUT_DIR = str(args.output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    print("=" * 80)
    print("Loading test dataset")
    print("=" * 80)
    print(f"TEST_FILE: {TEST_FILE}")

    test_df = pd.read_csv(TEST_FILE)


    if LABEL_COL not in test_df.columns:
        raise ValueError(f"Cannot find label column: {LABEL_COL}")

    y_test_full = test_df[LABEL_COL].values


    if not np.issubdtype(y_test_full.dtype, np.number):
        print("String labels detected. Mapping labels to numeric IDs...")
        y_test_full = pd.Series(y_test_full).map(LABEL_MAPPING).values

        if pd.isna(y_test_full).any():
            bad_labels = test_df.loc[pd.isna(pd.Series(test_df[LABEL_COL]).map(LABEL_MAPPING)), LABEL_COL].unique()
            raise ValueError(f"Some labels cannot be mapped: {bad_labels}")

        y_test_full = y_test_full.astype(int)


    drop_cols = [c for c in [ID_COL, LABEL_COL] if c in test_df.columns]
    X_test_full = test_df.drop(columns=drop_cols)


    X_test_full = X_test_full.select_dtypes(include=[np.number])

    print(f"Test feature matrix: {X_test_full.shape}")
    print(f"Test labels: {len(y_test_full)}")
    print("Class distribution:")
    print(pd.Series(y_test_full).value_counts().sort_index().rename(index=LABEL_NAMES))
    print("=" * 80)


    def per_location_metrics(y_true, y_pred, classes):


        records = []

        for c in classes:
            y_true_bin = (y_true == c).astype(int)
            y_pred_bin = (y_pred == c).astype(int)


            tn, fp, fn, tp = confusion_matrix(
                y_true_bin,
                y_pred_bin,
                labels=[0, 1]
            ).ravel()

            mcc = matthews_corrcoef(y_true_bin, y_pred_bin)

            precision = precision_score(
                y_true_bin,
                y_pred_bin,
                zero_division=0
            )

            recall = recall_score(
                y_true_bin,
                y_pred_bin,
                zero_division=0
            )

            f1 = f1_score(
                y_true_bin,
                y_pred_bin,
                zero_division=0
            )

            records.append({
                "class_id": c,
                "location": LABEL_NAMES[c],
                "MCC": round(float(mcc), 4),
                "Precision": round(float(precision), 4),
                "Recall": round(float(recall), 4),
                "F1": round(float(f1), 4),
                "Support": int(y_true_bin.sum()),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn)
            })

        return pd.DataFrame(records)


    def overall_metrics(y_true, y_pred):


        return {
            "MCC": matthews_corrcoef(y_true, y_pred),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
            "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "Precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0)
        }


    def align_features_to_pipeline(X_test, pipe):


        if hasattr(pipe["scaler"], "feature_names_in_"):
            train_feature_order = list(pipe["scaler"].feature_names_in_)

            missing = [c for c in train_feature_order if c not in X_test.columns]
            extra = [c for c in X_test.columns if c not in train_feature_order]

            if len(missing) > 0:
                raise ValueError(
                    f"Test set is missing {len(missing)} training features. "
                    f"Examples: {missing[:10]}"
                )

            if len(extra) > 0:

                X_test = X_test[train_feature_order]
            else:
                X_test = X_test[train_feature_order]

        return X_test


    print("\n" + "=" * 80)
    print(f"Start evaluating {REPEATS * OUTER_FOLDS} pipelines")
    print("=" * 80)

    all_location_records = []
    all_overall_records = []
    failed_records = []

    for run_id in range(1, REPEATS + 1):
        for fold_idx in range(1, OUTER_FOLDS + 1):

            run_fold_name = f"R{run_id}_F{fold_idx}"

            pkl_path = os.path.join(
                ROOT_SAVE_DIR,
                f"run_{run_id:02d}",
                f"fold_{fold_idx:02d}",
                "complete_pipeline.pkl"
            )

            if not os.path.exists(pkl_path):
                print(f"SKIP {run_fold_name}: pipeline file not found")
                failed_records.append({
                    "Run_Fold": run_fold_name,
                    "Reason": "File not found",
                    "Path": pkl_path
                })
                continue

            try:
                with open(pkl_path, "rb") as f:
                    pipe = pickle.load(f)


                X_test_aligned = align_features_to_pipeline(X_test_full.copy(), pipe)


                X_test_scaled = pipe["scaler"].transform(X_test_aligned)
                X_test_selected = pipe["rfe"].transform(X_test_scaled)

                y_pred = pipe["model"].predict(X_test_selected)


                om = overall_metrics(y_test_full, y_pred)
                om["Run_Fold"] = run_fold_name
                om["run_id"] = run_id
                om["fold_id"] = fold_idx
                all_overall_records.append(om)


                loc_df = per_location_metrics(y_test_full, y_pred, CLASSES)
                loc_df["Run_Fold"] = run_fold_name
                loc_df["run_id"] = run_id
                loc_df["fold_id"] = fold_idx

                all_location_records.append(loc_df)

                print(
                    f"OK {run_fold_name} | "
                    f"MCC={om['MCC']:.4f} | "
                    f"F1_weighted={om['F1_weighted']:.4f}"
                )

            except Exception as e:
                print(f"ERR {run_fold_name}: {e}")
                failed_records.append({
                    "Run_Fold": run_fold_name,
                    "Reason": str(e),
                    "Path": pkl_path
                })


    overall_df = pd.DataFrame(all_overall_records)

    print("\n" + "=" * 80)
    print("Overall pipeline performance summary")
    print("=" * 80)

    if len(overall_df) > 0:
        summary_rows = []

        metric_cols = [
            "MCC",
            "Accuracy",
            "Balanced_Accuracy",
            "F1_weighted",
            "F1_macro",
            "Precision_weighted",
            "Recall_weighted"
        ]

        for metric in metric_cols:
            values = overall_df[metric].dropna()

            if len(values) == 0:
                continue

            mean = values.mean()
            std = values.std()

            if len(values) > 1:
                ci_low, ci_high = stats.t.interval(
                    confidence=0.95,
                    df=len(values) - 1,
                    loc=mean,
                    scale=stats.sem(values)
                )
            else:
                ci_low, ci_high = np.nan, np.nan

            summary_rows.append({
                "Metric": metric,
                "Mean": round(mean, 4),
                "Std": round(std, 4),
                "Min": round(values.min(), 4),
                "Max": round(values.max(), 4),
                "95CI_low": round(ci_low, 4) if not np.isnan(ci_low) else np.nan,
                "95CI_high": round(ci_high, 4) if not np.isnan(ci_high) else np.nan
            })

        overall_summary_df = pd.DataFrame(summary_rows)
        print(overall_summary_df.to_string(index=False))

    else:
        overall_summary_df = pd.DataFrame()
        print("No successful pipeline evaluation.")


    print("\n" + "=" * 80)
    print("Per-location performance summary across pipelines")
    print("=" * 80)

    if len(all_location_records) > 0:
        location_all_df = pd.concat(all_location_records, ignore_index=True)

        location_summary_df = (
            location_all_df
            .groupby(["class_id", "location"])
            .agg(
                MCC_mean=("MCC", "mean"),
                MCC_std=("MCC", "std"),
                MCC_min=("MCC", "min"),
                MCC_max=("MCC", "max"),

                F1_mean=("F1", "mean"),
                F1_std=("F1", "std"),
                F1_min=("F1", "min"),
                F1_max=("F1", "max"),

                Precision_mean=("Precision", "mean"),
                Precision_std=("Precision", "std"),

                Recall_mean=("Recall", "mean"),
                Recall_std=("Recall", "std"),

                Support=("Support", "first")
            )
            .reset_index()
            .round(4)
            .sort_values("MCC_mean", ascending=False)
        )

        print(location_summary_df.to_string(index=False))

    else:
        location_all_df = pd.DataFrame()
        location_summary_df = pd.DataFrame()
        print("No per-location result generated.")


    overall_path = os.path.join(OUTPUT_DIR, "overall_pipeline_metrics.csv")
    overall_summary_path = os.path.join(OUTPUT_DIR, "overall_pipeline_metrics_summary.csv")
    location_all_path = os.path.join(OUTPUT_DIR, "per_location_metrics_all_pipelines.csv")
    location_summary_path = os.path.join(OUTPUT_DIR, "per_location_metrics_summary.csv")
    failed_path = os.path.join(OUTPUT_DIR, "failed_pipelines.csv")

    overall_df.to_csv(overall_path, index=False)
    overall_summary_df.to_csv(overall_summary_path, index=False)
    location_all_df.to_csv(location_all_path, index=False)
    location_summary_df.to_csv(location_summary_path, index=False)

    failed_df = pd.DataFrame(failed_records)
    failed_df.to_csv(failed_path, index=False)

    print("\n" + "=" * 80)
    print("Saved files")
    print("=" * 80)
    print(f"Overall metrics:             {overall_path}")
    print(f"Overall metrics summary:     {overall_summary_path}")
    print(f"Per-location all pipelines:  {location_all_path}")
    print(f"Per-location summary:        {location_summary_path}")
    print(f"Failed pipelines:            {failed_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
