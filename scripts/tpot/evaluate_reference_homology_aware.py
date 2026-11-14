"""Evaluate Reference homology-aware TPOT workflows.
See docs/reference_homology_aware_tpot.md for source and cleanup details.
"""

import argparse
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Evaluate Reference homology-aware TPOT workflows.')
    parser.add_argument("--train-csv", type=Path, default=project_root / "data/reference/homology_aware/Train_Features_NoHomology.csv")
    parser.add_argument("--test-csv", type=Path, default=project_root / "data/reference/homology_aware/Test_Features_NoHomology.csv")
    parser.add_argument("--model-dir", type=Path, default=project_root / "runs/protein_tpot_rfe_NoHomology_Final")
    parser.add_argument("--output-dir", type=Path, default=project_root / "runs/protein_tpot_rfe_NoHomology_Final/tpot_test_per_location_results")
    args = parser.parse_args()
    for path in (args.train_csv, args.test_csv):
        if not path.is_file():
            parser.error(f"Input CSV not found: {path}")

    import os
    import glob
    import pickle
    import warnings
    import numpy as np
    import pandas as pd

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
        average_precision_score
    )
    from sklearn.preprocessing import label_binarize

    warnings.filterwarnings("ignore")

    # ============================================================
    # 1. CONFIG
    # ============================================================

    ROOT_SAVE_DIR = str(args.model_dir)      # your TPOT saved pipeline folder
    TRAIN_FILE = str(args.train_csv)     # same training file used before
    TEST_FILE = str(args.test_csv)       # change this to your test file name

    OUTPUT_DIR = str(args.output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ID_COLS = ["ProteinID"]
    LABEL_COL = "label"

    # label mapping: must be same as your training code
    label_mapping = {
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

    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    ALL_CLASSES = np.array(sorted(inverse_label_mapping.keys()))


    # ============================================================
    # 2. Helper functions
    # ============================================================

    def convert_labels(y):
        """
        Convert string labels to numeric labels if needed.
        如果 label 是文字，就转成数字；如果已经是数字，就直接转成 int。
        """
        y_series = pd.Series(y)

        if not np.issubdtype(y_series.dtype, np.number):
            y_num = y_series.map(label_mapping)

            if y_num.isna().any():
                bad_labels = y_series[y_num.isna()].unique()
                raise ValueError(f"Unknown labels found: {bad_labels}")

            return y_num.values.astype(int)

        return y_series.values.astype(int)


    def get_pipeline_id(pkl_path):
        """
        Extract run and fold ID from path like:
        ./protein_tpot_nohomology/run_01/fold_03/complete_pipeline.pkl
        """
        parts = pkl_path.replace("\\", "/").split("/")
        run_id = None
        fold_id = None

        for p in parts:
            if p.startswith("run_"):
                run_id = int(p.replace("run_", ""))
            if p.startswith("fold_"):
                fold_id = int(p.replace("fold_", ""))

        pipeline_id = f"R{run_id}_F{fold_id}" if run_id is not None and fold_id is not None else os.path.basename(pkl_path)
        return run_id, fold_id, pipeline_id


    def safe_predict_proba(model, X):
        """
        Some TPOT pipelines may not support predict_proba.
        有些 TPOT pipeline 没有 predict_proba，所以这里安全处理。
        """
        if hasattr(model, "predict_proba"):
            try:
                return model.predict_proba(X)
            except Exception:
                return None
        return None


    def align_proba_to_all_classes(y_proba, model_classes, all_classes):
        """
        Align probability columns to global class order.
        把 predict_proba 的列顺序对齐到 0-10 的全局类别顺序。
        """
        if y_proba is None:
            return None

        proba_full = np.zeros((y_proba.shape[0], len(all_classes)))

        model_classes = np.array(model_classes).astype(int)

        for j, cls in enumerate(model_classes):
            if cls in all_classes:
                target_col = np.where(all_classes == cls)[0][0]
                proba_full[:, target_col] = y_proba[:, j]

        return proba_full


    def compute_overall_metrics(y_true, y_pred, y_proba=None, classes=ALL_CLASSES):
        """
        Overall multi-class test metrics.
        整体多分类指标。
        """
        m = {
            "MCC": matthews_corrcoef(y_true, y_pred),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
            "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "Precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        if y_proba is not None:
            try:
                y_true_bin = label_binarize(y_true, classes=classes)
                m["ROC_AUC_macro"] = roc_auc_score(
                    y_true_bin,
                    y_proba,
                    average="macro",
                    multi_class="ovr"
                )
                m["PR_AUC_macro"] = average_precision_score(
                    y_true_bin,
                    y_proba,
                    average="macro"
                )
            except Exception as e:
                m["ROC_AUC_macro"] = np.nan
                m["PR_AUC_macro"] = np.nan
                m["AUC_Error"] = str(e)
        else:
            m["ROC_AUC_macro"] = np.nan
            m["PR_AUC_macro"] = np.nan

        return m


    def compute_per_location_metrics(y_true, y_pred, classes=ALL_CLASSES):
        """
        Per-location one-vs-rest metrics.
        每个 location 单独看成 one-vs-rest，计算该位置自己的 MCC / F1 / precision / recall。
        """
        records = []

        for cls in classes:
            y_true_bin = (y_true == cls).astype(int)
            y_pred_bin = (y_pred == cls).astype(int)

            support = int(y_true_bin.sum())
            pred_support = int(y_pred_bin.sum())

            record = {
                "class_id": int(cls),
                "location": inverse_label_mapping.get(int(cls), str(cls)),
                "support": support,
                "pred_support": pred_support,
                "MCC": matthews_corrcoef(y_true_bin, y_pred_bin),
                "Precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
                "Recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
                "F1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
            }

            records.append(record)

        return pd.DataFrame(records)


    # ============================================================
    # 3. Load train/test feature structure
    # ============================================================

    print("=" * 80)
    print("Loading train/test data")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    drop_cols = [c for c in ID_COLS + [LABEL_COL] if c in train_df.columns]
    feature_cols = train_df.drop(columns=drop_cols).columns.tolist()

    missing_features = [c for c in feature_cols if c not in test_df.columns]
    extra_features = [c for c in test_df.columns if c not in feature_cols + ID_COLS + [LABEL_COL]]

    if len(missing_features) > 0:
        raise ValueError(f"Test file is missing {len(missing_features)} training features. Example: {missing_features[:10]}")

    print(f"Number of training features: {len(feature_cols)}")
    print(f"Number of test samples: {test_df.shape[0]}")
    print(f"Extra columns in test file ignored: {len(extra_features)}")

    X_test = test_df[feature_cols]
    y_test = convert_labels(test_df[LABEL_COL].values)

    protein_ids = test_df["ProteinID"].values if "ProteinID" in test_df.columns else np.arange(len(test_df))

    print("\nTest label distribution:")
    print(pd.Series(y_test).map(inverse_label_mapping).value_counts())


    # ============================================================
    # 4. Find all saved TPOT pipelines
    # ============================================================

    pkl_files = sorted(
        glob.glob(os.path.join(ROOT_SAVE_DIR, "run_*", "fold_*", "complete_pipeline.pkl"))
    )

    print("\n" + "=" * 80)
    print(f"Found {len(pkl_files)} saved TPOT pipelines")
    print("=" * 80)

    if len(pkl_files) == 0:
        raise FileNotFoundError(
            f"No complete_pipeline.pkl files found under: {ROOT_SAVE_DIR}/run_*/fold_*/"
        )


    # ============================================================
    # 5. Evaluate each pipeline on test set
    # ============================================================

    overall_records = []
    per_location_records = []
    prediction_records = []

    for pkl_path in pkl_files:
        run_id, fold_id, pipeline_id = get_pipeline_id(pkl_path)

        try:
            with open(pkl_path, "rb") as f:
                pipe_obj = pickle.load(f)

            scaler = pipe_obj["scaler"]
            rfe = pipe_obj["rfe"]
            model = pipe_obj["model"]
            selected_features = pipe_obj.get("selected_features", None)
            val_metrics = pipe_obj.get("val_metrics", {})

            # Apply the exact same preprocessing as training
            X_test_scaled = scaler.transform(X_test)
            X_test_selected = rfe.transform(X_test_scaled)

            y_pred = model.predict(X_test_selected)

            y_proba_raw = safe_predict_proba(model, X_test_selected)

            model_classes = getattr(model, "classes_", ALL_CLASSES)
            y_proba = align_proba_to_all_classes(y_proba_raw, model_classes, ALL_CLASSES)

            # Overall metrics
            overall_m = compute_overall_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                classes=ALL_CLASSES
            )

            row = {
                "pipeline_id": pipeline_id,
                "run_id": run_id,
                "fold_id": fold_id,
                "pkl_path": pkl_path,
                "val_MCC": val_metrics.get("MCC", np.nan),
                "n_selected_features": len(selected_features) if selected_features is not None else np.nan,
            }
            row.update(overall_m)
            overall_records.append(row)

            # Per-location metrics
            per_loc_df = compute_per_location_metrics(y_test, y_pred, classes=ALL_CLASSES)
            per_loc_df.insert(0, "pipeline_id", pipeline_id)
            per_loc_df.insert(1, "run_id", run_id)
            per_loc_df.insert(2, "fold_id", fold_id)
            per_location_records.append(per_loc_df)

            # Save sample-level predictions
            tmp_pred_df = pd.DataFrame({
                "pipeline_id": pipeline_id,
                "run_id": run_id,
                "fold_id": fold_id,
                "ProteinID": protein_ids,
                "true_label_id": y_test,
                "true_location": pd.Series(y_test).map(inverse_label_mapping).values,
                "pred_label_id": y_pred,
                "pred_location": pd.Series(y_pred).map(inverse_label_mapping).values,
                "correct": y_test == y_pred
            })
            prediction_records.append(tmp_pred_df)

            print(f"OK {pipeline_id} | Test MCC={overall_m['MCC']:.4f} | F1_weighted={overall_m['F1_weighted']:.4f}")

        except Exception as e:
            print(f"FAILED {pipeline_id} | {pkl_path}")
            print(f"Reason: {e}")

            overall_records.append({
                "pipeline_id": pipeline_id,
                "run_id": run_id,
                "fold_id": fold_id,
                "pkl_path": pkl_path,
                "error": str(e)
            })


    # ============================================================
    # 6. Save full result tables
    # ============================================================

    overall_df = pd.DataFrame(overall_records)
    overall_df = overall_df.sort_values("MCC", ascending=False, na_position="last")

    per_location_all_df = pd.concat(per_location_records, ignore_index=True)
    predictions_all_df = pd.concat(prediction_records, ignore_index=True)

    overall_path = os.path.join(OUTPUT_DIR, "tpot_test_all_pipelines_summary.csv")
    per_location_all_path = os.path.join(OUTPUT_DIR, "tpot_test_per_location_all_pipelines_long.csv")
    predictions_path = os.path.join(OUTPUT_DIR, "tpot_test_predictions_all_pipelines.csv")

    overall_df.to_csv(overall_path, index=False)
    per_location_all_df.to_csv(per_location_all_path, index=False)
    predictions_all_df.to_csv(predictions_path, index=False)

    print("\nSaved:")
    print(overall_path)
    print(per_location_all_path)
    print(predictions_path)


    # ============================================================
    # 7. Mean per-location metrics across all pipelines
    # ============================================================

    per_location_mean_df = (
        per_location_all_df
        .groupby(["class_id", "location"], as_index=False)
        .agg(
            support=("support", "first"),
            mean_pred_support=("pred_support", "mean"),
            mean_MCC=("MCC", "mean"),
            sd_MCC=("MCC", "std"),
            mean_Precision=("Precision", "mean"),
            sd_Precision=("Precision", "std"),
            mean_Recall=("Recall", "mean"),
            sd_Recall=("Recall", "std"),
            mean_F1=("F1", "mean"),
            sd_F1=("F1", "std"),
            n_pipelines=("pipeline_id", "nunique")
        )
        .sort_values("mean_MCC", ascending=False)
    )

    per_location_mean_path = os.path.join(OUTPUT_DIR, "tpot_test_per_location_mean_across_pipelines.csv")
    per_location_mean_df.to_csv(per_location_mean_path, index=False)

    print("\n" + "=" * 80)
    print("Mean per-location performance across all TPOT pipelines")
    print("=" * 80)
    print(per_location_mean_df)


    # ============================================================
    # 8. Best pipeline by test MCC
    # ============================================================

    valid_overall_df = overall_df.dropna(subset=["MCC"]).copy()

    if valid_overall_df.empty:
        raise ValueError("No valid pipelines were successfully evaluated.")

    best_row = valid_overall_df.iloc[0]
    best_pipeline_id = best_row["pipeline_id"]

    best_per_location_df = (
        per_location_all_df[per_location_all_df["pipeline_id"] == best_pipeline_id]
        .sort_values("MCC", ascending=False)
    )

    best_per_location_path = os.path.join(OUTPUT_DIR, "tpot_test_best_pipeline_per_location_metrics.csv")
    best_per_location_df.to_csv(best_per_location_path, index=False)

    best_predictions_df = predictions_all_df[predictions_all_df["pipeline_id"] == best_pipeline_id].copy()
    best_predictions_path = os.path.join(OUTPUT_DIR, "tpot_test_best_pipeline_predictions.csv")
    best_predictions_df.to_csv(best_predictions_path, index=False)

    print("\n" + "=" * 80)
    print("Best TPOT pipeline on test set")
    print("=" * 80)
    print(f"Best pipeline: {best_pipeline_id}")
    print(f"Run: {best_row['run_id']}, Fold: {best_row['fold_id']}")
    print(f"Test MCC: {best_row['MCC']:.4f}")
    print(f"Test Accuracy: {best_row['Accuracy']:.4f}")
    print(f"Test Balanced Accuracy: {best_row['Balanced_Acc']:.4f}")
    print(f"Test F1 weighted: {best_row['F1_weighted']:.4f}")
    print(f"Validation MCC before test: {best_row['val_MCC']:.4f}")
    print(f"Path: {best_row['pkl_path']}")

    print("\nBest pipeline per-location metrics:")
    print(best_per_location_df)

    print("\nSaved:")
    print(per_location_mean_path)
    print(best_per_location_path)
    print(best_predictions_path)


    # ============================================================
    # 9. Optional: compare validation MCC vs test MCC
    # ============================================================

    compare_path = os.path.join(OUTPUT_DIR, "tpot_val_vs_test_mcc.csv")

    compare_df = valid_overall_df[[
        "pipeline_id",
        "run_id",
        "fold_id",
        "val_MCC",
        "MCC",
        "Accuracy",
        "Balanced_Acc",
        "F1_weighted",
        "pkl_path"
    ]].rename(columns={"MCC": "test_MCC"})

    compare_df.to_csv(compare_path, index=False)

    print("\nValidation MCC vs Test MCC table saved:")
    print(compare_path)

if __name__ == "__main__":
    main()
