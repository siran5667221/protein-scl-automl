import argparse
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Evaluate the 100 saved Reference Random TPOT workflows.')
    parser.add_argument("--test-csv", type=Path, default=project_root / "data/reference/random/test_numeric_labels.csv")
    parser.add_argument("--model-dir", type=Path, default=project_root / "runs/protein_tpot_rfe_longtime_final")
    parser.add_argument("--output-dir", type=Path, default=project_root / "runs/protein_tpot_rfe_longtime_final/per_location_evaluation")
    args = parser.parse_args()
    if not args.test_csv.is_file():
        parser.error(f"Test CSV not found: {args.test_csv}")
    missing = [args.model_dir / f"run_{r:02d}/fold_{f:02d}/complete_pipeline.pkl" for r in range(1,11) for f in range(1,11) if not (args.model_dir / f"run_{r:02d}/fold_{f:02d}/complete_pipeline.pkl").is_file()]
    if missing:
        parser.error(f"Missing {len(missing)} fitted workflow files. First: {missing[0]}")
    print("The best_pipeline_result export is a retrospective test-MCC maximum, not independent model selection.")


    import os
    import json
    import pickle
    import warnings
    import logging
    import pandas as pd
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score,
        matthews_corrcoef, confusion_matrix
    )
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score

    warnings.filterwarnings('ignore')


    TPOT_ROOT_DIR = str(args.model_dir)


    TEST_CSV_PATH = str(args.test_csv)


    OUTPUT_DIR = str(args.output_dir)


    AUTOML_METHOD = "TPOT"
    SPLIT_TYPE    = "random"


    REPEATS     = 10
    OUTER_FOLDS = 10


    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    best_pipeline_dir = output_path / "best_pipeline_result"
    best_pipeline_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"TPOT Random Split — Test Set Per-Location Evaluation")
    print(f"TPOT根目录  : {TPOT_ROOT_DIR}")
    print(f"Test CSV    : {TEST_CSV_PATH}")
    print(f"输出目录    : {OUTPUT_DIR}")
    print("=" * 70)


    print("\n[1] 加载 Test Set...")
    test_df = pd.read_csv(TEST_CSV_PATH)


    y_test = test_df['label'].values
    X_test_full = test_df.drop(['ProteinID', 'label'], axis=1)

    print(f"    Test set shape : {X_test_full.shape}")
    print(f"    类别分布：\n{pd.Series(y_test).value_counts().sort_index()}")


    def compute_per_location_metrics(y_true, y_pred, class_id, location_name,
                                      run_fold_label, run_id, fold_id):


        y_true_bin = (y_true == class_id).astype(int)
        y_pred_bin = (y_pred == class_id).astype(int)

        support = int(y_true_bin.sum())
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

        mcc  = float(matthews_corrcoef(y_true_bin, y_pred_bin))
        prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
        rec  = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
        f1   = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

        return {
            'class_id'  : class_id,
            'location'  : location_name,
            'MCC'       : mcc,
            'Precision' : prec,
            'Recall'    : rec,
            'F1'        : f1,
            'Support'   : support,
            'TP'        : int(tp),
            'FP'        : int(fp),
            'FN'        : int(fn),
            'TN'        : int(tn),
            'Run_Fold'  : run_fold_label,
            'run_id'    : run_id,
            'fold_id'   : fold_id,
        }


    def compute_overall_metrics(y_true, y_pred, run_fold_label, run_id, fold_id):
        mcc      = float(matthews_corrcoef(y_true, y_pred))
        acc      = float(accuracy_score(y_true, y_pred))
        bal_acc  = float(balanced_accuracy_score(y_true, y_pred))
        f1_w     = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        f1_mac   = float(f1_score(y_true, y_pred, average='macro',    zero_division=0))
        prec_w   = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        rec_w    = float(recall_score(y_true, y_pred, average='weighted',    zero_division=0))

        return {
            'MCC'               : mcc,
            'Accuracy'          : acc,
            'Balanced_Accuracy' : bal_acc,
            'F1_weighted'       : f1_w,
            'F1_macro'          : f1_mac,
            'Precision_weighted': prec_w,
            'Recall_weighted'   : rec_w,
            'Run_Fold'          : run_fold_label,
            'run_id'            : run_id,
            'fold_id'           : fold_id,
        }


    print("\n[2] 开始遍历 pipelines...")

    overall_records     = []
    per_location_records = []
    failed_pipelines    = []


    best_mcc      = -np.inf
    best_pipeline_info = None

    for run_id in range(1, REPEATS + 1):
        for fold_id in range(1, OUTER_FOLDS + 1):

            run_fold_label = f"R{run_id}_F{fold_id}"


            pkl_path = Path(TPOT_ROOT_DIR) / f"run_{run_id:02d}" / f"fold_{fold_id:02d}" / "complete_pipeline.pkl"


            if not pkl_path.exists():
                msg = f"[MISSING] {pkl_path}"
                print(f"  ⚠ {msg}")
                failed_pipelines.append({
                    'run_id'   : run_id,
                    'fold_id'  : fold_id,
                    'Run_Fold' : run_fold_label,
                    'reason'   : 'file_not_found',
                    'path'     : str(pkl_path)
                })
                continue


            try:
                with open(pkl_path, 'rb') as f:
                    pipeline_config = pickle.load(f)
            except Exception as e:
                print(f"  ✗ 加载失败 {run_fold_label}: {e}")
                failed_pipelines.append({
                    'run_id'  : run_id,
                    'fold_id' : fold_id,
                    'Run_Fold': run_fold_label,
                    'reason'  : f'load_error: {e}',
                    'path'    : str(pkl_path)
                })
                continue


            try:
                scaler           = pipeline_config['scaler']
                rfe              = pipeline_config['rfe']
                model            = pipeline_config['model']
                selected_features = pipeline_config['selected_features']
            except KeyError as e:
                print(f"  ✗ pkl 结构异常 {run_fold_label}: 缺少key {e}")
                failed_pipelines.append({
                    'run_id'  : run_id,
                    'fold_id' : fold_id,
                    'Run_Fold': run_fold_label,
                    'reason'  : f'key_error: {e}',
                    'path'    : str(pkl_path)
                })
                continue


            try:
                X_test_scaled = scaler.transform(X_test_full)
                X_test_sel    = rfe.transform(X_test_scaled)
            except Exception as e:
                print(f"  ✗ 预处理失败 {run_fold_label}: {e}")
                failed_pipelines.append({
                    'run_id'  : run_id,
                    'fold_id' : fold_id,
                    'Run_Fold': run_fold_label,
                    'reason'  : f'preprocess_error: {e}',
                    'path'    : str(pkl_path)
                })
                continue


            try:
                y_pred = model.predict(X_test_sel)
            except Exception as e:
                print(f"  ✗ 预测失败 {run_fold_label}: {e}")
                failed_pipelines.append({
                    'run_id'  : run_id,
                    'fold_id' : fold_id,
                    'Run_Fold': run_fold_label,
                    'reason'  : f'predict_error: {e}',
                    'path'    : str(pkl_path)
                })
                continue


            overall_row = compute_overall_metrics(y_test, y_pred, run_fold_label, run_id, fold_id)
            overall_records.append(overall_row)

            test_mcc = overall_row['MCC']
            print(f"  ✓ {run_fold_label}  Test MCC: {test_mcc:.4f}")


            if test_mcc > best_mcc:
                best_mcc = test_mcc
                best_pipeline_info = {
                    'run_fold_label'   : run_fold_label,
                    'run_id'           : run_id,
                    'fold_id'          : fold_id,
                    'y_pred'           : y_pred,
                    'overall_metrics'  : overall_row,
                    'selected_features': selected_features,
                }


            LABEL_MAP = {
                0 : "Cell.membrane",
                1 : "Cytoplasm",
                2 : "Cytoplasm-Nucleus",
                3 : "Endoplasmic.reticulum",
                4 : "Extracellular",
                5 : "Golgi.apparatus",
                6 : "Lysosome/Vacuole",
                7 : "Mitochondrion",
                8 : "Nucleus",
                9 : "Peroxisome",
                10: "Plastid",
            }

            unique_classes = sorted(np.unique(y_test))
            for class_id in unique_classes:
                loc_name = LABEL_MAP.get(class_id, f"Location_{class_id}")
                row = compute_per_location_metrics(
                    y_test, y_pred, class_id, loc_name,
                    run_fold_label, run_id, fold_id
                )
                per_location_records.append(row)

    print(f"\n  完成！成功: {len(overall_records)}，失败: {len(failed_pipelines)}")


    print("\n[3] 保存 overall_pipeline_metrics.csv ...")
    if len(overall_records) != REPEATS * OUTER_FOLDS:
        pd.DataFrame(failed_pipelines).to_csv(output_path / "failed_pipelines.csv", index=False)
        raise RuntimeError(f"Expected 100 successful workflows; evaluated {len(overall_records)}. See failed_pipelines.csv.")
    overall_df = pd.DataFrame(overall_records)
    overall_df.to_csv(output_path / "overall_pipeline_metrics.csv", index=False)
    print(f"    → 已保存：{output_path / 'overall_pipeline_metrics.csv'}")


    print("[4] 保存 overall_pipeline_metrics_summary.csv ...")
    metric_cols = ['MCC', 'Accuracy', 'Balanced_Accuracy',
                   'F1_weighted', 'F1_macro', 'Precision_weighted', 'Recall_weighted']
    summary_rows = []
    for col in metric_cols:
        summary_rows.append({
            'metric' : col,
            'mean'   : overall_df[col].mean(),
            'std'    : overall_df[col].std(),
            'median' : overall_df[col].median(),
            'min'    : overall_df[col].min(),
            'max'    : overall_df[col].max(),
            'count'  : overall_df[col].count(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path / "overall_pipeline_metrics_summary.csv", index=False)
    print(f"    → 已保存：{output_path / 'overall_pipeline_metrics_summary.csv'}")


    print("[5] 保存 per_location_metrics_all_pipelines.csv ...")
    per_loc_df = pd.DataFrame(per_location_records)
    per_loc_df.to_csv(output_path / "per_location_metrics_all_pipelines.csv", index=False)
    print(f"    → 已保存：{output_path / 'per_location_metrics_all_pipelines.csv'}")


    print("[6] 保存 per_location_metrics_summary.csv ...")
    per_loc_summary = (
        per_loc_df
        .groupby(['class_id', 'location'])[['MCC', 'Precision', 'Recall', 'F1']]
        .agg(['mean', 'std', 'median', 'min', 'max'])
        .reset_index()
    )

    per_loc_summary.columns = [
        '_'.join(col).strip('_') for col in per_loc_summary.columns
    ]
    per_loc_summary.to_csv(output_path / "per_location_metrics_summary.csv", index=False)
    print(f"    → 已保存：{output_path / 'per_location_metrics_summary.csv'}")


    print("[7] 保存 best pipeline per-location metrics ...")
    if best_pipeline_info is not None:
        best_label  = best_pipeline_info['run_fold_label']
        best_run    = best_pipeline_info['run_id']
        best_fold   = best_pipeline_info['fold_id']
        best_y_pred = best_pipeline_info['y_pred']

        best_per_loc_records = []
        for class_id in unique_classes:
            loc_name = LABEL_MAP.get(class_id, f"Location_{class_id}")
            row = compute_per_location_metrics(
                y_test, best_y_pred, class_id, loc_name,
                best_label, best_run, best_fold
            )
            best_per_loc_records.append(row)

        best_per_loc_df = pd.DataFrame(best_per_loc_records)


        best_filename = f"best_pipeline_{best_label}_per_location_metrics.csv"
        best_per_loc_df.to_csv(best_pipeline_dir / best_filename, index=False)

        print(f"    → Best pipeline: {best_label}  (Test MCC = {best_mcc:.4f})")
        print(f"    → 已保存：{best_pipeline_dir / best_filename}")


        best_overall_df = pd.DataFrame([best_pipeline_info['overall_metrics']])
        best_overall_df.to_csv(best_pipeline_dir / f"best_pipeline_{best_label}_overall_metrics.csv", index=False)


    print("[8] 保存 failed_pipelines.csv ...")
    if failed_pipelines:
        failed_df = pd.DataFrame(failed_pipelines)
        failed_df.to_csv(output_path / "failed_pipelines.csv", index=False)
        print(f"    → {len(failed_pipelines)} 个失败记录已保存")
    else:
        print("    → 无失败记录")


    print("\n" + "=" * 70)
    print("评估完成！输出文件汇总：")
    print(f"  overall_pipeline_metrics.csv          — {len(overall_df)} 条 pipeline 记录")
    print(f"  overall_pipeline_metrics_summary.csv  — 各指标统计汇总")
    print(f"  per_location_metrics_all_pipelines.csv — {len(per_loc_df)} 条 per-location 记录")
    print(f"  per_location_metrics_summary.csv      — 每个 location 的均值统计")
    if best_pipeline_info:
        print(f"  best_pipeline_result/                 — Best: {best_pipeline_info['run_fold_label']} (MCC={best_mcc:.4f})")
    if failed_pipelines:
        print(f"  failed_pipelines.csv                  — {len(failed_pipelines)} 个失败 pipeline")
    print("=" * 70)

if __name__ == "__main__":
    main()
