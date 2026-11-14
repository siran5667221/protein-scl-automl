import argparse
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Evaluate saved Reference Random Auto-sklearn workflows.')
    parser.add_argument("--test-csv", type=Path, default=project_root / "data/reference/random/test_numeric_labels.csv")
    parser.add_argument("--model-dir", type=Path, default=project_root / "runs/protein_rfe_autosklearn_longtime_method_final")
    parser.add_argument("--output-dir", type=Path, default=project_root / "runs/protein_rfe_autosklearn_longtime_method_final/per_location_evaluation")
    args = parser.parse_args()
    if not args.test_csv.is_file():
        parser.error(f"Test CSV not found: {args.test_csv}")
    args.output_dir.mkdir(parents=True, exist_ok=True)


    import os
    import pickle
    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score,
        recall_score, f1_score, matthews_corrcoef, roc_auc_score,
        average_precision_score
    )


    ROOT_SAVE_DIR = str(args.model_dir)
    TEST_FILE     = str(args.test_csv)
    REPEATS       = 10
    OUTER_FOLDS   = 10

    LABEL_NAMES = {
        0: 'Cell.membrane',         1: 'Cytoplasm',       2: 'Cytoplasm-Nucleus',
        3: 'Endoplasmic.reticulum', 4: 'Extracellular',   5: 'Golgi.apparatus',
        6: 'Lysosome/Vacuole',      7: 'Mitochondrion',   8: 'Nucleus',
        9: 'Peroxisome',           10: 'Plastid'
    }


    print(f"加载测试集: {TEST_FILE}")
    test_df      = pd.read_csv(TEST_FILE)
    X_test_full  = test_df.drop(columns=["ProteinID", "label"], errors="ignore")
    X_test_full  = X_test_full.select_dtypes(include=[np.number])
    y_test_full  = test_df["label"].values
    print(f"测试集: {X_test_full.shape}  类别数: {len(np.unique(y_test_full))}")


    def per_class_mcc(y_true, y_pred, classes):

        results = []
        for c in classes:
            y_true_bin = (y_true == c).astype(int)
            y_pred_bin = (y_pred == c).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
            denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            mcc   = (tp*tn - fp*fn) / denom if denom > 0 else 0.0
            results.append({
                "class_id":  c,
                "location":  LABEL_NAMES[c],
                "MCC":       round(mcc, 4),
                "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                "n_samples": int(y_true_bin.sum())
            })
        return pd.DataFrame(results)


    def compute_metrics_audit(y_true, y_pred, y_proba, classes):
        m = {
            'MCC':          float(matthews_corrcoef(y_true, y_pred)),
            'Accuracy':     float(accuracy_score(y_true, y_pred)),
            'Balanced_Acc': float(balanced_accuracy_score(y_true, y_pred)),
        }
        for avg in ['micro', 'macro', 'weighted']:
            m[f'Precision_{avg}'] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
            m[f'Recall_{avg}']    = float(recall_score(y_true, y_pred,    average=avg, zero_division=0))
            m[f'F1_{avg}']        = float(f1_score(y_true, y_pred,        average=avg, zero_division=0))
        try:
            if y_proba is not None:
                y_true_bin = label_binarize(y_true, classes=classes)
                m['ROC_AUC_macro'] = float(roc_auc_score(
                    y_true_bin, y_proba, multi_class='ovr', average='macro'))
                m['PR_AUC_macro']  = float(average_precision_score(
                    y_true_bin, y_proba, average='macro'))
            else:
                m['ROC_AUC_macro'] = np.nan
                m['PR_AUC_macro']  = np.nan
        except Exception as e:
            m['AUC_Error'] = str(e)
        return m


    print(f"\n{'='*60}")
    print(f"开始评估：{REPEATS * OUTER_FOLDS} 个 Pipeline")
    print(f"{'='*60}")

    all_class_mcc    = []
    all_test_records = []
    classes          = list(LABEL_NAMES.keys())

    for run_id in range(1, REPEATS + 1):
        for fold_idx in range(1, OUTER_FOLDS + 1):
            pkl_path = os.path.join(
                ROOT_SAVE_DIR,
                f"run_{run_id:02d}",
                f"fold_{fold_idx:02d}",
                "complete_pipeline.pkl"
            )

            if not os.path.exists(pkl_path):
                print(f"  SKIP R{run_id}_F{fold_idx} — 文件不存在")
                continue

            try:
                with open(pkl_path, "rb") as f:
                    pipe = pickle.load(f)


                X_s    = pipe['scaler'].transform(X_test_full)
                X_sel  = pipe['rfe'].transform(X_s)
                y_pred = pipe['model'].predict(X_sel)

                y_proba = None
                if hasattr(pipe['model'], "predict_proba"):
                    y_proba = pipe['model'].predict_proba(X_sel)


                metrics = compute_metrics_audit(y_test_full, y_pred, y_proba, classes)
                all_test_records.append({
                    'Run_Fold':   f"R{run_id}_F{fold_idx}",
                    'MCC':        metrics.get('MCC'),
                    'ROC_AUC':    metrics.get('ROC_AUC_macro'),
                    'PR_AUC':     metrics.get('PR_AUC_macro'),
                    'F1_Weighted':metrics.get('F1_weighted'),
                    'Accuracy':   metrics.get('Accuracy'),
                })


                df_mcc             = per_class_mcc(y_test_full, y_pred, classes)
                df_mcc["Run_Fold"] = f"R{run_id}_F{fold_idx}"
                all_class_mcc.append(df_mcc)

                print(f"  OK  R{run_id}_F{fold_idx} | MCC={metrics['MCC']:.4f}")

            except Exception as e:
                print(f"  ERR R{run_id}_F{fold_idx}: {e}")


    res_df = pd.DataFrame(all_test_records)

    print("\n" + "="*60)
    print("整体指标汇总（100个pipeline）")
    print("="*60)
    stats_list = []
    for col in ['MCC', 'ROC_AUC', 'PR_AUC', 'F1_Weighted', 'Accuracy']:
        if col not in res_df.columns:
            continue
        data = res_df[col].dropna()
        mean = data.mean()
        std  = data.std()
        ci_low, ci_high = stats.t.interval(
            0.95, len(data)-1, loc=mean, scale=stats.sem(data)
        )
        stats_list.append({
            'Metric':  col,
            'Mean':    f"{mean:.4f}",
            'Std':     f"{std:.4f}",
            '95% CI':  f"[{ci_low:.4f}, {ci_high:.4f}]",
            'Min':     f"{data.min():.4f}",
            'Max':     f"{data.max():.4f}",
        })
    print(pd.DataFrame(stats_list).to_string(index=False))


    df_all   = pd.concat(all_class_mcc, ignore_index=True)
    summary  = (
        df_all.groupby("location")["MCC"]
        .agg(["mean","std","min","max"])
        .round(4)
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    print("\n" + "="*60)
    print("每个亚细胞定位的 MCC（100个pipeline均值）")
    print("="*60)
    print(summary.to_string(index=False))


    res_df.to_csv(os.path.join(args.output_dir, "full_multi_metric_test_results.csv"), index=False)
    df_all.to_csv(os.path.join(args.output_dir, "per_class_mcc_all_pipelines.csv"),    index=False)
    summary.to_csv(os.path.join(args.output_dir, "per_class_mcc_summary.csv"),          index=False)

    print(f"\nSaved results to: {args.output_dir}/")

if __name__ == "__main__":
    main()
