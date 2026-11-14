import argparse
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Train Reference homology-aware TPOT workflows.')
    parser.add_argument("--train-csv", type=Path, default=project_root / "data/reference/homology_aware/Train_Features_NoHomology.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "runs/protein_tpot_rfe_NoHomology_Final")
    args = parser.parse_args()
    if not args.train_csv.is_file():
        parser.error(f"Training CSV not found: {args.train_csv}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty. Use a new directory; this training loop has no resume mode.")

    import os
    import json
    import pickle
    import logging
    import warnings
    import pandas as pd
    import numpy as np


    warnings.filterwarnings('ignore')
    os.environ['PYTHONWARNINGS'] = 'ignore'

    for logger_name in ['tpot', 'deap', 'update_checker']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False


    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import MinMaxScaler, label_binarize
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score,
        recall_score, f1_score, matthews_corrcoef, roc_auc_score,
        average_precision_score
    )
    from tpot import TPOTClassifier


    TRAIN_FILE    = str(args.train_csv)
    ROOT_SAVE_DIR = str(args.output_dir)

    REPEATS         = 10
    OUTER_FOLDS     = 10
    TARGET_FEATURES = 100

    TPOT_CONFIG = {
        'generations':        100,
        'population_size':    50,
        'offspring_size':     None,
        'scoring':            'matthews_corrcoef',
        'max_time_mins':      120,
        'max_eval_time_mins': 10,
        'cv':                 10,
        'verbosity':          0,
        'n_jobs':             -1,
        'early_stop':         5,
        'template':           None,
    }


    os.makedirs(ROOT_SAVE_DIR, exist_ok=True)

    print("=" * 80)
    print("TPOT + RFE 嵌套CV — 非同源训练集版本")
    print("=" * 80)
    print(f"训练集:    {TRAIN_FILE}")
    print(f"保存目录:  {ROOT_SAVE_DIR}")
    print(f"重复次数:  {REPEATS}")
    print(f"CV折数:    {OUTER_FOLDS}")
    print(f"目标特征:  {TARGET_FEATURES}")
    print("=" * 80)


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
                if len(classes) == 2:
                    prob_col = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
                    m['ROC_AUC_macro'] = float(roc_auc_score(y_true, prob_col))
                    m['PR_AUC_macro']  = float(average_precision_score(y_true, prob_col))
                else:
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


    print(f"\n加载训练集: {TRAIN_FILE} ...")
    train_df     = pd.read_csv(TRAIN_FILE)
    drop_cols    = [c for c in ['ProteinID', 'label'] if c in train_df.columns]
    X_train_full = train_df.drop(columns=drop_cols)
    y_train_full = train_df['label'].values

    if not np.issubdtype(y_train_full.dtype, np.number):
        mapping = {
            'Cell.membrane': 0, 'Cytoplasm': 1, 'Cytoplasm-Nucleus': 2,
            'Endoplasmic.reticulum': 3, 'Extracellular': 4, 'Golgi.apparatus': 5,
            'Lysosome/Vacuole': 6, 'Mitochondrion': 7, 'Nucleus': 8,
            'Peroxisome': 9, 'Plastid': 10
        }
        y_train_full = pd.Series(y_train_full).map(mapping).values.astype(int)

    print(f"训练集: {X_train_full.shape}")
    print(f"类别分布:\n{pd.Series(y_train_full).value_counts().sort_index()}\n")


    cv_summary_records = []
    all_fold_pipelines = {}

    for run_id in range(1, REPEATS + 1):
        print(f"\nRun {run_id}/{REPEATS}")
        run_dir = os.path.join(ROOT_SAVE_DIR, f"run_{run_id:02d}")
        os.makedirs(run_dir, exist_ok=True)

        skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=run_id * 777)
        run_cv_scores    = []
        run_fold_pipelines = []

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
            current_seed = run_id * 100 + fold_idx
            fold_dir     = os.path.join(run_dir, f"fold_{fold_idx:02d}")
            os.makedirs(fold_dir, exist_ok=True)

            print(f"  Fold {fold_idx}/{OUTER_FOLDS} (Seed: {current_seed})", end='', flush=True)

            X_tr  = X_train_full.iloc[tr_idx]
            X_val = X_train_full.iloc[val_idx]
            y_tr  = y_train_full[tr_idx]
            y_val = y_train_full[val_idx]


            scaler  = MinMaxScaler()
            X_tr_s  = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)


            rfe = RFE(
                estimator=RandomForestClassifier(
                    n_estimators=50,
                    class_weight='balanced',
                    random_state=current_seed,
                    n_jobs=-1
                ),
                n_features_to_select=TARGET_FEATURES,
                step=0.1
            )
            X_tr_sel  = rfe.fit_transform(X_tr_s, y_tr)
            X_val_sel = rfe.transform(X_val_s)
            selected_features = X_train_full.columns[rfe.get_support()].tolist()


            tpot_clf = TPOTClassifier(random_state=current_seed, **TPOT_CONFIG)
            tpot_clf.fit(X_tr_sel, y_tr)

            classes    = tpot_clf.classes_
            best_model = tpot_clf.fitted_pipeline_

            y_val_pred  = best_model.predict(X_val_sel)
            y_val_proba = None
            if hasattr(best_model, "predict_proba"):
                try:
                    y_val_proba = best_model.predict_proba(X_val_sel)
                except Exception:
                    y_val_proba = None

            val_metrics = compute_metrics_audit(y_val, y_val_pred, y_val_proba, classes)
            print(f" → Val MCC: {val_metrics['MCC']:.4f}")
            run_cv_scores.append(val_metrics['MCC'])


            pipeline_config = {
                'fold_idx':          fold_idx,
                'seed':              current_seed,
                'scaler':            scaler,
                'rfe':               rfe,
                'selected_features': selected_features,
                'model':             best_model,
                'val_metrics':       val_metrics,
                'train_file':        TRAIN_FILE,
            }
            run_fold_pipelines.append(pipeline_config)

            with open(os.path.join(fold_dir, "complete_pipeline.pkl"), "wb") as f:
                pickle.dump(pipeline_config, f)

            with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
                json.dump({
                    "meta":         {"run_id": run_id, "fold_id": fold_idx,
                                     "seed": current_seed, "train_file": TRAIN_FILE},
                    "val_metrics":  val_metrics,
                    "pipeline_str": str(best_model)
                }, f, indent=4)

            row = val_metrics.copy()
            row.update({'run_id': run_id, 'fold_id': fold_idx, 'seed': current_seed})
            cv_summary_records.append(row)

        all_fold_pipelines[run_id] = run_fold_pipelines
        run_avg = np.mean(run_cv_scores)
        run_std = np.std(run_cv_scores)
        print(f"  Run {run_id} 平均CV MCC: {run_avg:.4f} ± {run_std:.4f}")


    cv_df = pd.DataFrame(cv_summary_records)
    cv_df.to_csv(os.path.join(ROOT_SAVE_DIR, "nested_cv_results.csv"), index=False)

    print("\n" + "=" * 80)
    print("嵌套CV结果汇总")
    print("=" * 80)
    for metric in ['MCC', 'Accuracy', 'Balanced_Acc', 'F1_macro', 'ROC_AUC_macro']:
        if metric in cv_df.columns:
            values = cv_df[metric].dropna()
            if len(values) > 0:
                print(f"{metric:20s}: {values.mean():.4f} ± {values.std():.4f}")

    print(f"\n训练完成！Pipeline 保存至: {ROOT_SAVE_DIR}/")

if __name__ == "__main__":
    main()
