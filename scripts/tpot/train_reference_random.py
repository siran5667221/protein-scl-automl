import argparse
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Train the 100 Reference Random TPOT workflows.')
    parser.add_argument("--train-csv", type=Path, default=project_root / "data/reference/random/train_numeric_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "runs/protein_tpot_rfe_longtime_final")
    args = parser.parse_args()
    if not args.train_csv.is_file():
        parser.error(f"Training CSV not found: {args.train_csv}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty. Use a new directory; this training script has no resume mode.")

    import os
    import json
    import pickle
    import logging
    import warnings
    import pandas as pd
    import numpy as np
    from datetime import datetime


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


    ROOT_SAVE_DIR = str(args.output_dir)
    os.makedirs(ROOT_SAVE_DIR, exist_ok=True)


    REPEATS = 10
    OUTER_FOLDS = 10
    TARGET_FEATURES = 100


    TPOT_CONFIG = {
        'generations': 100,
        'population_size': 50,
        'offspring_size': None,
        'scoring': 'matthews_corrcoef',
        'max_time_mins': 120,
        'max_eval_time_mins': 10,
        'cv': 10,
        'verbosity': 0,
        'n_jobs': -1,
        'early_stop': 5,
        'template': None
    }

    print("="*80)
    print("TPOT + RFE 嵌套CV实现 (严格对照版)")
    print("="*80)
    print(f"重复次数: {REPEATS}")
    print(f"CV折数: {OUTER_FOLDS}")
    print(f"目标特征数: {TARGET_FEATURES}")
    print("="*80)


    def compute_metrics_audit(y_true, y_pred, y_proba, classes):
        m = {
            'MCC': float(matthews_corrcoef(y_true, y_pred)),
            'Accuracy': float(accuracy_score(y_true, y_pred)),
            'Balanced_Acc': float(balanced_accuracy_score(y_true, y_pred)),
        }
        for avg in ['micro', 'macro', 'weighted']:
            m[f'Precision_{avg}'] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
            m[f'Recall_{avg}'] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
            m[f'F1_{avg}'] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))

        try:

            if y_proba is not None:
                y_true_bin = label_binarize(y_true, classes=classes)
                if len(classes) == 2:

                    prob_col = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
                    m['ROC_AUC_macro'] = float(roc_auc_score(y_true, prob_col))
                    m['PR_AUC_macro'] = float(average_precision_score(y_true, prob_col))
                else:
                    m['ROC_AUC_macro'] = float(roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='macro'))
                    m['PR_AUC_macro'] = float(average_precision_score(y_true_bin, y_proba, average='macro'))
            else:
                m['ROC_AUC_macro'] = np.nan
                m['PR_AUC_macro'] = np.nan
        except Exception as e:
            m['AUC_Error'] = str(e)
        return m


    print("\n加载数据...")
    train_df = pd.read_csv(args.train_csv)
    X_train_full = train_df.drop(['ProteinID', 'label'], axis=1)
    y_train_full = train_df['label'].values


    cv_summary_records = []
    all_fold_pipelines = {}

    for run_id in range(1, REPEATS + 1):
        print(f"\nRun {run_id}/{REPEATS}")
        run_dir = os.path.join(ROOT_SAVE_DIR, f"run_{run_id:02d}")
        os.makedirs(run_dir, exist_ok=True)

        skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=run_id * 777)
        run_cv_scores = []
        run_fold_pipelines = []

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
            current_seed = run_id * 100 + fold_idx
            fold_dir = os.path.join(run_dir, f"fold_{fold_idx:02d}")
            os.makedirs(fold_dir, exist_ok=True)

            print(f"  Fold {fold_idx}/{OUTER_FOLDS} (Seed: {current_seed})", end='')

            X_tr, X_val = X_train_full.iloc[tr_idx], X_train_full.iloc[val_idx]
            y_tr, y_val = y_train_full[tr_idx], y_train_full[val_idx]


            scaler = MinMaxScaler()
            X_tr_s = scaler.fit_transform(X_tr)
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
            X_tr_sel = rfe.fit_transform(X_tr_s, y_tr)
            X_val_sel = rfe.transform(X_val_s)
            selected_features = X_train_full.columns[rfe.get_support()].tolist()


            tpot_clf = TPOTClassifier(
                random_state=current_seed,
                **TPOT_CONFIG
            )
            tpot_clf.fit(X_tr_sel, y_tr)


            classes = tpot_clf.classes_

            best_model = tpot_clf.fitted_pipeline_

            y_val_pred = best_model.predict(X_val_sel)


            y_val_proba = None
            if hasattr(best_model, "predict_proba"):
                try:
                    y_val_proba = best_model.predict_proba(X_val_sel)
                except:
                    y_val_proba = None

            val_metrics = compute_metrics_audit(y_val, y_val_pred, y_val_proba, classes)
            print(f" → Val MCC: {val_metrics['MCC']:.4f}")
            run_cv_scores.append(val_metrics['MCC'])


            pipeline_config = {
                'fold_idx': fold_idx,
                'seed': current_seed,
                'scaler': scaler,
                'rfe': rfe,
                'selected_features': selected_features,
                'model': best_model,
                'val_metrics': val_metrics
            }
            run_fold_pipelines.append(pipeline_config)


            with open(os.path.join(fold_dir, "complete_pipeline.pkl"), "wb") as f:
                pickle.dump(pipeline_config, f)

            with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
                json.dump({
                    "meta": {"run_id": run_id, "fold_id": fold_idx, "seed": current_seed},
                    "val_metrics": val_metrics,
                    "pipeline_str": str(best_model)
                }, f, indent=4)

            row = val_metrics.copy()
            row.update({'run_id': run_id, 'fold_id': fold_idx, 'seed': current_seed})
            cv_summary_records.append(row)

        all_fold_pipelines[run_id] = run_fold_pipelines


    cv_df = pd.DataFrame(cv_summary_records)
    cv_df.to_csv(os.path.join(ROOT_SAVE_DIR, "nested_cv_results.csv"), index=False)

    print("\n" + "="*80)
    print("阶段1完成。")

if __name__ == "__main__":
    main()
