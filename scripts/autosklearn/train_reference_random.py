import argparse
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Train Reference Random Auto-sklearn SingleBest workflows.')
    parser.add_argument("--test-csv", type=Path, default=project_root / "data/reference/random/test_numeric_labels.csv")
    parser.add_argument("--train-csv", type=Path, default=project_root / "data/reference/random/train_numeric_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "runs/protein_rfe_autosklearn_longtime_method_final")
    args = parser.parse_args()
    for path in (args.train_csv, args.test_csv):
        if not path.is_file():
            parser.error(f"Input CSV not found: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty. Choose a new directory; the original training loop has no resume mode.")

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


    for logger_name in ['autosklearn', 'smac', 'dask', 'distributed']:
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


    import autosklearn.classification
    from autosklearn.metrics import make_scorer
    from autosklearn.ensembles.singlebest_ensemble import SingleBest


    autosklearn_mcc_scorer = make_scorer(
        name="mcc",
        score_func=matthews_corrcoef,
        optimum=1.0,
        worst_possible_result=-1.0,
        greater_is_better=True
    )

    ROOT_SAVE_DIR = str(args.output_dir)
    os.makedirs(ROOT_SAVE_DIR, exist_ok=True)


    REPEATS = 10
    OUTER_FOLDS = 10
    TARGET_FEATURES = 100


    AUTOSKLEARN_CONFIG = {
        'time_left_for_this_task': 7200,
        'per_run_time_limit': 600,

        'ensemble_class': SingleBest,
        'ensemble_size': 1,
        'ensemble_nbest': 1,
        'include': None,
        'n_jobs': 3,
        'metric': autosklearn_mcc_scorer,
        'memory_limit': 60000,
    }

    print("="*80)
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
                    m['ROC_AUC_macro'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                    m['PR_AUC_macro'] = float(average_precision_score(y_true, y_proba[:, 1]))
                else:
                    m['ROC_AUC_macro'] = float(roc_auc_score(y_true_bin, y_proba,
                                                             multi_class='ovr', average='macro'))
                    m['PR_AUC_macro'] = float(average_precision_score(y_true_bin, y_proba,
                                                                      average='macro'))
        except Exception as e:
            m['AUC_Error'] = str(e)

        return m


    print("\n加载数据...")
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    X_train_full = train_df.drop(['ProteinID', 'label'], axis=1)
    y_train_full = train_df['label'].values
    X_test_full = test_df.drop(['ProteinID', 'label'], axis=1)
    y_test_full = test_df['label'].values

    print(f"训练集: {X_train_full.shape}")
    print(f"测试集: {X_test_full.shape}")


    print("\n" + "="*80)
    print("阶段1: 嵌套CV（保存所有fold的pipeline配置）")
    print("="*80)

    cv_summary_records = []
    run_summaries = []
    all_fold_pipelines = {}

    for run_id in range(1, REPEATS + 1):
        print(f"\n{'='*60}")
        print(f"Run {run_id}/{REPEATS}")
        print(f"{'='*60}")

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

            del X_val_s

            selected_features = X_train_full.columns[rfe.get_support()].tolist()


            clf = autosklearn.classification.AutoSklearnClassifier(
                seed=current_seed,
                **AUTOSKLEARN_CONFIG
            )
            clf.fit(X_tr_sel, y_tr)
            del X_tr_sel


            classes = clf.classes_
            y_val_pred = clf.predict(X_val_sel)
            y_val_proba = clf.predict_proba(X_val_sel)
            val_metrics = compute_metrics_audit(y_val, y_val_pred, y_val_proba, classes)

            del X_val_sel
            del y_val_pred
            del y_val_proba

            print(f" → Val MCC: {val_metrics['MCC']:.4f}")
            run_cv_scores.append(val_metrics['MCC'])


            pipeline_config = {
                'fold_idx': fold_idx,
                'seed': current_seed,
                'scaler': scaler,
                'rfe': rfe,
                'selected_features': selected_features,
                'model': clf,
                'val_metrics': val_metrics
            }
            run_fold_pipelines.append(pipeline_config)


            with open(os.path.join(fold_dir, "complete_pipeline.pkl"), "wb") as f:
                pickle.dump(pipeline_config, f)

            with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
                json.dump({
                    "meta": {"run_id": run_id, "fold_id": fold_idx, "seed": current_seed},
                    "val_metrics": val_metrics
                }, f, indent=4)

            row = val_metrics.copy()
            row.update({
                'run_id': run_id,
                'fold_id': fold_idx,
                'seed': current_seed,
            })
            cv_summary_records.append(row)


        all_fold_pipelines[run_id] = run_fold_pipelines

        run_avg_mcc = np.mean(run_cv_scores)
        run_std_mcc = np.std(run_cv_scores)
        print(f"  Run {run_id} 平均CV MCC: {run_avg_mcc:.4f} ± {run_std_mcc:.4f}")

        run_summaries.append({
            'run_id': run_id,
            'cv_mcc_mean': run_avg_mcc,
            'cv_mcc_std': run_std_mcc
        })


    cv_df = pd.DataFrame(cv_summary_records)
    cv_df.to_csv(os.path.join(ROOT_SAVE_DIR, "nested_cv_results.csv"), index=False)

    print("\n" + "="*80)
    print("嵌套CV结果汇总")
    print("="*80)

    for metric in ['MCC', 'Accuracy', 'Balanced_Acc', 'F1_macro', 'ROC_AUC_macro']:
        if metric in cv_df.columns:
            values = cv_df[metric].dropna()
            if len(values) > 0:
                print(f"{metric:20s}: {values.mean():.4f} ± {values.std():.4f}")

if __name__ == "__main__":
    main()
