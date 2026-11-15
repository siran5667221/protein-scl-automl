import os
import gc
import json
import pickle
import shutil
import logging
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
for logger_name in ['autosklearn', 'smac', 'dask', 'distributed']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    logger.propagate = False
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
import autosklearn.classification
from autosklearn.metrics import make_scorer
from autosklearn.ensembles.singlebest_ensemble import SingleBest
DATASET_NAME = 'subcell_graphpart17100_homology_train'
TRAIN_PATH = os.environ.get('SUBCELL_FEATURES_CSV', 'data/features/subcell/homology_aware/subcell_graphpart17100_full_features_with_split.csv')
LABEL_COL = 'label'
ROOT_SAVE_DIR = os.environ.get('AUTOSKLEARN_PIPELINE_DIR', 'runs/subcell_graphpart17100_homology_autosklearn_100_pipelines_seed100_resume')
os.makedirs(ROOT_SAVE_DIR, exist_ok=True)
REPEATS = 10
OUTER_FOLDS = 10
TARGET_FEATURES = 100
TOTAL_PIPELINES = REPEATS * OUTER_FOLDS
SEED_START = 100
RESUME = True
EXPECTED_N_FEATURES = 44158
EXPECTED_TOTAL_SAMPLES = 17100
EXPECTED_TRAIN_SAMPLES = 12026
EXPECTED_TEST_SAMPLES = 5074
autosklearn_mcc_scorer = make_scorer(name='mcc', score_func=matthews_corrcoef, optimum=1.0, worst_possible_result=-1.0, greater_is_better=True)
AUTOSKLEARN_CONFIG = {'time_left_for_this_task': 7200, 'per_run_time_limit': 600, 'ensemble_class': SingleBest, 'ensemble_size': 1, 'ensemble_nbest': 1, 'include': None, 'n_jobs': 2, 'metric': autosklearn_mcc_scorer, 'memory_limit': 60000}

def compute_metrics_audit(y_true, y_pred, y_proba, classes):
    metrics = {'MCC': float(matthews_corrcoef(y_true, y_pred)), 'Accuracy': float(accuracy_score(y_true, y_pred)), 'Balanced_Acc': float(balanced_accuracy_score(y_true, y_pred))}
    for avg in ['micro', 'macro', 'weighted']:
        metrics[f'Precision_{avg}'] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        metrics[f'Recall_{avg}'] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
        metrics[f'F1_{avg}'] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    try:
        if y_proba is not None:
            y_true_bin = label_binarize(y_true, classes=classes)
            if len(classes) == 2:
                metrics['ROC_AUC_macro'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                metrics['PR_AUC_macro'] = float(average_precision_score(y_true, y_proba[:, 1]))
            else:
                metrics['ROC_AUC_macro'] = float(roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='macro'))
                metrics['PR_AUC_macro'] = float(average_precision_score(y_true_bin, y_proba, average='macro'))
    except Exception as e:
        metrics['AUC_Error'] = str(e)
    return metrics

def is_pipeline_complete(fold_dir):
    pipeline_path = os.path.join(fold_dir, 'complete_pipeline.pkl')
    metrics_path = os.path.join(fold_dir, 'metrics.json')
    return os.path.exists(pipeline_path) and os.path.exists(metrics_path)

def load_existing_metrics(fold_dir):
    metrics_path = os.path.join(fold_dir, 'metrics.json')
    pipeline_path = os.path.join(fold_dir, 'complete_pipeline.pkl')
    with open(metrics_path, 'r') as f:
        metrics_json = json.load(f)
    val_metrics = metrics_json.get('val_metrics', {})
    row = val_metrics.copy()
    row.update({'dataset': metrics_json.get('dataset', DATASET_NAME), 'run_id': metrics_json.get('run_id'), 'fold_id': metrics_json.get('fold_id'), 'pipeline_id': metrics_json.get('pipeline_id'), 'seed': metrics_json.get('seed'), 'n_selected_features': TARGET_FEATURES, 'pipeline_path': pipeline_path, 'resume_loaded': True})
    return row
print('=' * 80)
print('Train 100 AutoSklearn pipelines on GraphPart17100 homology train set')
print('=' * 80)
print(f'Dataset name: {DATASET_NAME}')
print(f'Train file: {TRAIN_PATH}')
print(f'Output directory: {ROOT_SAVE_DIR}')
print(f'Repeats: {REPEATS}')
print(f'Outer folds: {OUTER_FOLDS}')
print(f'Total pipelines: {TOTAL_PIPELINES}')
print(f'Seed start: {SEED_START}')
print(f'Target selected features: {TARGET_FEATURES}')
print(f'Expected full samples: {EXPECTED_TOTAL_SAMPLES}')
print(f'Expected GraphPart train samples: {EXPECTED_TRAIN_SAMPLES}')
print(f'Expected held-out GraphPart test samples: {EXPECTED_TEST_SAMPLES}')
print('=' * 80)
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f'Cannot find train file: {TRAIN_PATH}')
train_df = pd.read_csv(TRAIN_PATH)
print('\nRaw train dataframe:')
print(f'Shape: {train_df.shape}')
print(f'First 10 columns: {train_df.columns[:10].tolist()}')
print(f'Last 5 columns: {train_df.columns[-5:].tolist()}')
if LABEL_COL not in train_df.columns:
    raise ValueError(f"Cannot find label column '{LABEL_COL}'. Available columns are: {train_df.columns.tolist()[:30]} ...")
if 'split' not in train_df.columns:
    raise ValueError("The input file does not contain a 'split' column. This experiment requires the GraphPart-defined train/test split.")
split_series = train_df['split'].astype(str).str.strip().str.lower()
split_counts = split_series.value_counts(dropna=False)
print('\nFull GraphPart split distribution:')
print(split_counts)
unexpected_split_values = sorted(set(split_series.unique()) - {'train', 'test'})
if unexpected_split_values:
    raise ValueError(f"Unexpected values were found in the split column: {unexpected_split_values}. Expected only 'train' and 'test'.")
observed_total = int(len(train_df))
observed_train = int((split_series == 'train').sum())
observed_test = int((split_series == 'test').sum())
if observed_total != EXPECTED_TOTAL_SAMPLES:
    raise ValueError(f'Expected {EXPECTED_TOTAL_SAMPLES} total GraphPart samples, but found {observed_total}.')
if observed_train != EXPECTED_TRAIN_SAMPLES:
    raise ValueError(f'Expected {EXPECTED_TRAIN_SAMPLES} GraphPart training samples, but found {observed_train}.')
if observed_test != EXPECTED_TEST_SAMPLES:
    raise ValueError(f'Expected {EXPECTED_TEST_SAMPLES} GraphPart test samples, but found {observed_test}.')
train_df = train_df.loc[split_series == 'train'].copy()
if len(train_df) != EXPECTED_TRAIN_SAMPLES:
    raise RuntimeError(f'Internal filtering error: the final training dataframe does not contain exactly {EXPECTED_TRAIN_SAMPLES} rows.')
print('\nGraphPart split validation passed.')
print(f'Total samples in full file: {observed_total}')
print(f'Training samples retained:  {len(train_df)}')
print(f'Test samples held out:      {observed_test}')
print(f'Filtered training shape:    {train_df.shape}')
label_encoder = LabelEncoder()
y_train_original = train_df[LABEL_COL].astype(str).values
y_train_full = label_encoder.fit_transform(y_train_original)
label_mapping = {str(class_name): int(encoded_label) for encoded_label, class_name in enumerate(label_encoder.classes_)}
inverse_label_mapping = {int(encoded_label): str(class_name) for encoded_label, class_name in enumerate(label_encoder.classes_)}
with open(os.path.join(ROOT_SAVE_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)
with open(os.path.join(ROOT_SAVE_DIR, 'label_mapping.json'), 'w') as f:
    json.dump(label_mapping, f, indent=4)
with open(os.path.join(ROOT_SAVE_DIR, 'inverse_label_mapping.json'), 'w') as f:
    json.dump(inverse_label_mapping, f, indent=4)
print('\nLabel mapping:')
for original_label, encoded_label in label_mapping.items():
    print(f'  {original_label} -> {encoded_label}')
metadata_cols = {LABEL_COL, 'protein_id', 'ProteinID', 'ProteinId', 'proteinID', 'partition', 'split', 'header', 'original_header', 'original_id', 'Unnamed: 0', 'index'}
drop_cols = [col for col in train_df.columns if col in metadata_cols]
unnamed_cols = [col for col in train_df.columns if col.startswith('Unnamed')]
drop_cols.extend([col for col in unnamed_cols if col not in drop_cols])
print('\nColumns dropped from feature matrix:')
print(drop_cols)
X_train_full = train_df.drop(columns=drop_cols)
X_train_full = X_train_full.apply(pd.to_numeric, errors='coerce')
na_count = int(X_train_full.isna().sum().sum())
if na_count > 0:
    print(f'\nWarning: {na_count} NA values found after numeric conversion.')
    print('Filling NA values with 0.')
    X_train_full = X_train_full.fillna(0)
print('\nFinal training matrix:')
print(f'X_train_full shape: {X_train_full.shape}')
print(f'y_train_full shape: {y_train_full.shape}')
print(f'Number of classes: {len(np.unique(y_train_full))}')
print(f'Encoded classes: {np.unique(y_train_full)}')
if X_train_full.shape[0] != EXPECTED_TRAIN_SAMPLES:
    raise ValueError(f'Expected {EXPECTED_TRAIN_SAMPLES} training rows in the feature matrix, but got {X_train_full.shape[0]}.')
if X_train_full.shape[1] != EXPECTED_N_FEATURES:
    raise ValueError(f'Expected {EXPECTED_N_FEATURES} feature columns, but got {X_train_full.shape[1]}. Please check metadata column removal.')
feature_columns = X_train_full.columns.tolist()
with open(os.path.join(ROOT_SAVE_DIR, 'feature_columns.json'), 'w') as f:
    json.dump(feature_columns, f, indent=4)
print(f'\nSaved feature column names: {len(feature_columns)}')
class_distribution = pd.DataFrame({'original_label': label_encoder.inverse_transform(np.unique(y_train_full)), 'encoded_label': np.unique(y_train_full), 'count': [int(np.sum(y_train_full == label_id)) for label_id in np.unique(y_train_full)]})
class_distribution_path = os.path.join(ROOT_SAVE_DIR, 'class_distribution.csv')
class_distribution.to_csv(class_distribution_path, index=False)
print('\nClass distribution:')
print(class_distribution)
min_class_count = class_distribution['count'].min()
if min_class_count < OUTER_FOLDS:
    raise ValueError(f'The smallest class has only {min_class_count} samples, but OUTER_FOLDS={OUTER_FOLDS}. Please reduce OUTER_FOLDS or remove extremely rare classes.')
cv_summary_records = []
run_summaries = []
for run_id in range(1, REPEATS + 1):
    print('\n' + '=' * 80)
    print(f'Run {run_id}/{REPEATS}')
    print('=' * 80)
    run_dir = os.path.join(ROOT_SAVE_DIR, f'run_{run_id:02d}')
    os.makedirs(run_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED_START + run_id - 1)
    run_cv_scores = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
        pipeline_id = (run_id - 1) * OUTER_FOLDS + fold_idx
        current_seed = SEED_START + pipeline_id - 1
        fold_dir = os.path.join(run_dir, f'fold_{fold_idx:02d}')
        os.makedirs(fold_dir, exist_ok=True)
        pipeline_path = os.path.join(fold_dir, 'complete_pipeline.pkl')
        if RESUME and is_pipeline_complete(fold_dir):
            print(f'\nPipeline {pipeline_id}/{TOTAL_PIPELINES} already completed. Skip. Run {run_id} Fold {fold_idx} | Seed {current_seed}')
            try:
                existing_row = load_existing_metrics(fold_dir)
                cv_summary_records.append(existing_row)
                if 'MCC' in existing_row and pd.notna(existing_row['MCC']):
                    run_cv_scores.append(float(existing_row['MCC']))
            except Exception as e:
                print(f'  Warning: could not load existing metrics: {e}')
            continue
        print(f'\nPipeline {pipeline_id}/{TOTAL_PIPELINES} | Run {run_id} Fold {fold_idx} | Seed {current_seed}')
        X_tr = X_train_full.iloc[tr_idx]
        X_val = X_train_full.iloc[val_idx]
        y_tr = y_train_full[tr_idx]
        y_val = y_train_full[val_idx]
        print(f'  Training fold shape: {X_tr.shape}')
        print(f'  Validation fold shape: {X_val.shape}')
        scaler = MinMaxScaler()
        X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)
        rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=current_seed, n_jobs=2), n_features_to_select=TARGET_FEATURES, step=0.1)
        X_tr_sel = rfe.fit_transform(X_tr_s, y_tr).astype(np.float32)
        X_val_sel = rfe.transform(X_val_s).astype(np.float32)
        selected_features = X_train_full.columns[rfe.get_support()].tolist()
        del X_tr_s
        del X_val_s
        gc.collect()
        print(f'  Selected features: {len(selected_features)}')
        autosklearn_tmp = os.path.join(fold_dir, f'autosklearn_tmp_seed_{current_seed}')
        autosklearn_output = os.path.join(fold_dir, f'autosklearn_output_seed_{current_seed}')
        if os.path.exists(autosklearn_tmp):
            shutil.rmtree(autosklearn_tmp, ignore_errors=True)
        if os.path.exists(autosklearn_output):
            shutil.rmtree(autosklearn_output, ignore_errors=True)
        clf = autosklearn.classification.AutoSklearnClassifier(seed=current_seed, tmp_folder=autosklearn_tmp, delete_tmp_folder_after_terminate=True, **AUTOSKLEARN_CONFIG)
        try:
            clf.fit(X_tr_sel, y_tr)
        except Exception as e:
            print(f'  ERROR: AutoSklearn failed at pipeline {pipeline_id}.')
            print(f'  Run {run_id}, Fold {fold_idx}, Seed {current_seed}')
            print(f'  Error message: {e}')
            error_path = os.path.join(fold_dir, 'error.txt')
            with open(error_path, 'w') as f:
                f.write(f'Dataset: {DATASET_NAME}\n')
                f.write(f'Pipeline: {pipeline_id}\n')
                f.write(f'Run: {run_id}\n')
                f.write(f'Fold: {fold_idx}\n')
                f.write(f'Seed: {current_seed}\n')
                f.write(f'Error: {str(e)}\n')
            del X_tr_sel
            del X_val_sel
            gc.collect()
            continue
        classes = clf.classes_
        y_val_pred = clf.predict(X_val_sel)
        try:
            y_val_proba = clf.predict_proba(X_val_sel)
        except Exception:
            y_val_proba = None
        val_metrics = compute_metrics_audit(y_true=y_val, y_pred=y_val_pred, y_proba=y_val_proba, classes=classes)
        run_cv_scores.append(val_metrics['MCC'])
        print(f"  Validation MCC: {val_metrics['MCC']:.4f}")
        print(f"  Validation Accuracy: {val_metrics['Accuracy']:.4f}")
        print(f"  Validation Balanced Acc: {val_metrics['Balanced_Acc']:.4f}")
        pipeline_config = {'dataset': DATASET_NAME, 'run_id': run_id, 'fold_idx': fold_idx, 'pipeline_id': pipeline_id, 'seed': current_seed, 'scaler': scaler, 'rfe': rfe, 'selected_features': selected_features, 'feature_columns': feature_columns, 'model': clf, 'label_encoder': label_encoder, 'label_mapping': label_mapping, 'inverse_label_mapping': inverse_label_mapping, 'classes': classes, 'val_metrics': val_metrics, 'train_path': TRAIN_PATH, 'target_features': TARGET_FEATURES, 'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline_config, f)
        with open(os.path.join(fold_dir, 'selected_features.json'), 'w') as f:
            json.dump(selected_features, f, indent=4)
        metrics_json = {'dataset': DATASET_NAME, 'run_id': run_id, 'fold_id': fold_idx, 'pipeline_id': pipeline_id, 'seed': current_seed, 'n_selected_features': len(selected_features), 'pipeline_path': pipeline_path, 'val_metrics': val_metrics}
        with open(os.path.join(fold_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_json, f, indent=4)
        row = val_metrics.copy()
        row.update({'dataset': DATASET_NAME, 'run_id': run_id, 'fold_id': fold_idx, 'pipeline_id': pipeline_id, 'seed': current_seed, 'n_selected_features': len(selected_features), 'pipeline_path': pipeline_path, 'resume_loaded': False})
        cv_summary_records.append(row)
        partial_cv_path = os.path.join(ROOT_SAVE_DIR, 'nested_cv_results_partial.csv')
        pd.DataFrame(cv_summary_records).to_csv(partial_cv_path, index=False)
        del X_tr_sel
        del X_val_sel
        del y_val_pred
        del y_val_proba
        gc.collect()
    if len(run_cv_scores) > 0:
        run_avg_mcc = float(np.mean(run_cv_scores))
        run_std_mcc = float(np.std(run_cv_scores))
    else:
        run_avg_mcc = np.nan
        run_std_mcc = np.nan
    run_summary = {'dataset': DATASET_NAME, 'run_id': run_id, 'cv_mcc_mean': run_avg_mcc, 'cv_mcc_std': run_std_mcc, 'n_folds_completed_or_loaded': len(run_cv_scores), 'n_folds_expected': OUTER_FOLDS}
    run_summaries.append(run_summary)
    print('\n' + '-' * 60)
    print(f'Run {run_id} mean CV MCC: {run_avg_mcc:.4f} ± {run_std_mcc:.4f}')
    print('-' * 60)
    partial_run_summary_path = os.path.join(ROOT_SAVE_DIR, 'run_summary_partial.csv')
    pd.DataFrame(run_summaries).to_csv(partial_run_summary_path, index=False)
cv_df = pd.DataFrame(cv_summary_records)
run_summary_df = pd.DataFrame(run_summaries)
cv_result_path = os.path.join(ROOT_SAVE_DIR, 'nested_cv_results.csv')
run_summary_path = os.path.join(ROOT_SAVE_DIR, 'run_summary.csv')
cv_df.to_csv(cv_result_path, index=False)
run_summary_df.to_csv(run_summary_path, index=False)
print('\n' + '=' * 80)
print('Final nested CV summary')
print('=' * 80)
for metric in ['MCC', 'Accuracy', 'Balanced_Acc', 'F1_macro', 'F1_weighted', 'ROC_AUC_macro', 'PR_AUC_macro']:
    if metric in cv_df.columns:
        values = cv_df[metric].dropna()
        if len(values) > 0:
            print(f'{metric:20s}: {values.mean():.4f} ± {values.std():.4f}')
print('\nSaved files:')
print(f'  CV results: {cv_result_path}')
print(f'  Run summary: {run_summary_path}')
print(f'  Class distribution: {class_distribution_path}')
print(f"  Feature columns: {os.path.join(ROOT_SAVE_DIR, 'feature_columns.json')}")
print(f"  Label encoder: {os.path.join(ROOT_SAVE_DIR, 'label_encoder.pkl')}")
print(f"  Label mapping: {os.path.join(ROOT_SAVE_DIR, 'label_mapping.json')}")
print(f"  Inverse label mapping: {os.path.join(ROOT_SAVE_DIR, 'inverse_label_mapping.json')}")
print(f'  Pipelines saved under: {ROOT_SAVE_DIR}/run_xx/fold_xx/complete_pipeline.pkl')
print('\nDone.')
