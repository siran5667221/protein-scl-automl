import os
import gc
import json
import pickle
import hashlib
import logging
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
for logger_name in ['tpot', 'deap', 'update_checker']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    logger.propagate = False
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, make_scorer
from tpot import TPOTClassifier
DATASET_NAME = 'subcell_random18097_train'
TRAIN_PATH = os.environ.get('SUBCELL_TRAIN_CSV', 'data/features/subcell/random/subcell_random18097_train_features_with_labels.csv')
LABEL_COL = 'label'
ROOT_SAVE_DIR = os.environ.get('TPOT_PIPELINE_DIR', 'runs/subcell_random18097_tpot_100_pipelines_seed100_resume')
os.makedirs(ROOT_SAVE_DIR, exist_ok=True)
REPEATS = 10
OUTER_FOLDS = 10
TARGET_FEATURES = 100
TOTAL_PIPELINES = REPEATS * OUTER_FOLDS
SEED_START = 100
RESUME = True
STRICT_RESUME_VALIDATION = True
MCC_SCORER = make_scorer(matthews_corrcoef, greater_is_better=True)
TPOT_CONFIG = {'generations': 100, 'population_size': 50, 'offspring_size': None, 'scoring': MCC_SCORER, 'max_time_mins': 120, 'max_eval_time_mins': 10, 'cv': 10, 'verbosity': 0, 'n_jobs': 2, 'early_stop': 5, 'template': None, 'config_dict': 'TPOT light'}

def atomic_json_dump(data, path):
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    with open(temp_path, 'w') as handle:
        json.dump(data, handle, indent=4, allow_nan=True)
    os.replace(temp_path, path)

def atomic_pickle_dump(data, path):
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    with open(temp_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, path)

def atomic_csv_dump(dataframe, path):
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    dataframe.to_csv(temp_path, index=False)
    os.replace(temp_path, path)

def sha256_of_strings(values):
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(str(value).encode('utf-8'))
        hasher.update(b'\n')
    return hasher.hexdigest()

def align_probabilities(raw_proba, model_classes, all_classes):
    if raw_proba is None:
        return None
    raw_proba = np.asarray(raw_proba, dtype=np.float64)
    model_classes = np.asarray(model_classes, dtype=int)
    all_classes = np.asarray(all_classes, dtype=int)
    aligned = np.zeros((raw_proba.shape[0], len(all_classes)), dtype=np.float64)
    class_to_column = {int(class_value): column_index for column_index, class_value in enumerate(all_classes)}
    for source_column, class_value in enumerate(model_classes):
        class_value = int(class_value)
        if class_value not in class_to_column:
            raise ValueError(f'Model class {class_value} is absent from the global class set.')
        aligned[:, class_to_column[class_value]] = raw_proba[:, source_column]
    return aligned

def compute_metrics_audit(y_true, y_pred, y_proba, all_classes):
    all_classes = np.asarray(all_classes, dtype=int)
    metrics = {'MCC': float(matthews_corrcoef(y_true, y_pred)), 'Accuracy': float(accuracy_score(y_true, y_pred)), 'Balanced_Acc': float(balanced_accuracy_score(y_true, y_pred))}
    for average in ['micro', 'macro', 'weighted']:
        metrics[f'Precision_{average}'] = float(precision_score(y_true, y_pred, labels=all_classes, average=average, zero_division=0))
        metrics[f'Recall_{average}'] = float(recall_score(y_true, y_pred, labels=all_classes, average=average, zero_division=0))
        metrics[f'F1_{average}'] = float(f1_score(y_true, y_pred, labels=all_classes, average=average, zero_division=0))
    if y_proba is None:
        metrics['ROC_AUC_macro'] = np.nan
        metrics['PR_AUC_macro'] = np.nan
        return metrics
    try:
        y_true_binary = label_binarize(y_true, classes=all_classes)
        if len(all_classes) == 2:
            positive_probability = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
            metrics['ROC_AUC_macro'] = float(roc_auc_score(y_true, positive_probability))
            metrics['PR_AUC_macro'] = float(average_precision_score(y_true, positive_probability))
        else:
            metrics['ROC_AUC_macro'] = float(roc_auc_score(y_true_binary, y_proba, multi_class='ovr', average='macro'))
            metrics['PR_AUC_macro'] = float(average_precision_score(y_true_binary, y_proba, average='macro'))
    except Exception as exc:
        metrics['ROC_AUC_macro'] = np.nan
        metrics['PR_AUC_macro'] = np.nan
        metrics['AUC_Error'] = str(exc)
    return metrics

def is_pipeline_complete(fold_dir):
    pipeline_path = os.path.join(fold_dir, 'complete_pipeline.pkl')
    metrics_path = os.path.join(fold_dir, 'metrics.json')
    return os.path.exists(pipeline_path) and os.path.exists(metrics_path)

def count_existing_complete_pipelines(root_dir):
    return len(list(Path(root_dir).glob('run_*/fold_*/complete_pipeline.pkl')))

def load_existing_metrics(fold_dir, expected_run_id, expected_fold_id, expected_pipeline_id, expected_seed):
    metrics_path = os.path.join(fold_dir, 'metrics.json')
    pipeline_path = os.path.join(fold_dir, 'complete_pipeline.pkl')
    selected_features_path = os.path.join(fold_dir, 'selected_features.json')
    with open(metrics_path, 'r') as handle:
        metrics_json = json.load(handle)
    checks = {'dataset': (metrics_json.get('dataset'), DATASET_NAME), 'run_id': (metrics_json.get('run_id'), expected_run_id), 'fold_id': (metrics_json.get('fold_id'), expected_fold_id), 'pipeline_id': (metrics_json.get('pipeline_id'), expected_pipeline_id), 'seed': (metrics_json.get('seed'), expected_seed)}
    mismatches = {field: {'saved': saved_value, 'expected': expected_value} for field, (saved_value, expected_value) in checks.items() if saved_value != expected_value}
    if mismatches:
        raise ValueError(f'Saved fold metadata do not match the current experiment: {mismatches}')
    val_metrics = metrics_json.get('val_metrics')
    if not isinstance(val_metrics, dict) or 'MCC' not in val_metrics:
        raise ValueError(f'Invalid or incomplete val_metrics in {metrics_path}')
    if os.path.exists(selected_features_path):
        with open(selected_features_path, 'r') as handle:
            selected_features = json.load(handle)
        n_selected_features = len(selected_features)
    else:
        n_selected_features = metrics_json.get('n_selected_features', TARGET_FEATURES)
    row = val_metrics.copy()
    row.update({'dataset': DATASET_NAME, 'automl_tool': 'TPOT', 'run_id': expected_run_id, 'fold_id': expected_fold_id, 'pipeline_id': expected_pipeline_id, 'seed': expected_seed, 'n_selected_features': n_selected_features, 'pipeline_path': pipeline_path, 'pipeline_str': metrics_json.get('pipeline_str', ''), 'resume_loaded': True})
    return row
print('=' * 88)
print('Train 100 TPOT pipelines on the Subcell random-18,097 training partition')
print('=' * 88)
print(f'Dataset:                {DATASET_NAME}')
print(f'Training file:          {TRAIN_PATH}')
print(f'Output directory:       {ROOT_SAVE_DIR}')
print(f'Repeated outer CV:      {REPEATS} repeats × {OUTER_FOLDS} folds')
print(f'Total pipelines:        {TOTAL_PIPELINES}')
print(f'Target RFE features:    {TARGET_FEATURES}')
print(f'Pipeline seeds:         {SEED_START}–{SEED_START + TOTAL_PIPELINES - 1}')
print(f'Resume mode:            {RESUME}')
print('=' * 88)
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f'Cannot find the training file:\n{TRAIN_PATH}')
train_df = pd.read_csv(TRAIN_PATH)
print('\nRaw dataframe:')
print(f'Shape: {train_df.shape}')
print(f'First columns: {train_df.columns[:10].tolist()}')
print(f'Last columns:  {train_df.columns[-10:].tolist()}')
if LABEL_COL not in train_df.columns:
    raise ValueError(f"Cannot find label column '{LABEL_COL}'. Available columns include: {train_df.columns.tolist()[:30]}")
if 'split' in train_df.columns:
    split_values = train_df['split'].astype(str).str.strip().str.lower()
    unexpected_values = sorted(set(split_values.unique()) - {'train', 'test'})
    if unexpected_values:
        raise ValueError(f'Unexpected values in the optional split column: {unexpected_values}')
    n_test_rows = int((split_values == 'test').sum())
    if n_test_rows > 0:
        print(f'\nWARNING: the supplied file contains {n_test_rows} test rows. They will be removed before training.')
    train_df = train_df.loc[split_values == 'train'].copy()
if train_df.empty:
    raise ValueError('No training rows remain after data validation.')
label_encoder = LabelEncoder()
y_train_original = train_df[LABEL_COL].astype(str).to_numpy()
y_train_full = label_encoder.fit_transform(y_train_original)
all_classes = np.arange(len(label_encoder.classes_), dtype=int)
label_mapping = {str(class_name): int(encoded_label) for encoded_label, class_name in enumerate(label_encoder.classes_)}
inverse_label_mapping = {int(encoded_label): str(class_name) for encoded_label, class_name in enumerate(label_encoder.classes_)}
atomic_pickle_dump(label_encoder, os.path.join(ROOT_SAVE_DIR, 'label_encoder.pkl'))
atomic_json_dump(label_mapping, os.path.join(ROOT_SAVE_DIR, 'label_mapping.json'))
atomic_json_dump(inverse_label_mapping, os.path.join(ROOT_SAVE_DIR, 'inverse_label_mapping.json'))
print('\nLabel mapping:')
for original_label, encoded_label in label_mapping.items():
    print(f'  {original_label} -> {encoded_label}')
metadata_columns = {LABEL_COL, 'protein_id', 'ProteinID', 'ProteinId', 'proteinID', 'partition', 'split', 'header', 'original_header', 'original_id', 'sample_index', 'index'}
drop_columns = [column for column in train_df.columns if column in metadata_columns or column.startswith('Unnamed')]
print('\nColumns excluded from the feature matrix:')
print(drop_columns)
X_train_full = train_df.drop(columns=drop_columns)
if X_train_full.columns.duplicated().any():
    duplicate_columns = X_train_full.columns[X_train_full.columns.duplicated(keep=False)].astype(str).tolist()
    raise ValueError(f'Duplicate feature-column names detected: {duplicate_columns[:20]}')
X_train_full = X_train_full.apply(pd.to_numeric, errors='coerce')
X_train_full = X_train_full.replace([np.inf, -np.inf], np.nan)
problem_value_count = int(X_train_full.isna().sum().sum())
if problem_value_count > 0:
    print(f'\nWARNING: {problem_value_count} NA/non-numeric/infinite feature values found. Filling them with 0.')
    X_train_full = X_train_full.fillna(0)
X_train_full = X_train_full.astype(np.float32)
feature_columns = X_train_full.columns.astype(str).tolist()
if len(feature_columns) < TARGET_FEATURES:
    raise ValueError(f'Only {len(feature_columns)} feature columns are available, but TARGET_FEATURES={TARGET_FEATURES}.')
atomic_json_dump(feature_columns, os.path.join(ROOT_SAVE_DIR, 'feature_columns.json'))
print('\nFinal training matrix:')
print(f'X_train_full shape: {X_train_full.shape}')
print(f'y_train_full shape: {y_train_full.shape}')
print(f'Number of classes:  {len(all_classes)}')
print(f'Feature columns:    {len(feature_columns)}')
unique_labels, class_counts = np.unique(y_train_full, return_counts=True)
class_distribution = pd.DataFrame({'original_label': label_encoder.inverse_transform(unique_labels), 'encoded_label': unique_labels, 'count': class_counts.astype(int)})
class_distribution_path = os.path.join(ROOT_SAVE_DIR, 'class_distribution.csv')
atomic_csv_dump(class_distribution, class_distribution_path)
print('\nClass distribution:')
print(class_distribution.to_string(index=False))
minimum_class_count = int(class_counts.min())
if minimum_class_count < OUTER_FOLDS:
    raise ValueError(f'The smallest class has only {minimum_class_count} samples, but OUTER_FOLDS={OUTER_FOLDS}.')
minimum_outer_training_count = minimum_class_count - int(np.ceil(minimum_class_count / OUTER_FOLDS))
inner_cv_folds = int(TPOT_CONFIG['cv'])
if minimum_outer_training_count < inner_cv_folds:
    raise ValueError(f'The rarest class would have too few samples in an outer-training fold for TPOT cv={inner_cv_folds}. Estimated minimum outer-training count: {minimum_outer_training_count}.')
class_count_manifest = {str(row.original_label): int(row.count) for row in class_distribution.itertuples(index=False)}
experiment_manifest = {'dataset': DATASET_NAME, 'train_path': os.path.abspath(TRAIN_PATH), 'train_file_size_bytes': int(os.path.getsize(TRAIN_PATH)), 'n_training_rows': int(X_train_full.shape[0]), 'n_feature_columns': int(X_train_full.shape[1]), 'feature_columns_sha256': sha256_of_strings(feature_columns), 'labels_sha256': sha256_of_strings(y_train_original), 'class_counts': class_count_manifest, 'repeats': REPEATS, 'outer_folds': OUTER_FOLDS, 'target_features': TARGET_FEATURES, 'seed_start': SEED_START, 'total_pipelines': TOTAL_PIPELINES, 'outer_split_seed_rule': 'run_id * 777', 'tpot_generations': TPOT_CONFIG['generations'], 'tpot_population_size': TPOT_CONFIG['population_size'], 'tpot_max_time_mins': TPOT_CONFIG['max_time_mins'], 'tpot_max_eval_time_mins': TPOT_CONFIG['max_eval_time_mins'], 'tpot_inner_cv': TPOT_CONFIG['cv'], 'tpot_early_stop': TPOT_CONFIG['early_stop'], 'tpot_config_dict': TPOT_CONFIG['config_dict']}
manifest_path = os.path.join(ROOT_SAVE_DIR, 'experiment_manifest.json')
existing_pipeline_count = count_existing_complete_pipelines(ROOT_SAVE_DIR)
if os.path.exists(manifest_path):
    with open(manifest_path, 'r') as handle:
        saved_manifest = json.load(handle)
    if saved_manifest != experiment_manifest:
        raise RuntimeError(f'The existing output folder belongs to a different dataset or configuration. Refusing to mix old and new pipelines.\nOutput folder: {ROOT_SAVE_DIR}\nUse a new ROOT_SAVE_DIR, or restore the matching configuration.')
elif existing_pipeline_count > 0 and STRICT_RESUME_VALIDATION:
    raise RuntimeError(f'Found {existing_pipeline_count} existing pipeline files, but no experiment_manifest.json is available to verify them. Refusing to resume because old and new experiments could be mixed.\nUse a new ROOT_SAVE_DIR, or set STRICT_RESUME_VALIDATION=False only after manually confirming every saved pipeline belongs to this exact random-18,097 experiment.')
else:
    atomic_json_dump(experiment_manifest, manifest_path)
print('\nExperiment manifest validated.')
print(f'Existing complete pipelines: {existing_pipeline_count}')
cv_summary_records = []
run_summaries = []
for run_id in range(1, REPEATS + 1):
    print('\n' + '=' * 88)
    print(f'Run {run_id}/{REPEATS}')
    print('=' * 88)
    run_dir = os.path.join(ROOT_SAVE_DIR, f'run_{run_id:02d}')
    os.makedirs(run_dir, exist_ok=True)
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=run_id * 777)
    run_cv_scores = []
    for fold_idx, (training_indices, validation_indices) in enumerate(outer_cv.split(X_train_full, y_train_full), start=1):
        pipeline_id = (run_id - 1) * OUTER_FOLDS + fold_idx
        current_seed = SEED_START + pipeline_id - 1
        fold_dir = os.path.join(run_dir, f'fold_{fold_idx:02d}')
        os.makedirs(fold_dir, exist_ok=True)
        pipeline_path = os.path.join(fold_dir, 'complete_pipeline.pkl')
        if RESUME and is_pipeline_complete(fold_dir):
            print(f'\nPipeline {pipeline_id}/{TOTAL_PIPELINES} already complete. Loading saved metrics. Run {run_id}, Fold {fold_idx}, Seed {current_seed}')
            existing_row = load_existing_metrics(fold_dir=fold_dir, expected_run_id=run_id, expected_fold_id=fold_idx, expected_pipeline_id=pipeline_id, expected_seed=current_seed)
            cv_summary_records.append(existing_row)
            run_cv_scores.append(float(existing_row['MCC']))
            continue
        print(f'\nPipeline {pipeline_id}/{TOTAL_PIPELINES} | Run {run_id} | Fold {fold_idx} | Seed {current_seed}')
        X_outer_train = X_train_full.iloc[training_indices]
        X_outer_validation = X_train_full.iloc[validation_indices]
        y_outer_train = y_train_full[training_indices]
        y_outer_validation = y_train_full[validation_indices]
        print(f'  Outer-training shape:   {X_outer_train.shape}')
        print(f'  Outer-validation shape: {X_outer_validation.shape}')
        scaler = MinMaxScaler()
        X_outer_train_scaled = scaler.fit_transform(X_outer_train).astype(np.float32)
        X_outer_validation_scaled = scaler.transform(X_outer_validation).astype(np.float32)
        rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=current_seed, n_jobs=-1), n_features_to_select=TARGET_FEATURES, step=0.1)
        X_outer_train_selected = rfe.fit_transform(X_outer_train_scaled, y_outer_train).astype(np.float32)
        X_outer_validation_selected = rfe.transform(X_outer_validation_scaled).astype(np.float32)
        selected_features = X_train_full.columns[rfe.get_support()].astype(str).tolist()
        del X_outer_train_scaled
        del X_outer_validation_scaled
        gc.collect()
        if len(selected_features) != TARGET_FEATURES:
            raise RuntimeError(f'Pipeline {pipeline_id} selected {len(selected_features)} features instead of {TARGET_FEATURES}.')
        print(f'  Selected features: {len(selected_features)}')
        tpot_classifier = TPOTClassifier(random_state=current_seed, **TPOT_CONFIG)
        try:
            tpot_classifier.fit(X_outer_train_selected, y_outer_train)
            best_model = tpot_classifier.fitted_pipeline_
        except Exception as exc:
            print(f'  ERROR: TPOT failed for pipeline {pipeline_id}.')
            print(f'  Error: {exc}')
            error_details = {'dataset': DATASET_NAME, 'run_id': run_id, 'fold_id': fold_idx, 'pipeline_id': pipeline_id, 'seed': current_seed, 'error': repr(exc), 'failed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            atomic_json_dump(error_details, os.path.join(fold_dir, 'error.json'))
            del X_outer_train_selected
            del X_outer_validation_selected
            del tpot_classifier
            gc.collect()
            continue
        model_classes = np.asarray(best_model.classes_, dtype=int)
        y_validation_pred = np.asarray(best_model.predict(X_outer_validation_selected), dtype=int)
        try:
            raw_validation_proba = best_model.predict_proba(X_outer_validation_selected)
            y_validation_proba = align_probabilities(raw_proba=raw_validation_proba, model_classes=model_classes, all_classes=all_classes)
        except Exception as probability_error:
            print(f'  Probability prediction unavailable: {probability_error}')
            y_validation_proba = None
        validation_metrics = compute_metrics_audit(y_true=y_outer_validation, y_pred=y_validation_pred, y_proba=y_validation_proba, all_classes=all_classes)
        run_cv_scores.append(validation_metrics['MCC'])
        print(f"  Validation MCC:          {validation_metrics['MCC']:.4f}")
        print(f"  Validation Accuracy:     {validation_metrics['Accuracy']:.4f}")
        print(f"  Validation Balanced Acc: {validation_metrics['Balanced_Acc']:.4f}")
        pipeline_config = {'dataset': DATASET_NAME, 'automl_tool': 'TPOT', 'run_id': run_id, 'fold_idx': fold_idx, 'pipeline_id': pipeline_id, 'seed': current_seed, 'train_path': os.path.abspath(TRAIN_PATH), 'experiment_manifest': experiment_manifest, 'feature_columns': feature_columns, 'scaler': scaler, 'rfe': rfe, 'selected_features': selected_features, 'model': best_model, 'pipeline_str': str(best_model), 'label_encoder': label_encoder, 'label_mapping': label_mapping, 'inverse_label_mapping': inverse_label_mapping, 'classes': model_classes.tolist(), 'val_metrics': validation_metrics, 'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        atomic_pickle_dump(pipeline_config, pipeline_path)
        atomic_json_dump(selected_features, os.path.join(fold_dir, 'selected_features.json'))
        metrics_json = {'dataset': DATASET_NAME, 'automl_tool': 'TPOT', 'run_id': run_id, 'fold_id': fold_idx, 'pipeline_id': pipeline_id, 'seed': current_seed, 'n_selected_features': len(selected_features), 'pipeline_path': pipeline_path, 'val_metrics': validation_metrics, 'pipeline_str': str(best_model)}
        atomic_json_dump(metrics_json, os.path.join(fold_dir, 'metrics.json'))
        error_path = os.path.join(fold_dir, 'error.json')
        if os.path.exists(error_path):
            os.remove(error_path)
        row = validation_metrics.copy()
        row.update({'dataset': DATASET_NAME, 'automl_tool': 'TPOT', 'run_id': run_id, 'fold_id': fold_idx, 'pipeline_id': pipeline_id, 'seed': current_seed, 'n_selected_features': len(selected_features), 'pipeline_path': pipeline_path, 'pipeline_str': str(best_model), 'resume_loaded': False})
        cv_summary_records.append(row)
        partial_cv_df = pd.DataFrame(cv_summary_records).sort_values(['run_id', 'fold_id'])
        atomic_csv_dump(partial_cv_df, os.path.join(ROOT_SAVE_DIR, 'nested_cv_results_partial.csv'))
        del pipeline_config
        del scaler
        del rfe
        del tpot_classifier
        del best_model
        del X_outer_train_selected
        del X_outer_validation_selected
        del y_validation_pred
        if y_validation_proba is not None:
            del y_validation_proba
        gc.collect()
    if run_cv_scores:
        run_average_mcc = float(np.mean(run_cv_scores))
        run_standard_deviation_mcc = float(np.std(run_cv_scores, ddof=1)) if len(run_cv_scores) > 1 else 0.0
    else:
        run_average_mcc = np.nan
        run_standard_deviation_mcc = np.nan
    run_summary = {'dataset': DATASET_NAME, 'automl_tool': 'TPOT', 'run_id': run_id, 'cv_mcc_mean': run_average_mcc, 'cv_mcc_std': run_standard_deviation_mcc, 'n_folds_completed_or_loaded': len(run_cv_scores), 'n_folds_expected': OUTER_FOLDS}
    run_summaries.append(run_summary)
    print('\n' + '-' * 64)
    print(f'Run {run_id} mean CV MCC: {run_average_mcc:.4f} ± {run_standard_deviation_mcc:.4f}')
    print(f'Completed or loaded folds: {len(run_cv_scores)}/{OUTER_FOLDS}')
    print('-' * 64)
    atomic_csv_dump(pd.DataFrame(run_summaries), os.path.join(ROOT_SAVE_DIR, 'run_summary_partial.csv'))
cv_results_df = pd.DataFrame(cv_summary_records).sort_values(['run_id', 'fold_id']).reset_index(drop=True)
run_summary_df = pd.DataFrame(run_summaries)
cv_result_path = os.path.join(ROOT_SAVE_DIR, 'nested_cv_results.csv')
run_summary_path = os.path.join(ROOT_SAVE_DIR, 'run_summary.csv')
atomic_csv_dump(cv_results_df, cv_result_path)
atomic_csv_dump(run_summary_df, run_summary_path)
if not cv_results_df.empty and cv_results_df['pipeline_id'].duplicated().any():
    duplicated_pipeline_ids = cv_results_df.loc[cv_results_df['pipeline_id'].duplicated(keep=False), 'pipeline_id'].astype(int).tolist()
    raise RuntimeError(f'Duplicated pipeline IDs found: {duplicated_pipeline_ids}')
completed_pipeline_ids = set(cv_results_df['pipeline_id'].astype(int)) if not cv_results_df.empty else set()
expected_pipeline_ids = set(range(1, TOTAL_PIPELINES + 1))
missing_pipeline_ids = sorted(expected_pipeline_ids - completed_pipeline_ids)
print('\n' + '=' * 88)
print('Final TPOT outer-CV summary')
print('=' * 88)
for metric in ['MCC', 'Accuracy', 'Balanced_Acc', 'F1_macro', 'F1_weighted', 'ROC_AUC_macro', 'PR_AUC_macro']:
    if metric in cv_results_df.columns:
        values = pd.to_numeric(cv_results_df[metric], errors='coerce').dropna()
        if not values.empty:
            print(f'{metric:20s}: {values.mean():.4f} ± {values.std(ddof=1):.4f}')
print('\nSaved files:')
print(f'  CV results:          {cv_result_path}')
print(f'  Run summary:         {run_summary_path}')
print(f'  Class distribution:  {class_distribution_path}')
print(f"  Feature columns:     {os.path.join(ROOT_SAVE_DIR, 'feature_columns.json')}")
print(f'  Experiment manifest: {manifest_path}')
print(f"  Label encoder:       {os.path.join(ROOT_SAVE_DIR, 'label_encoder.pkl')}")
print(f"  Label mapping:       {os.path.join(ROOT_SAVE_DIR, 'label_mapping.json')}")
print(f'  Pipelines:           {ROOT_SAVE_DIR}/run_xx/fold_xx/complete_pipeline.pkl')
print(f'\nCompleted pipelines: {len(completed_pipeline_ids)}/{TOTAL_PIPELINES}')
if missing_pipeline_ids:
    missing_path = os.path.join(ROOT_SAVE_DIR, 'missing_pipeline_ids.json')
    atomic_json_dump(missing_pipeline_ids, missing_path)
    raise RuntimeError(f'The experiment is not complete. Missing pipeline IDs: {missing_pipeline_ids}. All completed models and summaries have been saved. Run the same script again with RESUME=True.')
missing_path = os.path.join(ROOT_SAVE_DIR, 'missing_pipeline_ids.json')
if os.path.exists(missing_path):
    os.remove(missing_path)
print('\nAll 100 TPOT pipelines completed successfully.')
