import gc
import json
import os
import pickle
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import tpot
try:
    import xgboost
except ImportError:
    pass
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
warnings.filterwarnings('ignore')
TEST_PATH = os.environ.get('SUBCELL_TEST_CSV', 'data/features/subcell/random/subcell_random18097_test_features_with_labels.csv')
PIPELINE_ROOT = os.environ.get('TPOT_PIPELINE_DIR', 'runs/subcell_random18097_tpot_100_pipelines_seed100_resume')
OUTPUT_DIR = os.environ.get('TPOT_EVALUATION_DIR', os.path.join(PIPELINE_ROOT, 'independent_test_evaluation'))
LABEL_COL = 'label'
EXPECTED_PIPELINES = 100
SAVE_INDIVIDUAL_PREDICTIONS = True
USE_FAST_SELECTED_FEATURE_TRANSFORM = True
ID_COLUMN_CANDIDATES = ['protein_id', 'ProteinID', 'ProteinId', 'proteinID', 'header', 'original_header', 'original_id']

def load_json(path):
    with open(path, 'r') as handle:
        return json.load(handle)

def save_json(data, path):
    with open(path, 'w') as handle:
        json.dump(data, handle, indent=2)

def parse_run_fold(path):
    text = str(path)
    run_match = re.search('run_(\\d+)', text)
    fold_match = re.search('fold_(\\d+)', text)
    return (int(run_match.group(1)) if run_match else -1, int(fold_match.group(1)) if fold_match else -1)

def discover_pipelines(root):
    paths = list(Path(root).glob('run_*/fold_*/complete_pipeline.pkl'))
    return sorted(paths, key=parse_run_fold)

def extract_model(config):
    if not isinstance(config, dict):
        if hasattr(config, 'predict'):
            return config
        raise TypeError('The pickle is neither a config dictionary nor a predictor.')
    for key in ('model', 'pipeline', 'fitted_pipeline', 'best_pipeline'):
        candidate = config.get(key)
        if candidate is not None and hasattr(candidate, 'predict'):
            return candidate
    tpot_object = config.get('tpot') or config.get('tpot_model')
    if tpot_object is not None:
        candidate = getattr(tpot_object, 'fitted_pipeline_', None)
        if candidate is not None and hasattr(candidate, 'predict'):
            return candidate
    raise KeyError('No fitted TPOT predictor found. Expected one of: model, pipeline, fitted_pipeline, best_pipeline, or tpot.fitted_pipeline_.')

def align_probabilities(raw_proba, model_classes, all_classes):
    raw_proba = np.asarray(raw_proba, dtype=np.float64)
    if raw_proba.ndim != 2:
        raise ValueError(f'predict_proba returned shape {raw_proba.shape}')
    model_classes = np.asarray(model_classes, dtype=int)
    aligned = np.zeros((raw_proba.shape[0], len(all_classes)), dtype=np.float64)
    target = {int(value): i for i, value in enumerate(all_classes)}
    for source_col, class_id in enumerate(model_classes):
        if int(class_id) not in target:
            raise ValueError(f'Unknown predicted class: {class_id}')
        aligned[:, target[int(class_id)]] = raw_proba[:, source_col]
    row_sums = aligned.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-05):
        raise ValueError('Aligned class probabilities do not sum to one.')
    return aligned

def selected_feature_transform(test_features, feature_columns, selected_features, scaler, rfe):
    support = np.asarray(rfe.get_support(), dtype=bool)
    if support.size != len(feature_columns):
        raise ValueError(f'RFE support has {support.size} entries but training schema has {len(feature_columns)} features.')
    canonical = np.asarray(feature_columns, dtype=object)[support].tolist()
    if list(selected_features) != canonical:
        raise ValueError('selected_features disagrees with the saved RFE support mask.')
    if USE_FAST_SELECTED_FEATURE_TRANSFORM and hasattr(scaler, 'scale_') and hasattr(scaler, 'min_'):
        raw = test_features.loc[:, canonical].to_numpy(dtype=np.float32, copy=True)
        indices = np.flatnonzero(support)
        raw *= np.asarray(scaler.scale_, dtype=np.float32)[indices]
        raw += np.asarray(scaler.min_, dtype=np.float32)[indices]
        return raw
    raw_all = test_features.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
    return rfe.transform(scaler.transform(raw_all)).astype(np.float32)

def global_metrics(y_true, y_pred, y_proba, classes):
    result = {'MCC': float(matthews_corrcoef(y_true, y_pred)), 'Accuracy': float(accuracy_score(y_true, y_pred)), 'Balanced_Acc': float(balanced_accuracy_score(y_true, y_pred))}
    for average in ('micro', 'macro', 'weighted'):
        result[f'Precision_{average}'] = float(precision_score(y_true, y_pred, labels=classes, average=average, zero_division=0))
        result[f'Recall_{average}'] = float(recall_score(y_true, y_pred, labels=classes, average=average, zero_division=0))
        result[f'F1_{average}'] = float(f1_score(y_true, y_pred, labels=classes, average=average, zero_division=0))
    if y_proba is not None:
        binary = label_binarize(y_true, classes=classes)
        try:
            result['ROC_AUC_macro'] = float(roc_auc_score(binary, y_proba, multi_class='ovr', average='macro'))
            result['ROC_AUC_weighted'] = float(roc_auc_score(binary, y_proba, multi_class='ovr', average='weighted'))
            result['PR_AUC_macro'] = float(average_precision_score(binary, y_proba, average='macro'))
            result['PR_AUC_weighted'] = float(average_precision_score(binary, y_proba, average='weighted'))
        except ValueError as exc:
            result['probability_metric_error'] = str(exc)
    return result

def per_class_metrics(y_true, y_pred, y_proba, classes, encoder):
    rows = []
    for probability_col, class_id in enumerate(classes):
        true_binary = (y_true == class_id).astype(np.int8)
        pred_binary = (y_pred == class_id).astype(np.int8)
        row = {'encoded_label': int(class_id), 'class_name': str(encoder.inverse_transform([class_id])[0]), 'support': int(true_binary.sum()), 'predicted_count': int(pred_binary.sum()), 'MCC_ovr': float(matthews_corrcoef(true_binary, pred_binary)), 'Precision': float(precision_score(true_binary, pred_binary, zero_division=0)), 'Recall': float(recall_score(true_binary, pred_binary, zero_division=0)), 'F1': float(f1_score(true_binary, pred_binary, zero_division=0))}
        if y_proba is not None:
            scores = y_proba[:, probability_col]
            row['ROC_AUC'] = float(roc_auc_score(true_binary, scores)) if np.unique(true_binary).size == 2 else np.nan
            row['PR_AUC'] = float(average_precision_score(true_binary, scores))
        rows.append(row)
    return rows

def numeric_summary(frame, columns):
    rows = []
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors='coerce').dropna()
        if values.empty:
            continue
        rows.append({'metric': column, 'n_pipelines': int(values.size), 'mean': float(values.mean()), 'std': float(values.std(ddof=1)), 'median': float(values.median()), 'min': float(values.min()), 'max': float(values.max())})
    return pd.DataFrame(rows)
if not os.path.isfile(TEST_PATH):
    raise FileNotFoundError(f'Test CSV not found: {TEST_PATH}')
if not os.path.isdir(PIPELINE_ROOT):
    raise FileNotFoundError(f'Pipeline root not found: {PIPELINE_ROOT}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
prediction_dir = os.path.join(OUTPUT_DIR, 'predictions')
if SAVE_INDIVIDUAL_PREDICTIONS:
    os.makedirs(prediction_dir, exist_ok=True)
pipeline_paths = discover_pipelines(PIPELINE_ROOT)
if not pipeline_paths:
    raise FileNotFoundError(f'No run_*/fold_*/complete_pipeline.pkl found below {PIPELINE_ROOT}')
print(f'Found {len(pipeline_paths)} saved TPOT pipelines.')
if len(pipeline_paths) != EXPECTED_PIPELINES:
    print(f'WARNING: expected {EXPECTED_PIPELINES}, found {len(pipeline_paths)}.')
feature_columns = load_json(os.path.join(PIPELINE_ROOT, 'feature_columns.json'))
with open(os.path.join(PIPELINE_ROOT, 'label_encoder.pkl'), 'rb') as handle:
    label_encoder = pickle.load(handle)
classes = np.arange(len(label_encoder.classes_), dtype=int)
feature_set = set(feature_columns)
pipeline_metadata = []
selected_union = set()
for pipeline_path in pipeline_paths:
    run_id, fold_id = parse_run_fold(pipeline_path)
    selected_path = pipeline_path.parent / 'selected_features.json'
    if selected_path.exists():
        selected = load_json(selected_path)
    else:
        with open(pipeline_path, 'rb') as handle:
            saved = pickle.load(handle)
        if not isinstance(saved, dict) or 'selected_features' not in saved:
            raise KeyError(f'No selected_features in {pipeline_path}')
        selected = saved['selected_features']
        del saved
    unknown = [name for name in selected if name not in feature_set]
    if unknown:
        raise ValueError(f'Unknown selected features in {pipeline_path}: {unknown[:10]}')
    selected_union.update(selected)
    pipeline_metadata.append({'run_id': run_id, 'fold_id': fold_id, 'n_selected_features': len(selected), 'pipeline_path': str(pipeline_path)})
selected_union = [name for name in feature_columns if name in selected_union]
pd.DataFrame(pipeline_metadata).to_csv(os.path.join(OUTPUT_DIR, 'pipeline_manifest.csv'), index=False)
save_json(selected_union, os.path.join(OUTPUT_DIR, 'selected_feature_union.json'))
header = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()
if LABEL_COL not in header:
    raise ValueError(f"Test CSV has no '{LABEL_COL}' column.")
missing = [name for name in selected_union if name not in header]
if missing:
    raise ValueError(f'Test CSV is missing selected features: {missing[:20]}')
id_col = next((name for name in ID_COLUMN_CANDIDATES if name in header), None)
usecols = list(dict.fromkeys([LABEL_COL] + selected_union + ([id_col] if id_col else [])))
test_df = pd.read_csv(TEST_PATH, usecols=usecols)
test_ids = test_df[id_col].astype(str).to_numpy() if id_col else np.arange(len(test_df)).astype(str)
id_output_name = id_col or 'sample_index'
original_labels = test_df[LABEL_COL].astype(str).to_numpy()
known_labels = set(label_encoder.classes_.astype(str))
unseen = sorted(set(original_labels) - known_labels)
if unseen:
    raise ValueError(f'Labels absent from the training label encoder: {unseen}')
y_test = label_encoder.transform(original_labels)
test_features = test_df.loc[:, selected_union].apply(pd.to_numeric, errors='coerce')
test_features = test_features.replace([np.inf, -np.inf], np.nan)
bad_values = int(test_features.isna().sum().sum())
if bad_values:
    print(f'WARNING: replacing {bad_values} invalid values with 0.')
    test_features = test_features.fillna(0)
test_features = test_features.astype(np.float32)
del test_df
gc.collect()
global_rows, class_rows, failures = ([], [], [])
n_samples, n_classes = (len(y_test), len(classes))
votes = np.zeros((n_samples, n_classes), dtype=np.int16)
probability_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
n_probability_models = 0
for sequence, pipeline_path in enumerate(pipeline_paths, 1):
    path_run, path_fold = parse_run_fold(pipeline_path)
    print(f'[{sequence}/{len(pipeline_paths)}] run={path_run}, fold={path_fold}')
    try:
        with open(pipeline_path, 'rb') as handle:
            saved = pickle.load(handle)
        if not isinstance(saved, dict):
            raise TypeError('complete_pipeline.pkl must contain scaler, rfe and selected_features.')
        scaler, rfe = (saved['scaler'], saved['rfe'])
        selected = saved['selected_features']
        model = extract_model(saved)
        saved_schema = saved.get('feature_columns', feature_columns)
        if list(saved_schema) != list(feature_columns):
            raise ValueError('Pipeline feature schema differs from feature_columns.json.')
        run_id = int(saved.get('run_id', path_run))
        fold_id = int(saved.get('fold_idx', saved.get('fold_id', path_fold)))
        pipeline_id = int(saved.get('pipeline_id', (run_id - 1) * 10 + fold_id))
        seed = saved.get('seed', np.nan)
        X_test = selected_feature_transform(test_features, feature_columns, selected, scaler, rfe)
        y_pred = np.asarray(model.predict(X_test), dtype=int)
        if not set(np.unique(y_pred)).issubset(set(classes)):
            raise ValueError(f'Model returned invalid encoded labels: {np.unique(y_pred)}')
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                raw_proba = model.predict_proba(X_test)
                model_classes = getattr(model, 'classes_', saved.get('classes', classes))
                y_proba = align_probabilities(raw_proba, model_classes, classes)
            except Exception as exc:
                print(f'  probability metrics unavailable: {exc}')
        metrics = global_metrics(y_test, y_pred, y_proba, classes)
        global_rows.append({'run_id': run_id, 'fold_id': fold_id, 'pipeline_id': pipeline_id, 'seed': seed, 'n_selected_features': X_test.shape[1], 'pipeline_path': str(pipeline_path), **metrics})
        for row in per_class_metrics(y_test, y_pred, y_proba, classes, label_encoder):
            row.update({'run_id': run_id, 'fold_id': fold_id, 'pipeline_id': pipeline_id, 'seed': seed})
            class_rows.append(row)
        votes[np.arange(n_samples), y_pred] += 1
        if y_proba is not None:
            probability_sum += y_proba
            n_probability_models += 1
        if SAVE_INDIVIDUAL_PREDICTIONS:
            prediction = pd.DataFrame({id_output_name: test_ids, 'true_encoded': y_test, 'true_label': original_labels, 'pred_encoded': y_pred, 'pred_label': label_encoder.inverse_transform(y_pred)})
            if y_proba is not None:
                prediction['pred_confidence'] = y_proba.max(axis=1)
            prediction.to_csv(os.path.join(prediction_dir, f'pipeline_{pipeline_id:03d}_predictions.csv.gz'), index=False, compression='gzip')
        pd.DataFrame(global_rows).to_csv(os.path.join(OUTPUT_DIR, 'per_pipeline_test_metrics_partial.csv'), index=False)
        print(f"  MCC={metrics['MCC']:.4f}; macro-F1={metrics['F1_macro']:.4f}")
        del saved, scaler, rfe, model, X_test, y_pred
        gc.collect()
    except Exception as exc:
        print(f'  FAILED: {exc!r}')
        failures.append({'run_id': path_run, 'fold_id': path_fold, 'pipeline_path': str(pipeline_path), 'error': repr(exc)})
        gc.collect()
if not global_rows:
    pd.DataFrame(failures).to_csv(os.path.join(OUTPUT_DIR, 'failed_pipelines.csv'), index=False)
    raise RuntimeError('All saved TPOT pipelines failed; inspect failed_pipelines.csv.')
metric_columns = ['MCC', 'Accuracy', 'Balanced_Acc', 'Precision_micro', 'Precision_macro', 'Precision_weighted', 'Recall_micro', 'Recall_macro', 'Recall_weighted', 'F1_micro', 'F1_macro', 'F1_weighted', 'ROC_AUC_macro', 'ROC_AUC_weighted', 'PR_AUC_macro', 'PR_AUC_weighted']
global_df = pd.DataFrame(global_rows).sort_values(['run_id', 'fold_id'])
class_df = pd.DataFrame(class_rows).sort_values(['class_name', 'run_id', 'fold_id'])
global_metrics_path = os.path.join(OUTPUT_DIR, 'per_pipeline_test_metrics.csv')
per_class_metrics_path = os.path.join(OUTPUT_DIR, 'per_pipeline_per_class_metrics.csv')
metric_summary_path = os.path.join(OUTPUT_DIR, 'test_metric_summary.csv')
run_level_summary_path = os.path.join(OUTPUT_DIR, 'run_level_test_summary.csv')
per_class_summary_path = os.path.join(OUTPUT_DIR, 'per_class_test_summary.csv')
global_df.to_csv(global_metrics_path, index=False)
class_df.to_csv(per_class_metrics_path, index=False)
metric_summary_df = numeric_summary(global_df, metric_columns)
metric_summary_df.to_csv(metric_summary_path, index=False)
available_global_metrics = [x for x in metric_columns if x in global_df.columns]
run_level_summary_df = global_df.groupby('run_id', as_index=False)[available_global_metrics].agg(['mean', 'std'])
run_level_summary_df.columns = ['_'.join((str(part) for part in column if str(part))) for column in run_level_summary_df.columns.to_flat_index()]
run_level_summary_df.to_csv(run_level_summary_path, index=False)
per_class_measure_columns = ['MCC_ovr', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC']
for column in per_class_measure_columns:
    if column not in class_df.columns:
        class_df[column] = np.nan
class_df.to_csv(per_class_metrics_path, index=False)
per_class_summary_df = class_df.groupby('class_name', as_index=False)[per_class_measure_columns].agg(['mean', 'std', 'median', 'min', 'max'])
per_class_summary_df.columns = ['_'.join((str(part) for part in column if str(part))) for column in per_class_summary_df.columns.to_flat_index()]
per_class_summary_df.to_csv(per_class_summary_path, index=False)
ensemble_rows = []
ensemble_class_rows = []
hard_pred = votes.argmax(axis=1).astype(int)
hard_proba = votes.astype(np.float64) / float(len(global_rows))
hard_metrics = global_metrics(y_test, hard_pred, hard_proba, classes)
ensemble_rows.append({'ensemble': 'hard_vote_all_successful', 'n_models': len(global_rows), **hard_metrics})
for row in per_class_metrics(y_test, hard_pred, hard_proba, classes, label_encoder):
    row['ensemble'] = 'hard_vote_all_successful'
    row['n_models'] = len(global_rows)
    ensemble_class_rows.append(row)
soft_pred = None
mean_proba = None
if n_probability_models:
    mean_proba = probability_sum / n_probability_models
    soft_pred = mean_proba.argmax(axis=1)
    soft_metrics = global_metrics(y_test, soft_pred, mean_proba, classes)
    ensemble_rows.append({'ensemble': 'soft_vote_probability_models', 'n_models': n_probability_models, **soft_metrics})
    for row in per_class_metrics(y_test, soft_pred, mean_proba, classes, label_encoder):
        row['ensemble'] = 'soft_vote_probability_models'
        row['n_models'] = n_probability_models
        ensemble_class_rows.append(row)
ensemble_metrics_path = os.path.join(OUTPUT_DIR, 'ensemble_test_metrics.csv')
ensemble_per_class_path = os.path.join(OUTPUT_DIR, 'ensemble_per_class_metrics.csv')
pd.DataFrame(ensemble_rows).to_csv(ensemble_metrics_path, index=False)
pd.DataFrame(ensemble_class_rows).to_csv(ensemble_per_class_path, index=False)
ensemble_predictions = pd.DataFrame({id_output_name: test_ids, 'true_encoded': y_test, 'true_label': original_labels, 'hard_vote_encoded': hard_pred, 'hard_vote_label': label_encoder.inverse_transform(hard_pred), 'hard_vote_confidence': hard_proba.max(axis=1), 'hard_vote_agreement_count': votes.max(axis=1), 'hard_vote_model_count': len(global_rows)})
if soft_pred is not None:
    ensemble_predictions['soft_vote_encoded'] = soft_pred
    ensemble_predictions['soft_vote_label'] = label_encoder.inverse_transform(soft_pred)
    ensemble_predictions['soft_vote_confidence'] = mean_proba.max(axis=1)
ensemble_predictions.to_csv(os.path.join(OUTPUT_DIR, 'ensemble_predictions.csv'), index=False)
class_names = label_encoder.classes_.astype(str)
pd.DataFrame(confusion_matrix(y_test, hard_pred, labels=classes), index=[f'true::{x}' for x in class_names], columns=[f'pred::{x}' for x in class_names]).to_csv(os.path.join(OUTPUT_DIR, 'hard_vote_confusion_matrix.csv'))
if soft_pred is not None:
    pd.DataFrame(confusion_matrix(y_test, soft_pred, labels=classes), index=[f'true::{x}' for x in class_names], columns=[f'pred::{x}' for x in class_names]).to_csv(os.path.join(OUTPUT_DIR, 'soft_vote_confusion_matrix.csv'))
if failures:
    pd.DataFrame(failures).to_csv(os.path.join(OUTPUT_DIR, 'failed_pipelines.csv'), index=False)
print('\nTEST EVALUATION COMPLETE')
print(f'Successful pipelines: {len(global_rows)}/{len(pipeline_paths)}')
print(f'Probability pipelines: {n_probability_models}/{len(global_rows)}')
print(f'Outputs: {OUTPUT_DIR}')
print('\nOverall metric summary across individual pipelines:')
print(metric_summary_df.to_string(index=False))
print('IMPORTANT: report the distribution/summary of test performance.')
print('Do not identify a best pipeline by its test MCC.')
