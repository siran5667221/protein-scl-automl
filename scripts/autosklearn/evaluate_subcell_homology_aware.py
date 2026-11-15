import os
import re
import gc
import json
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import autosklearn.classification
from autosklearn.ensembles.singlebest_ensemble import SingleBest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
warnings.filterwarnings('ignore')
TEST_PATH = os.environ.get('SUBCELL_TEST_CSV', 'data/features/subcell/homology_aware/subcell_graphpart17100_test_features_with_labels.csv')
PIPELINE_ROOT = os.environ.get('AUTOSKLEARN_PIPELINE_DIR', 'runs/subcell_graphpart17100_homology_autosklearn_100_pipelines_seed100_resume')
OUTPUT_DIR = os.environ.get('AUTOSKLEARN_EVALUATION_DIR', os.path.join(PIPELINE_ROOT, 'independent_test_evaluation'))
LABEL_COL = 'label'
ID_COLUMN_CANDIDATES = ['protein_id', 'ProteinID', 'ProteinId', 'proteinID', 'header', 'original_header', 'original_id']
SAVE_INDIVIDUAL_PREDICTIONS = True
EXPECTED_PIPELINES = 100
USE_FAST_SELECTED_FEATURE_TRANSFORM = True

def parse_run_fold_from_path(path):
    text = str(path)
    run_match = re.search('run_(\\d+)', text)
    fold_match = re.search('fold_(\\d+)', text)
    run_id = int(run_match.group(1)) if run_match else -1
    fold_id = int(fold_match.group(1)) if fold_match else -1
    return (run_id, fold_id)

def discover_pipeline_paths(root_dir):
    paths = list(Path(root_dir).glob('run_*/fold_*/complete_pipeline.pkl'))
    return sorted(paths, key=lambda p: parse_run_fold_from_path(p))

def load_json(path):
    with open(path, 'r') as handle:
        return json.load(handle)

def save_json(data, path):
    with open(path, 'w') as handle:
        json.dump(data, handle, indent=4)

def align_probabilities(raw_proba, model_classes, all_classes):
    if raw_proba is None:
        return None
    raw_proba = np.asarray(raw_proba, dtype=np.float64)
    model_classes = np.asarray(model_classes)
    all_classes = np.asarray(all_classes)
    aligned = np.zeros((raw_proba.shape[0], len(all_classes)), dtype=np.float64)
    all_class_to_col = {int(c): i for i, c in enumerate(all_classes)}
    for source_col, class_value in enumerate(model_classes):
        class_value = int(class_value)
        if class_value not in all_class_to_col:
            raise ValueError(f'Model class {class_value} is absent from the global class set.')
        aligned[:, all_class_to_col[class_value]] = raw_proba[:, source_col]
    return aligned

def compute_global_metrics(y_true, y_pred, y_proba, all_classes):
    all_classes = np.asarray(all_classes)
    metrics = {'MCC': float(matthews_corrcoef(y_true, y_pred)), 'Accuracy': float(accuracy_score(y_true, y_pred)), 'Balanced_Acc': float(balanced_accuracy_score(y_true, y_pred))}
    for avg in ['micro', 'macro', 'weighted']:
        metrics[f'Precision_{avg}'] = float(precision_score(y_true, y_pred, labels=all_classes, average=avg, zero_division=0))
        metrics[f'Recall_{avg}'] = float(recall_score(y_true, y_pred, labels=all_classes, average=avg, zero_division=0))
        metrics[f'F1_{avg}'] = float(f1_score(y_true, y_pred, labels=all_classes, average=avg, zero_division=0))
    if y_proba is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=all_classes)
            if len(all_classes) == 2:
                metrics['ROC_AUC_macro'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                metrics['PR_AUC_macro'] = float(average_precision_score(y_true, y_proba[:, 1]))
            else:
                metrics['ROC_AUC_macro'] = float(roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='macro'))
                metrics['ROC_AUC_weighted'] = float(roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='weighted'))
                metrics['PR_AUC_macro'] = float(average_precision_score(y_true_bin, y_proba, average='macro'))
                metrics['PR_AUC_weighted'] = float(average_precision_score(y_true_bin, y_proba, average='weighted'))
        except Exception as exc:
            metrics['AUC_Error'] = str(exc)
    return metrics

def compute_per_class_metrics(y_true, y_pred, y_proba, all_classes, label_encoder):
    records = []
    for proba_col, class_id in enumerate(all_classes):
        class_id = int(class_id)
        class_name = str(label_encoder.inverse_transform([class_id])[0])
        y_true_binary = (y_true == class_id).astype(np.int8)
        y_pred_binary = (y_pred == class_id).astype(np.int8)
        row = {'encoded_label': class_id, 'class_name': class_name, 'support': int(y_true_binary.sum()), 'predicted_count': int(y_pred_binary.sum()), 'MCC_ovr': float(matthews_corrcoef(y_true_binary, y_pred_binary)), 'Precision': float(precision_score(y_true_binary, y_pred_binary, zero_division=0)), 'Recall': float(recall_score(y_true_binary, y_pred_binary, zero_division=0)), 'F1': float(f1_score(y_true_binary, y_pred_binary, zero_division=0))}
        if y_proba is not None:
            class_probability = y_proba[:, proba_col]
            try:
                if len(np.unique(y_true_binary)) == 2:
                    row['ROC_AUC'] = float(roc_auc_score(y_true_binary, class_probability))
                else:
                    row['ROC_AUC'] = np.nan
            except Exception:
                row['ROC_AUC'] = np.nan
            try:
                row['PR_AUC'] = float(average_precision_score(y_true_binary, class_probability))
            except Exception:
                row['PR_AUC'] = np.nan
        records.append(row)
    return records

def selected_feature_transform(test_feature_df, feature_columns, selected_features, scaler, rfe):
    support = np.asarray(rfe.get_support(), dtype=bool)
    if len(support) != len(feature_columns):
        raise ValueError(f'RFE support length does not match the saved feature-column count: {len(support)} versus {len(feature_columns)}.')
    canonical_selected = np.asarray(feature_columns, dtype=object)[support].tolist()
    if canonical_selected != list(selected_features):
        raise ValueError('Saved selected_features do not match feature_columns[rfe support]. The pipeline file may be inconsistent.')
    raw_selected = test_feature_df.loc[:, canonical_selected].to_numpy(dtype=np.float32, copy=True)
    if USE_FAST_SELECTED_FEATURE_TRANSFORM and hasattr(scaler, 'scale_') and hasattr(scaler, 'min_'):
        selected_indices = np.flatnonzero(support)
        scale = np.asarray(scaler.scale_, dtype=np.float32)[selected_indices]
        offset = np.asarray(scaler.min_, dtype=np.float32)[selected_indices]
        transformed = raw_selected * scale
        transformed += offset
        return transformed.astype(np.float32, copy=False)
    missing = [feature for feature in feature_columns if feature not in test_feature_df.columns]
    if missing:
        raise ValueError(f'The slow fallback requires all original features, but some are not loaded. First missing columns: {missing[:10]}')
    all_raw = test_feature_df.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
    all_scaled = scaler.transform(all_raw).astype(np.float32)
    transformed = rfe.transform(all_scaled).astype(np.float32)
    del all_raw, all_scaled
    gc.collect()
    return transformed

def summarise_numeric_metrics(metric_df, metric_columns):
    records = []
    for metric in metric_columns:
        if metric not in metric_df.columns:
            continue
        values = pd.to_numeric(metric_df[metric], errors='coerce').dropna()
        if values.empty:
            continue
        records.append({'metric': metric, 'n_pipelines': int(values.shape[0]), 'mean': float(values.mean()), 'std': float(values.std(ddof=1)), 'median': float(values.median()), 'min': float(values.min()), 'max': float(values.max())})
    return pd.DataFrame(records)
os.makedirs(OUTPUT_DIR, exist_ok=True)
prediction_dir = os.path.join(OUTPUT_DIR, 'predictions')
if SAVE_INDIVIDUAL_PREDICTIONS:
    os.makedirs(prediction_dir, exist_ok=True)
if not os.path.exists(TEST_PATH):
    parent = os.path.dirname(TEST_PATH)
    candidates = []
    if os.path.isdir(parent):
        candidates = sorted((str(path) for path in Path(parent).glob('*test*features*labels*.csv')))
    candidate_text = '\n'.join(candidates) if candidates else '(none found)'
    raise FileNotFoundError(f'Cannot find TEST_PATH:\n{TEST_PATH}\n\nPossible test files in the same folder:\n{candidate_text}')
if not os.path.isdir(PIPELINE_ROOT):
    raise FileNotFoundError(f'Cannot find PIPELINE_ROOT:\n{PIPELINE_ROOT}')
pipeline_paths = discover_pipeline_paths(PIPELINE_ROOT)
if not pipeline_paths:
    raise FileNotFoundError(f'No complete_pipeline.pkl files were found under:\n{PIPELINE_ROOT}/run_*/fold_*/')
print('=' * 88)
print('Independent test evaluation of saved Auto-sklearn pipelines')
print('=' * 88)
print(f'Test file:       {TEST_PATH}')
print(f'Pipeline root:   {PIPELINE_ROOT}')
print(f'Output folder:   {OUTPUT_DIR}')
print(f'Pipelines found: {len(pipeline_paths)}')
if len(pipeline_paths) != EXPECTED_PIPELINES:
    print(f'WARNING: expected {EXPECTED_PIPELINES} pipelines, but found {len(pipeline_paths)}.')
feature_columns_path = os.path.join(PIPELINE_ROOT, 'feature_columns.json')
label_encoder_path = os.path.join(PIPELINE_ROOT, 'label_encoder.pkl')
if not os.path.exists(feature_columns_path):
    raise FileNotFoundError(f'Missing shared feature-column file: {feature_columns_path}')
if not os.path.exists(label_encoder_path):
    raise FileNotFoundError(f'Missing shared label encoder: {label_encoder_path}')
feature_columns = load_json(feature_columns_path)
with open(label_encoder_path, 'rb') as handle:
    label_encoder = pickle.load(handle)
all_classes = np.arange(len(label_encoder.classes_), dtype=int)
feature_index = {feature_name: index for index, feature_name in enumerate(feature_columns)}
print(f'Original feature columns: {len(feature_columns)}')
print(f'Encoded classes:          {len(all_classes)}')
print('Class order:')
for class_id, class_name in enumerate(label_encoder.classes_):
    print(f'  {class_id:2d} -> {class_name}')
selected_feature_union = set()
selected_feature_manifest = []
manifest_failures = []
for pipeline_path in pipeline_paths:
    run_id, fold_id = parse_run_fold_from_path(pipeline_path)
    selected_json = pipeline_path.parent / 'selected_features.json'
    try:
        if selected_json.exists():
            selected_features = load_json(selected_json)
        else:
            with open(pipeline_path, 'rb') as handle:
                config = pickle.load(handle)
            selected_features = config['selected_features']
            del config
            gc.collect()
        missing_from_training_schema = [feature for feature in selected_features if feature not in feature_index]
        if missing_from_training_schema:
            raise ValueError(f'Selected features absent from shared feature_columns.json: {missing_from_training_schema[:10]}')
        selected_feature_union.update(selected_features)
        selected_feature_manifest.append({'run_id': run_id, 'fold_id': fold_id, 'pipeline_path': str(pipeline_path), 'n_selected_features': len(selected_features)})
    except Exception as exc:
        manifest_failures.append({'run_id': run_id, 'fold_id': fold_id, 'pipeline_path': str(pipeline_path), 'stage': 'selected_feature_manifest', 'error': repr(exc)})
if manifest_failures:
    failure_path = os.path.join(OUTPUT_DIR, 'manifest_failures.csv')
    pd.DataFrame(manifest_failures).to_csv(failure_path, index=False)
    raise RuntimeError(f'Could not build the selected-feature manifest for every pipeline. Details saved to: {failure_path}')
selected_feature_union = [feature for feature in feature_columns if feature in selected_feature_union]
pd.DataFrame(selected_feature_manifest).to_csv(os.path.join(OUTPUT_DIR, 'pipeline_manifest.csv'), index=False)
save_json(selected_feature_union, os.path.join(OUTPUT_DIR, 'selected_feature_union.json'))
print(f'Unique features selected across all pipelines: {len(selected_feature_union)}')
test_header = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()
if LABEL_COL not in test_header:
    raise ValueError(f"Test file does not contain label column '{LABEL_COL}'.")
id_col = next((column for column in ID_COLUMN_CANDIDATES if column in test_header), None)
missing_test_features = [feature for feature in selected_feature_union if feature not in test_header]
if missing_test_features:
    raise ValueError(f'The test CSV is missing {len(missing_test_features)} selected features. First missing columns:\n{missing_test_features[:20]}')
usecols = [LABEL_COL] + selected_feature_union
if id_col is not None:
    usecols.append(id_col)
usecols = list(dict.fromkeys(usecols))
print(f'Reading {len(usecols)} necessary columns from the test CSV...')
test_df = pd.read_csv(TEST_PATH, usecols=usecols)
if id_col is None:
    test_ids = np.arange(test_df.shape[0]).astype(str)
    id_output_name = 'sample_index'
else:
    test_ids = test_df[id_col].astype(str).to_numpy()
    id_output_name = id_col
y_test_original = test_df[LABEL_COL].astype(str).to_numpy()
unseen_labels = sorted(set(y_test_original) - set(label_encoder.classes_.astype(str)))
if unseen_labels:
    raise ValueError(f'The test set contains labels unseen during training:\n{unseen_labels}')
y_test = label_encoder.transform(y_test_original)
test_feature_df = test_df.loc[:, selected_feature_union].apply(pd.to_numeric, errors='coerce')
n_na = int(test_feature_df.isna().sum().sum())
n_inf = int(np.isinf(test_feature_df.to_numpy(dtype=np.float64, copy=False)).sum())
if n_inf > 0:
    print(f'WARNING: found {n_inf} infinite feature values; replacing with NA.')
    test_feature_df = test_feature_df.replace([np.inf, -np.inf], np.nan)
if n_na > 0 or n_inf > 0:
    total_problem_values = int(test_feature_df.isna().sum().sum())
    print(f"WARNING: replacing {total_problem_values} non-numeric/NA/inf feature values with 0, matching the training script's NA policy.")
    test_feature_df = test_feature_df.fillna(0)
test_feature_df = test_feature_df.astype(np.float32)
print(f'Test samples: {test_df.shape[0]}')
print('Test class distribution:')
print(pd.Series(y_test_original).value_counts().sort_index().to_string())
del test_df
gc.collect()
global_metric_records = []
per_class_records = []
failure_records = []
n_samples = len(y_test)
n_classes = len(all_classes)
vote_counts = np.zeros((n_samples, n_classes), dtype=np.int16)
probability_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
n_probability_models = 0
n_successful_models = 0
for sequence_number, pipeline_path in enumerate(pipeline_paths, start=1):
    path_run_id, path_fold_id = parse_run_fold_from_path(pipeline_path)
    print('\n' + '-' * 88)
    print(f'Pipeline {sequence_number}/{len(pipeline_paths)} | run={path_run_id}, fold={path_fold_id}')
    print(str(pipeline_path))
    try:
        with open(pipeline_path, 'rb') as handle:
            config = pickle.load(handle)
        scaler = config['scaler']
        rfe = config['rfe']
        model = config['model']
        pipeline_feature_columns = config.get('feature_columns', feature_columns)
        selected_features = config['selected_features']
        if list(pipeline_feature_columns) != list(feature_columns):
            raise ValueError("This pipeline's feature_columns differ from the shared feature_columns.json.")
        run_id = int(config.get('run_id', path_run_id))
        fold_id = int(config.get('fold_idx', config.get('fold_id', path_fold_id)))
        pipeline_id = int(config.get('pipeline_id', (run_id - 1) * 10 + fold_id))
        seed = config.get('seed', np.nan)
        X_test_selected = selected_feature_transform(test_feature_df=test_feature_df, feature_columns=feature_columns, selected_features=selected_features, scaler=scaler, rfe=rfe)
        expected_model_features = getattr(model, 'n_features_in_', None)
        if expected_model_features is not None and X_test_selected.shape[1] != expected_model_features:
            raise ValueError(f'Selected test feature count does not match model expectation: {X_test_selected.shape[1]} versus {expected_model_features}.')
        y_pred = np.asarray(model.predict(X_test_selected), dtype=int)
        try:
            raw_proba = model.predict_proba(X_test_selected)
            model_classes = getattr(model, 'classes_', config.get('classes', all_classes))
            y_proba = align_probabilities(raw_proba=raw_proba, model_classes=model_classes, all_classes=all_classes)
        except Exception as proba_exc:
            print(f'Probability prediction unavailable: {proba_exc}')
            y_proba = None
        global_metrics = compute_global_metrics(y_true=y_test, y_pred=y_pred, y_proba=y_proba, all_classes=all_classes)
        global_row = {'run_id': run_id, 'fold_id': fold_id, 'pipeline_id': pipeline_id, 'seed': seed, 'n_selected_features': int(X_test_selected.shape[1]), 'pipeline_path': str(pipeline_path), **global_metrics}
        global_metric_records.append(global_row)
        class_rows = compute_per_class_metrics(y_true=y_test, y_pred=y_pred, y_proba=y_proba, all_classes=all_classes, label_encoder=label_encoder)
        for row in class_rows:
            row.update({'run_id': run_id, 'fold_id': fold_id, 'pipeline_id': pipeline_id, 'seed': seed})
            per_class_records.append(row)
        vote_counts[np.arange(n_samples), y_pred] += 1
        if y_proba is not None:
            probability_sum += y_proba
            n_probability_models += 1
        n_successful_models += 1
        print(f"Test MCC={global_metrics['MCC']:.4f} | Accuracy={global_metrics['Accuracy']:.4f} | Macro-F1={global_metrics['F1_macro']:.4f}")
        if SAVE_INDIVIDUAL_PREDICTIONS:
            prediction_df = pd.DataFrame({id_output_name: test_ids, 'true_encoded': y_test, 'true_label': y_test_original, 'pred_encoded': y_pred, 'pred_label': label_encoder.inverse_transform(y_pred)})
            if y_proba is not None:
                prediction_df['pred_confidence'] = y_proba.max(axis=1)
            prediction_path = os.path.join(prediction_dir, f'pipeline_{pipeline_id:03d}_predictions.csv.gz')
            prediction_df.to_csv(prediction_path, index=False, compression='gzip')
            del prediction_df
        pd.DataFrame(global_metric_records).to_csv(os.path.join(OUTPUT_DIR, 'per_pipeline_test_metrics_partial.csv'), index=False)
        del config
        del scaler
        del rfe
        del model
        del X_test_selected
        del y_pred
        if y_proba is not None:
            del y_proba
        gc.collect()
    except Exception as exc:
        print(f'FAILED: {exc}')
        failure_records.append({'run_id': path_run_id, 'fold_id': path_fold_id, 'pipeline_path': str(pipeline_path), 'stage': 'test_evaluation', 'error': repr(exc)})
        gc.collect()
if n_successful_models == 0:
    failure_path = os.path.join(OUTPUT_DIR, 'failed_pipelines.csv')
    pd.DataFrame(failure_records).to_csv(failure_path, index=False)
    raise RuntimeError(f'All pipelines failed during test evaluation. Failure details: {failure_path}')
global_metrics_df = pd.DataFrame(global_metric_records).sort_values(['run_id', 'fold_id'])
per_class_metrics_df = pd.DataFrame(per_class_records).sort_values(['class_name', 'run_id', 'fold_id'])
global_metrics_path = os.path.join(OUTPUT_DIR, 'per_pipeline_test_metrics.csv')
per_class_metrics_path = os.path.join(OUTPUT_DIR, 'per_pipeline_per_class_metrics.csv')
global_metrics_df.to_csv(global_metrics_path, index=False)
per_class_metrics_df.to_csv(per_class_metrics_path, index=False)
metric_columns = ['MCC', 'Accuracy', 'Balanced_Acc', 'Precision_macro', 'Recall_macro', 'F1_macro', 'F1_weighted', 'ROC_AUC_macro', 'ROC_AUC_weighted', 'PR_AUC_macro', 'PR_AUC_weighted']
metric_summary_df = summarise_numeric_metrics(global_metrics_df, metric_columns)
metric_summary_path = os.path.join(OUTPUT_DIR, 'test_metric_summary.csv')
metric_summary_df.to_csv(metric_summary_path, index=False)
run_level_summary_df = global_metrics_df.groupby('run_id', as_index=False)[[column for column in metric_columns if column in global_metrics_df.columns]].agg(['mean', 'std'])
run_level_summary_df.columns = ['_'.join((str(part) for part in column_tuple if str(part))) for column_tuple in run_level_summary_df.columns.to_flat_index()]
run_level_summary_path = os.path.join(OUTPUT_DIR, 'run_level_test_summary.csv')
run_level_summary_df.to_csv(run_level_summary_path, index=False)
per_class_summary_df = per_class_metrics_df.groupby('class_name', as_index=False)[['MCC_ovr', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC']].agg(['mean', 'std', 'median', 'min', 'max'])
per_class_summary_df.columns = ['_'.join((str(part) for part in column_tuple if str(part))) for column_tuple in per_class_summary_df.columns.to_flat_index()]
per_class_summary_path = os.path.join(OUTPUT_DIR, 'per_class_test_summary.csv')
per_class_summary_df.to_csv(per_class_summary_path, index=False)
ensemble_metric_records = []
ensemble_per_class_records = []
hard_vote_pred = vote_counts.argmax(axis=1).astype(int)
hard_vote_proba = vote_counts.astype(np.float64) / float(n_successful_models)
hard_metrics = compute_global_metrics(y_true=y_test, y_pred=hard_vote_pred, y_proba=hard_vote_proba, all_classes=all_classes)
ensemble_metric_records.append({'ensemble': 'hard_vote_100_pipelines', 'n_models': n_successful_models, **hard_metrics})
hard_class_rows = compute_per_class_metrics(y_true=y_test, y_pred=hard_vote_pred, y_proba=hard_vote_proba, all_classes=all_classes, label_encoder=label_encoder)
for row in hard_class_rows:
    row['ensemble'] = 'hard_vote_100_pipelines'
    row['n_models'] = n_successful_models
    ensemble_per_class_records.append(row)
soft_vote_pred = None
soft_vote_proba = None
if n_probability_models > 0:
    soft_vote_proba = probability_sum / float(n_probability_models)
    soft_vote_pred = soft_vote_proba.argmax(axis=1).astype(int)
    soft_metrics = compute_global_metrics(y_true=y_test, y_pred=soft_vote_pred, y_proba=soft_vote_proba, all_classes=all_classes)
    ensemble_metric_records.append({'ensemble': 'soft_vote_available_pipelines', 'n_models': n_probability_models, **soft_metrics})
    soft_class_rows = compute_per_class_metrics(y_true=y_test, y_pred=soft_vote_pred, y_proba=soft_vote_proba, all_classes=all_classes, label_encoder=label_encoder)
    for row in soft_class_rows:
        row['ensemble'] = 'soft_vote_available_pipelines'
        row['n_models'] = n_probability_models
        ensemble_per_class_records.append(row)
ensemble_metrics_df = pd.DataFrame(ensemble_metric_records)
ensemble_per_class_df = pd.DataFrame(ensemble_per_class_records)
ensemble_metrics_path = os.path.join(OUTPUT_DIR, 'ensemble_test_metrics.csv')
ensemble_per_class_path = os.path.join(OUTPUT_DIR, 'ensemble_per_class_metrics.csv')
ensemble_metrics_df.to_csv(ensemble_metrics_path, index=False)
ensemble_per_class_df.to_csv(ensemble_per_class_path, index=False)
ensemble_predictions = pd.DataFrame({id_output_name: test_ids, 'true_encoded': y_test, 'true_label': y_test_original, 'hard_vote_encoded': hard_vote_pred, 'hard_vote_label': label_encoder.inverse_transform(hard_vote_pred), 'hard_vote_confidence': hard_vote_proba.max(axis=1), 'hard_vote_agreement_count': vote_counts.max(axis=1), 'hard_vote_model_count': n_successful_models})
if soft_vote_pred is not None:
    ensemble_predictions['soft_vote_encoded'] = soft_vote_pred
    ensemble_predictions['soft_vote_label'] = label_encoder.inverse_transform(soft_vote_pred)
    ensemble_predictions['soft_vote_confidence'] = soft_vote_proba.max(axis=1)
    ensemble_predictions['soft_vote_model_count'] = n_probability_models
ensemble_predictions_path = os.path.join(OUTPUT_DIR, 'ensemble_predictions.csv')
ensemble_predictions.to_csv(ensemble_predictions_path, index=False)
class_names = label_encoder.classes_.astype(str)
hard_confusion = confusion_matrix(y_test, hard_vote_pred, labels=all_classes)
pd.DataFrame(hard_confusion, index=[f'true::{name}' for name in class_names], columns=[f'pred::{name}' for name in class_names]).to_csv(os.path.join(OUTPUT_DIR, 'hard_vote_confusion_matrix.csv'))
if soft_vote_pred is not None:
    soft_confusion = confusion_matrix(y_test, soft_vote_pred, labels=all_classes)
    pd.DataFrame(soft_confusion, index=[f'true::{name}' for name in class_names], columns=[f'pred::{name}' for name in class_names]).to_csv(os.path.join(OUTPUT_DIR, 'soft_vote_confusion_matrix.csv'))
if failure_records:
    failure_path = os.path.join(OUTPUT_DIR, 'failed_pipelines.csv')
    pd.DataFrame(failure_records).to_csv(failure_path, index=False)
else:
    failure_path = None
print('\n' + '=' * 88)
print('TEST EVALUATION COMPLETE')
print('=' * 88)
print(f'Successful pipelines: {n_successful_models}/{len(pipeline_paths)}')
print(f'Pipelines with probabilities: {n_probability_models}/{n_successful_models}')
print('\nDescriptive results across individual pipelines:')
print(metric_summary_df.to_string(index=False))
print('\nPost-hoc ensemble results:')
print(ensemble_metrics_df.to_string(index=False))
print('\nSaved outputs:')
print(f'  {global_metrics_path}')
print(f'  {per_class_metrics_path}')
print(f'  {metric_summary_path}')
print(f'  {run_level_summary_path}')
print(f'  {per_class_summary_path}')
print(f'  {ensemble_metrics_path}')
print(f'  {ensemble_per_class_path}')
print(f'  {ensemble_predictions_path}')
if failure_path is not None:
    print(f'  {failure_path}')
print('\nImportant interpretation:')
print('- The mean and standard deviation across 100 pipelines describe pipeline variability; the pipelines are correlated, so this is not a formal confidence interval.')
print("- Do not choose the 'best' pipeline by test MCC. The test set should remain a final, untouched evaluation set.")
print("- Hard/soft voting here is a post-hoc ensemble across the 100 saved outer-fold models, not Auto-sklearn's internal ensemble.")
