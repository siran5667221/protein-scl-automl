import os
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from tabpfn import TabPFNClassifier
warnings.filterwarnings('ignore')
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = 'true'
TRAIN_FILE = os.environ.get('REFERENCE_TRAIN_CSV', 'data/features/reference/random/train_numeric_labels.csv')
TEST_FILE = os.environ.get('REFERENCE_TEST_CSV', 'data/features/reference/random/test_numeric_labels.csv')
OUTPUT_DIR = os.environ.get('TABPFN_OUTPUT_DIR', 'runs/tabpfn_10x10_100workflows')
RUN_IDS = range(1, 11)
OUTER_FOLDS = 10
TARGET_FEATURES = 100

def calculate_metrics(y_true, y_pred, y_proba, classes):
    results = {'MCC': float(matthews_corrcoef(y_true, y_pred)), 'Accuracy': float(accuracy_score(y_true, y_pred)), 'Balanced_Acc': float(balanced_accuracy_score(y_true, y_pred))}
    try:
        y_true_bin = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            results['ROC_AUC_macro'] = float(roc_auc_score(y_true, y_proba[:, 1]))
            results['PR_AUC_macro'] = float(average_precision_score(y_true, y_proba[:, 1]))
        else:
            results['ROC_AUC_macro'] = float(roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='macro'))
            results['PR_AUC_macro'] = float(average_precision_score(y_true_bin, y_proba, average='macro'))
    except Exception as error:
        results['ROC_AUC_macro'] = np.nan
        results['PR_AUC_macro'] = np.nan
        results['AUC_Error'] = str(error)
    return results

def read_completed_summary(summary_file):
    if not os.path.exists(summary_file):
        return None
    completed = pd.read_csv(summary_file)
    if len(completed) != 1:
        raise RuntimeError(f'Expected one row in {summary_file}')
    return completed.iloc[0].to_dict()
print('Loading data...', flush=True)
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
feature_columns = [column for column in train_df.columns if column not in ['ProteinID', 'label']]
test_feature_columns = [column for column in test_df.columns if column not in ['ProteinID', 'label']]
if feature_columns != test_feature_columns:
    raise ValueError('Training and locked-test feature columns do not match.')
X_train_full = train_df[feature_columns]
y_train_full = train_df['label'].to_numpy()
X_test_full = test_df[feature_columns]
y_test_full = test_df['label'].to_numpy()
print('Training shape:', X_train_full.shape, flush=True)
print('Locked-test shape:', X_test_full.shape, flush=True)
print('Feature count:', len(feature_columns), flush=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
all_results = []
for run_id in RUN_IDS:
    outer_split_seed = run_id * 777
    skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=outer_split_seed)
    fold_splits = list(skf.split(X_train_full, y_train_full))
    print('\n' + '=' * 80, flush=True)
    print(f'Starting TabPFN Run {run_id:02d}: outer split seed = {outer_split_seed}', flush=True)
    print('=' * 80, flush=True)
    for fold_id, (tr_idx, val_idx) in enumerate(fold_splits, start=1):
        workflow_id = f'TabPFN_R{run_id:02d}_F{fold_id:02d}'
        rfe_seed = run_id * 100 + fold_id
        fold_output_dir = os.path.join(OUTPUT_DIR, f'run_{run_id:02d}', f'fold_{fold_id:02d}')
        os.makedirs(fold_output_dir, exist_ok=True)
        workflow_summary_file = os.path.join(fold_output_dir, 'workflow_summary.csv')
        completed_row = read_completed_summary(workflow_summary_file)
        if completed_row is not None:
            all_results.append(completed_row)
            print(f'{workflow_id}: already complete; skipped.', flush=True)
            continue
        fold_start = time.time()
        print(f'\n{workflow_id}', flush=True)
        print(f'Outer split seed: {outer_split_seed}', flush=True)
        print(f'RFE seed: {rfe_seed}', flush=True)
        X_tr = X_train_full.iloc[tr_idx]
        X_val = X_train_full.iloc[val_idx]
        y_tr = y_train_full[tr_idx]
        y_val = y_train_full[val_idx]
        scaler = MinMaxScaler()
        X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32)
        X_test_scaled = scaler.transform(X_test_full).astype(np.float32)
        rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=rfe_seed, n_jobs=-1), n_features_to_select=TARGET_FEATURES, step=0.1)
        X_tr_selected = rfe.fit_transform(X_tr_scaled, y_tr)
        X_val_selected = rfe.transform(X_val_scaled)
        X_test_selected = rfe.transform(X_test_scaled)
        selected_features = [feature_columns[index] for index, selected in enumerate(rfe.get_support()) if selected]
        if len(selected_features) != TARGET_FEATURES:
            raise RuntimeError(f'{workflow_id}: expected {TARGET_FEATURES} features, got {len(selected_features)}.')
        pd.DataFrame({'feature': selected_features}).to_csv(os.path.join(fold_output_dir, 'selected_features.csv'), index=False)
        model = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
        model.fit(X_tr_selected, y_tr)
        classes = model.classes_
        y_val_pred = model.predict(X_val_selected)
        y_val_proba = model.predict_proba(X_val_selected)
        val_metrics = calculate_metrics(y_true=y_val, y_pred=y_val_pred, y_proba=y_val_proba, classes=classes)
        y_test_pred = model.predict(X_test_selected)
        y_test_proba = model.predict_proba(X_test_selected)
        test_metrics = calculate_metrics(y_true=y_test_full, y_pred=y_test_pred, y_proba=y_test_proba, classes=classes)
        prediction_df = pd.DataFrame({'ProteinID': test_df['ProteinID'], 'true_label': y_test_full, 'predicted_label': y_test_pred})
        for class_index, class_label in enumerate(classes):
            prediction_df[f'proba_class_{class_label}'] = y_test_proba[:, class_index]
        prediction_df.to_csv(os.path.join(fold_output_dir, 'locked_test_predictions.csv'), index=False)
        summary_row = {'method': 'TabPFN', 'workflow_id': workflow_id, 'run_id': run_id, 'fold_id': fold_id, 'outer_split_seed': outer_split_seed, 'rfe_seed': rfe_seed, 'n_training_fold': len(tr_idx), 'n_validation_fold': len(val_idx), 'n_selected_features': len(selected_features), 'runtime_seconds': time.time() - fold_start}
        for metric, value in val_metrics.items():
            summary_row[f'Validation_{metric}'] = value
        for metric, value in test_metrics.items():
            summary_row[f'Test_{metric}'] = value
        pd.DataFrame([summary_row]).to_csv(workflow_summary_file, index=False)
        fold_metadata = {**summary_row, 'scaler': 'MinMaxScaler fitted on outer training fold only', 'rfe': 'RF-RFE: RandomForest(50 trees, balanced), step=0.1', 'device': 'cpu'}
        with open(os.path.join(fold_output_dir, 'fold_metadata.json'), 'w', encoding='utf-8') as file:
            json.dump(fold_metadata, file, indent=2)
        all_results.append(summary_row)
        print(f"{workflow_id} | validation MCC = {val_metrics['MCC']:.4f} | locked-test MCC = {test_metrics['MCC']:.4f}", flush=True)
        del model, scaler, rfe
        del X_tr_scaled, X_val_scaled, X_test_scaled
        del X_tr_selected, X_val_selected, X_test_selected
        del y_val_pred, y_val_proba, y_test_pred, y_test_proba
        gc.collect()
    pd.DataFrame(all_results).to_csv(os.path.join(OUTPUT_DIR, 'tabpfn_10x10_progress.csv'), index=False)
results_df = pd.DataFrame(all_results).sort_values(['run_id', 'fold_id']).reset_index(drop=True)
if len(results_df) != 100:
    raise RuntimeError(f'Expected 100 completed workflows, found {len(results_df)}.')
results_df.to_csv(os.path.join(OUTPUT_DIR, 'tabpfn_locked_test_results_100workflows.csv'), index=False)
print('\n' + '=' * 80)
print('TabPFN 10 x 10: 100 workflows completed')
print('=' * 80)
print(results_df['Test_MCC'].describe())
print('\nResults saved to:')
print(OUTPUT_DIR)
