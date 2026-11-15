import gc
import json
import os
import time
import warnings
import numpy as np
import pandas as pd
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = 'true'
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, label_binarize
from tabpfn import TabPFNClassifier
TRAIN_FILE = os.environ.get('REFERENCE_TRAIN_CSV', 'data/features/reference/homology_aware/Train_Features_NoHomology.csv')
TEST_FILE = os.environ.get('REFERENCE_TEST_CSV', 'data/features/reference/homology_aware/Test_Features_NoHomology.csv')
ROOT_SAVE_DIR = os.environ.get('TABPFN_OUTPUT_DIR', 'runs/tabpfn_nohomology_10x10')
REPEATS = 10
OUTER_FOLDS = 10
TARGET_FEATURES = 100
ID_COL = 'ProteinID'
LABEL_COL = 'label'
LABEL_MAPPING = {'Cell.membrane': 0, 'Cytoplasm': 1, 'Cytoplasm-Nucleus': 2, 'Endoplasmic.reticulum': 3, 'Extracellular': 4, 'Golgi.apparatus': 5, 'Lysosome/Vacuole': 6, 'Mitochondrion': 7, 'Nucleus': 8, 'Peroxisome': 9, 'Plastid': 10}
EXPECTED_CLASSES = np.arange(11, dtype=int)

def compute_metrics(y_true, y_pred, y_proba, proba_classes):
    result = {'MCC': float(matthews_corrcoef(y_true, y_pred)), 'Accuracy': float(accuracy_score(y_true, y_pred)), 'Balanced_Acc': float(balanced_accuracy_score(y_true, y_pred))}
    for average in ['micro', 'macro', 'weighted']:
        result[f'Precision_{average}'] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        result[f'Recall_{average}'] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        result[f'F1_{average}'] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    try:
        aligned_proba = np.zeros((len(y_true), len(EXPECTED_CLASSES)), dtype=float)
        class_to_column = {int(label): i for i, label in enumerate(proba_classes)}
        for output_column, class_id in enumerate(EXPECTED_CLASSES):
            if int(class_id) in class_to_column:
                aligned_proba[:, output_column] = y_proba[:, class_to_column[int(class_id)]]
        y_true_bin = label_binarize(y_true, classes=EXPECTED_CLASSES)
        result['ROC_AUC_macro'] = float(roc_auc_score(y_true_bin, aligned_proba, multi_class='ovr', average='macro'))
        result['PR_AUC_macro'] = float(average_precision_score(y_true_bin, aligned_proba, average='macro'))
    except Exception as error:
        result['ROC_AUC_macro'] = np.nan
        result['PR_AUC_macro'] = np.nan
        result['AUC_Error'] = str(error)
    return result
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
if LABEL_COL not in train_df.columns:
    raise ValueError(f'Missing label column: {LABEL_COL}')
feature_columns = [column for column in train_df.columns if column not in [ID_COL, LABEL_COL]]
X_full = train_df[feature_columns]
y_full = train_df[LABEL_COL].to_numpy()
missing_test_features = [column for column in feature_columns if column not in test_df.columns]
if missing_test_features:
    raise ValueError(f'Locked test is missing {len(missing_test_features)} training features: {missing_test_features[:10]}')
X_test_full = test_df[feature_columns]
y_test_full = test_df[LABEL_COL].to_numpy()
if not np.issubdtype(y_full.dtype, np.number):
    mapped = pd.Series(y_full).map(LABEL_MAPPING)
    if mapped.isna().any():
        unknown = sorted(pd.Series(y_full)[mapped.isna()].astype(str).unique())
        raise ValueError(f'Unknown labels: {unknown}')
    y_full = mapped.to_numpy(dtype=int)
else:
    y_full = y_full.astype(int)
if not np.issubdtype(y_test_full.dtype, np.number):
    mapped_test = pd.Series(y_test_full).map(LABEL_MAPPING)
    if mapped_test.isna().any():
        unknown = sorted(pd.Series(y_test_full)[mapped_test.isna()].astype(str).unique())
        raise ValueError(f'Unknown locked-test labels: {unknown}')
    y_test_full = mapped_test.to_numpy(dtype=int)
else:
    y_test_full = y_test_full.astype(int)
observed_classes = np.sort(np.unique(y_full))
if not np.array_equal(observed_classes, EXPECTED_CLASSES):
    raise ValueError(f'Expected classes 0..10, but found {observed_classes.tolist()}')
os.makedirs(ROOT_SAVE_DIR, exist_ok=True)
all_results = []
print('=' * 80)
print('TabPFN 10x10 — No-homology training data')
print(f'Input: {TRAIN_FILE}')
print(f'Shape: {X_full.shape}')
print(f'Locked test: {TEST_FILE}')
print(f'Locked-test shape: {X_test_full.shape}')
print(f'Output: {ROOT_SAVE_DIR}')
print('=' * 80)
for run_id in range(1, REPEATS + 1):
    split_seed = run_id * 777
    splitter = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=split_seed)
    for fold_id, (tr_idx, val_idx) in enumerate(splitter.split(X_full, y_full), start=1):
        workflow_id = f'TabPFN_NoHomology_R{run_id:02d}_F{fold_id:02d}'
        rfe_seed = run_id * 100 + fold_id
        fold_dir = os.path.join(ROOT_SAVE_DIR, f'run_{run_id:02d}', f'fold_{fold_id:02d}')
        summary_file = os.path.join(fold_dir, 'workflow_summary.csv')
        locked_prediction_file = os.path.join(fold_dir, 'locked_test_predictions.csv')
        if os.path.exists(summary_file) and os.path.exists(locked_prediction_file):
            completed = pd.read_csv(summary_file)
            if len(completed) != 1 or 'Test_MCC' not in completed.columns:
                raise RuntimeError(f'Invalid summary file: {summary_file}')
            all_results.append(completed.iloc[0].to_dict())
            print(f'{workflow_id}: already complete; skipped.', flush=True)
            continue
        os.makedirs(fold_dir, exist_ok=True)
        start_time = time.time()
        print(f'\n{workflow_id} | split={split_seed} | RFE={rfe_seed}', flush=True)
        X_tr = X_full.iloc[tr_idx]
        X_val = X_full.iloc[val_idx]
        y_tr = y_full[tr_idx]
        y_val = y_full[val_idx]
        scaler = MinMaxScaler()
        X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32)
        X_test_scaled = scaler.transform(X_test_full).astype(np.float32)
        rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=rfe_seed, n_jobs=-1), n_features_to_select=TARGET_FEATURES, step=0.1)
        X_tr_selected = rfe.fit_transform(X_tr_scaled, y_tr).astype(np.float32)
        X_val_selected = rfe.transform(X_val_scaled).astype(np.float32)
        X_test_selected = rfe.transform(X_test_scaled).astype(np.float32)
        selected_features = [feature_columns[index] for index, selected in enumerate(rfe.get_support()) if selected]
        if len(selected_features) != TARGET_FEATURES:
            raise RuntimeError(f'{workflow_id}: expected 100 selected features, found {len(selected_features)}')
        pd.DataFrame({'feature': selected_features}).to_csv(os.path.join(fold_dir, 'selected_features.csv'), index=False)
        model = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
        model.fit(X_tr_selected, y_tr)
        y_val_pred = model.predict(X_val_selected).astype(int)
        y_val_proba = model.predict_proba(X_val_selected)
        val_metrics = compute_metrics(y_true=y_val, y_pred=y_val_pred, y_proba=y_val_proba, proba_classes=model.classes_)
        y_test_pred = model.predict(X_test_selected).astype(int)
        y_test_proba = model.predict_proba(X_test_selected)
        test_metrics = compute_metrics(y_true=y_test_full, y_pred=y_test_pred, y_proba=y_test_proba, proba_classes=model.classes_)
        prediction_data = {'row_index': train_df.index[val_idx], 'true_label': y_val, 'predicted_label': y_val_pred}
        if ID_COL in train_df.columns:
            prediction_data[ID_COL] = train_df.iloc[val_idx][ID_COL].to_numpy()
        class_to_column = {int(label): index for index, label in enumerate(model.classes_)}
        for class_id in EXPECTED_CLASSES:
            probability = np.zeros(len(val_idx), dtype=float)
            if int(class_id) in class_to_column:
                probability = y_val_proba[:, class_to_column[int(class_id)]]
            prediction_data[f'proba_class_{class_id}'] = probability
        pd.DataFrame(prediction_data).to_csv(os.path.join(fold_dir, 'outer_validation_predictions.csv'), index=False)
        locked_prediction_data = {'true_label': y_test_full, 'predicted_label': y_test_pred}
        if ID_COL in test_df.columns:
            locked_prediction_data[ID_COL] = test_df[ID_COL].to_numpy()
        for class_id in EXPECTED_CLASSES:
            probability = np.zeros(len(y_test_full), dtype=float)
            if int(class_id) in class_to_column:
                probability = y_test_proba[:, class_to_column[int(class_id)]]
            locked_prediction_data[f'proba_class_{class_id}'] = probability
        pd.DataFrame(locked_prediction_data).to_csv(locked_prediction_file, index=False)
        summary_row = {'method': 'TabPFN', 'condition': 'NoHomology', 'workflow_id': workflow_id, 'run_id': run_id, 'fold_id': fold_id, 'outer_split_seed': split_seed, 'rfe_seed': rfe_seed, 'n_training_fold': len(tr_idx), 'n_validation_fold': len(val_idx), 'n_locked_test': len(y_test_full), 'n_selected_features': len(selected_features), 'runtime_seconds': time.time() - start_time, **{f'Validation_{key}': value for key, value in val_metrics.items()}, **{f'Test_{key}': value for key, value in test_metrics.items()}}
        pd.DataFrame([summary_row]).to_csv(summary_file, index=False)
        with open(os.path.join(fold_dir, 'metadata.json'), 'w', encoding='utf-8') as file:
            json.dump(summary_row, file, indent=2)
        all_results.append(summary_row)
        print(f"{workflow_id} | Validation MCC={val_metrics['MCC']:.4f} | Locked-test MCC={test_metrics['MCC']:.4f}", flush=True)
        del model, scaler, rfe
        del X_tr_scaled, X_val_scaled, X_test_scaled
        del X_tr_selected, X_val_selected, X_test_selected
        del y_val_pred, y_val_proba, y_test_pred, y_test_proba
        gc.collect()
    pd.DataFrame(all_results).to_csv(os.path.join(ROOT_SAVE_DIR, 'tabpfn_nohomology_progress.csv'), index=False)
results_df = pd.DataFrame(all_results).sort_values(['run_id', 'fold_id']).reset_index(drop=True)
if len(results_df) != REPEATS * OUTER_FOLDS:
    raise RuntimeError(f'Expected {REPEATS * OUTER_FOLDS} completed workflows, found {len(results_df)}')
results_file = os.path.join(ROOT_SAVE_DIR, 'tabpfn_nohomology_locked_test_results_100workflows.csv')
results_df.to_csv(results_file, index=False)
print('\n' + '=' * 80)
print('TabPFN NoHomology 10x10: 100 workflows completed')
print('=' * 80)
print('\nOuter-validation MCC:')
print(results_df['Validation_MCC'].describe())
print('\nLocked-test MCC:')
print(results_df['Test_MCC'].describe())
print(f'\nResults saved to: {ROOT_SAVE_DIR}')
