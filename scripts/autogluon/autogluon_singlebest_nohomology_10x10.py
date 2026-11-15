import gc
import json
import os
import time
import warnings
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, label_binarize
warnings.filterwarnings('ignore')
TRAIN_FILE = os.environ.get('REFERENCE_TRAIN_CSV', 'data/features/reference/homology_aware/Train_Features_NoHomology.csv')
TEST_FILE = os.environ.get('REFERENCE_TEST_CSV', 'data/features/reference/homology_aware/Test_Features_NoHomology.csv')
OUTPUT_DIR = os.environ.get('AUTOGLUON_OUTPUT_DIR', 'runs/autogluon_singlebest_nohomology_10x10')
RUN_IDS = range(1, 11)
OUTER_FOLDS = 10
TARGET_FEATURES = 100
TIME_LIMIT_SECONDS = 120 * 60
ID_COL = 'ProteinID'
LABEL_COL = 'label'
CLASSES = list(range(11))
LABEL_MAPPING = {'Cell.membrane': 0, 'Cytoplasm': 1, 'Cytoplasm-Nucleus': 2, 'Endoplasmic.reticulum': 3, 'Extracellular': 4, 'Golgi.apparatus': 5, 'Lysosome/Vacuole': 6, 'Mitochondrion': 7, 'Nucleus': 8, 'Peroxisome': 9, 'Plastid': 10}

def encode_labels(values, source_name):
    values = pd.Series(values)
    if np.issubdtype(values.dtype, np.number):
        encoded = values.astype(int)
    else:
        encoded = values.map(LABEL_MAPPING)
        if encoded.isna().any():
            unknown = sorted(values[encoded.isna()].astype(str).unique())
            raise ValueError(f'Unknown labels in {source_name}: {unknown}')
        encoded = encoded.astype(int)
    observed = sorted(encoded.unique().tolist())
    if observed != CLASSES:
        raise ValueError(f'Expected classes {CLASSES} in {source_name}, found {observed}')
    return encoded.to_numpy()

def align_probabilities(probability_df):
    renamed = probability_df.copy()
    renamed.columns = [int(column) for column in renamed.columns]
    return renamed.reindex(columns=CLASSES, fill_value=0.0).to_numpy()

def calculate_metrics(y_true, y_pred, y_proba):
    y_true_bin = label_binarize(y_true, classes=CLASSES)
    result = {'MCC': float(matthews_corrcoef(y_true, y_pred)), 'Accuracy': float(accuracy_score(y_true, y_pred)), 'Balanced_Acc': float(balanced_accuracy_score(y_true, y_pred))}
    for average in ['micro', 'macro', 'weighted']:
        result[f'Precision_{average}'] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        result[f'Recall_{average}'] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        result[f'F1_{average}'] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    try:
        result['ROC_AUC_macro'] = float(roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='macro'))
        result['PR_AUC_macro'] = float(average_precision_score(y_true_bin, y_proba, average='macro'))
    except Exception as error:
        result['ROC_AUC_macro'] = np.nan
        result['PR_AUC_macro'] = np.nan
        result['AUC_Error'] = str(error)
    return result

def prediction_table(source_df, y_true, y_pred, y_proba):
    data = {'true_label': y_true, 'predicted_label': y_pred}
    if ID_COL in source_df.columns:
        data[ID_COL] = source_df[ID_COL].to_numpy()
    for class_index, class_id in enumerate(CLASSES):
        data[f'proba_class_{class_id}'] = y_proba[:, class_index]
    return pd.DataFrame(data)
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
feature_columns = [column for column in train_df.columns if column not in [ID_COL, LABEL_COL]]
test_feature_columns = [column for column in test_df.columns if column not in [ID_COL, LABEL_COL]]
if feature_columns != test_feature_columns:
    missing = [column for column in feature_columns if column not in test_df.columns]
    extra = [column for column in test_feature_columns if column not in train_df.columns]
    raise ValueError(f'Training and locked-test feature columns/order do not match. Missing={missing[:10]}, extra={extra[:10]}')
X_full = train_df[feature_columns]
y_full = encode_labels(train_df[LABEL_COL], TRAIN_FILE)
X_test_full = test_df[feature_columns]
y_test = encode_labels(test_df[LABEL_COL], TEST_FILE)
os.makedirs(OUTPUT_DIR, exist_ok=True)
all_results = []
print('=' * 80)
print('AutoGluon SingleBest — NoHomology 10x10')
print(f'Train: {TRAIN_FILE} | {X_full.shape}')
print(f'Locked test: {TEST_FILE} | {X_test_full.shape}')
print(f'Maximum model-search time per workflow: {TIME_LIMIT_SECONDS / 60:.0f} min')
print('=' * 80)
for run_id in RUN_IDS:
    split_seed = run_id * 777
    splitter = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=split_seed)
    for fold_id, (tr_idx, val_idx) in enumerate(splitter.split(X_full, y_full), start=1):
        workflow_id = f'AutoGluon_SingleBest_NoHomology_R{run_id:02d}_F{fold_id:02d}'
        rfe_seed = run_id * 100 + fold_id
        fold_dir = os.path.join(OUTPUT_DIR, f'run_{run_id:02d}', f'fold_{fold_id:02d}')
        summary_file = os.path.join(fold_dir, 'workflow_summary.csv')
        test_prediction_file = os.path.join(fold_dir, 'locked_test_predictions.csv')
        if os.path.exists(summary_file) and os.path.exists(test_prediction_file):
            completed = pd.read_csv(summary_file)
            if len(completed) == 1 and 'Test_MCC' in completed.columns:
                all_results.append(completed.iloc[0].to_dict())
                print(f'{workflow_id}: already complete; skipped.', flush=True)
                continue
        os.makedirs(fold_dir, exist_ok=True)
        start_time = time.time()
        print(f'\n{workflow_id} | split={split_seed} | RFE seed={rfe_seed}', flush=True)
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
            raise RuntimeError(f'{workflow_id}: expected 100 RFE features, found {len(selected_features)}')
        pd.DataFrame({'feature': selected_features}).to_csv(os.path.join(fold_dir, 'selected_features.csv'), index=False)
        train_for_autogluon = pd.DataFrame(X_tr_selected, columns=selected_features)
        train_for_autogluon[LABEL_COL] = y_tr
        validation_features = pd.DataFrame(X_val_selected, columns=selected_features)
        test_features = pd.DataFrame(X_test_selected, columns=selected_features)
        predictor = TabularPredictor(label=LABEL_COL, problem_type='multiclass', eval_metric='mcc', path=os.path.join(fold_dir, 'autogluon_models'), verbosity=2)
        predictor.fit(train_data=train_for_autogluon, time_limit=TIME_LIMIT_SECONDS, presets='medium_quality', num_bag_folds=0, num_stack_levels=0, auto_stack=False, fit_weighted_ensemble=False, keep_only_best=True)
        selected_model = predictor.model_best
        predictor.leaderboard(silent=True).to_csv(os.path.join(fold_dir, 'leaderboard.csv'), index=False)
        y_val_pred = predictor.predict(validation_features).to_numpy().astype(int)
        y_test_pred = predictor.predict(test_features).to_numpy().astype(int)
        y_val_proba = align_probabilities(predictor.predict_proba(validation_features, as_multiclass=True))
        y_test_proba = align_probabilities(predictor.predict_proba(test_features, as_multiclass=True))
        validation_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
        validation_source = train_df.iloc[val_idx].reset_index(drop=True)
        prediction_table(validation_source, y_val, y_val_pred, y_val_proba).to_csv(os.path.join(fold_dir, 'outer_validation_predictions.csv'), index=False)
        prediction_table(test_df, y_test, y_test_pred, y_test_proba).to_csv(test_prediction_file, index=False)
        summary_row = {'method': 'AutoGluon_SingleBest', 'condition': 'NoHomology', 'workflow_id': workflow_id, 'run_id': run_id, 'fold_id': fold_id, 'outer_split_seed': split_seed, 'rfe_seed': rfe_seed, 'selected_model': selected_model, 'n_training_fold': len(tr_idx), 'n_validation_fold': len(val_idx), 'n_locked_test': len(y_test), 'n_selected_features': len(selected_features), 'time_limit_seconds': TIME_LIMIT_SECONDS, 'runtime_seconds': time.time() - start_time, **{f'Validation_{key}': value for key, value in validation_metrics.items()}, **{f'Test_{key}': value for key, value in test_metrics.items()}}
        pd.DataFrame([summary_row]).to_csv(summary_file, index=False)
        with open(os.path.join(fold_dir, 'metadata.json'), 'w', encoding='utf-8') as file:
            json.dump(summary_row, file, indent=2)
        all_results.append(summary_row)
        print(f"{workflow_id} | model={selected_model} | Validation MCC={validation_metrics['MCC']:.4f} | Test MCC={test_metrics['MCC']:.4f}", flush=True)
        del predictor, scaler, rfe
        del X_tr_scaled, X_val_scaled, X_test_scaled
        del X_tr_selected, X_val_selected, X_test_selected
        del y_val_pred, y_val_proba, y_test_pred, y_test_proba
        gc.collect()
    pd.DataFrame(all_results).to_csv(os.path.join(OUTPUT_DIR, 'autogluon_singlebest_nohomology_progress.csv'), index=False)
results_df = pd.DataFrame(all_results).sort_values(['run_id', 'fold_id']).reset_index(drop=True)
if len(results_df) != 100:
    raise RuntimeError(f'Expected 100 workflows, found {len(results_df)}')
results_df.to_csv(os.path.join(OUTPUT_DIR, 'autogluon_singlebest_nohomology_results_100workflows.csv'), index=False)
print('\n' + '=' * 80)
print('AutoGluon SingleBest NoHomology: 100 workflows completed')
print('=' * 80)
print('\nOuter-validation MCC:')
print(results_df['Validation_MCC'].describe())
print('\nLocked-test MCC:')
print(results_df['Test_MCC'].describe())
print(f'\nResults saved to: {OUTPUT_DIR}')
