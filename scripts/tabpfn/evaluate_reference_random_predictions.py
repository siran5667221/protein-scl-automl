import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
ROOT = os.environ.get('TABPFN_OUTPUT_DIR', 'runs/tabpfn_10x10_100workflows')
CLASSES = list(range(11))
overall, per_class, matrices = ([], [], [])
for run_id in range(1, 11):
    for fold_id in range(1, 11):
        workflow_id = f'TabPFN_R{run_id:02d}_F{fold_id:02d}'
        path = os.path.join(ROOT, f'run_{run_id:02d}', f'fold_{fold_id:02d}', 'locked_test_predictions.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        y_true, y_pred = (df.true_label.to_numpy(), df.predicted_label.to_numpy())
        proba_cols = [f'proba_class_{c}' for c in CLASSES]
        if not set(proba_cols).issubset(df.columns):
            raise ValueError(f'Missing probability columns in {path}')
        y_proba = df[proba_cols].to_numpy()
        y_bin = label_binarize(y_true, classes=CLASSES)
        row = {'method': 'TabPFN', 'workflow_id': workflow_id, 'run_id': run_id, 'fold_id': fold_id, 'MCC': matthews_corrcoef(y_true, y_pred), 'Accuracy': accuracy_score(y_true, y_pred), 'Balanced_Acc': balanced_accuracy_score(y_true, y_pred), 'F1_macro': f1_score(y_true, y_pred, labels=CLASSES, average='macro', zero_division=0), 'F1_weighted': f1_score(y_true, y_pred, labels=CLASSES, average='weighted', zero_division=0), 'ROC_AUC_macro': roc_auc_score(y_bin, y_proba, multi_class='ovr', average='macro'), 'PR_AUC_macro': average_precision_score(y_bin, y_proba, average='macro')}
        overall.append(row)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
        for i, c in enumerate(CLASSES):
            yt, yp = ((y_true == c).astype(int), (y_pred == c).astype(int))
            per_class.append({'workflow_id': workflow_id, 'run_id': run_id, 'fold_id': fold_id, 'class_id': c, 'precision': precision[i], 'recall': recall[i], 'F1': f1[i], 'MCC': matthews_corrcoef(yt, yp), 'n_test_samples': support[i]})
        matrices.append(confusion_matrix(y_true, y_pred, labels=CLASSES))
overall_df = pd.DataFrame(overall)
per_class_df = pd.DataFrame(per_class)
summary = []
for metric in ['MCC', 'Accuracy', 'Balanced_Acc', 'F1_macro', 'F1_weighted', 'ROC_AUC_macro', 'PR_AUC_macro']:
    x = overall_df[metric]
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    summary.append({'method': 'TabPFN', 'metric': metric, 'n_workflows': len(x), 'mean': x.mean(), 'median': x.median(), 'std': x.std(ddof=1), 'IQR': x.quantile(0.75) - x.quantile(0.25), '95CI_low': lo, '95CI_high': hi, 'minimum': x.min(), 'maximum': x.max()})
location_summary = per_class_df.groupby('class_id', as_index=False).agg(n_workflows=('MCC', 'count'), n_test_samples=('n_test_samples', 'first'), mean_MCC=('MCC', 'mean'), median_MCC=('MCC', 'median'), mean_F1=('F1', 'mean'), median_F1=('F1', 'median'), mean_precision=('precision', 'mean'), mean_recall=('recall', 'mean'))
overall_df.to_csv(os.path.join(ROOT, 'tabpfn_full_metrics_100workflows.csv'), index=False)
pd.DataFrame(summary).to_csv(os.path.join(ROOT, 'tabpfn_full_metrics_summary_100workflows.csv'), index=False)
per_class_df.to_csv(os.path.join(ROOT, 'tabpfn_per_location_full_metrics_100workflows.csv'), index=False)
location_summary.to_csv(os.path.join(ROOT, 'tabpfn_per_location_full_metrics_summary_100workflows.csv'), index=False)
np.save(os.path.join(ROOT, 'tabpfn_confusion_matrices_100workflows.npy'), np.stack(matrices))
print(pd.DataFrame(summary))
print(location_summary)
