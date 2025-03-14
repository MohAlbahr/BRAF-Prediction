#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve, auc

# Function to parse JSON-like lists in columns for XGBoost CV data
def parse_json_like_list(column):
    def parse_value(x):
        if isinstance(x, str):
            try:
                return np.array(eval(x))
            except (SyntaxError, NameError):
                raise ValueError(f"Could not parse string as a JSON-like list: {x}")
        elif isinstance(x, (list, np.ndarray)):
            return np.array(x)
        elif isinstance(x, (int, float)):
            return np.array([x])
        else:
            raise ValueError(f"Unexpected value type: {type(x)}")
    return column.apply(parse_value)

# Function to calculate metrics and ROC AUC with bootstrap CI
def analyze_predictions(y_true, y_pred, title_prefix):
    if len(np.unique(y_true)) == 1:
        print(f"Skipping {title_prefix} due to only one class present in y_true.")
        return None, None, None, None
    
    roc_auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    distance = np.sqrt((1 - tpr)**2 + fpr**2)
    optimal_threshold = thresholds[np.argmin(distance)]

    # Stratified Bootstrap CI
    ci_low, ci_high, bootstrapped_aucs = stratified_bootstrap_auc(y_true, y_pred)
    print(f'{title_prefix} - 95% CI for AUC with stratified bootstrap: [{ci_low:.3f}, {ci_high:.3f}]')

    y_pred_binary_optimal = (y_pred >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary_optimal).ravel()
    sensitivity_optimal = tp / (tp + fn)
    specificity_optimal = tn / (tn + fp)
    print(f'{title_prefix} - Sensitivity with optimal threshold: {sensitivity_optimal:.3f}, Specificity with optimal threshold: {specificity_optimal:.3f}')

    metrics = {
        'roc_auc': roc_auc,
        'sensitivity': sensitivity_optimal,
        'specificity': specificity_optimal,
        'ci_low': ci_low,
        'ci_high': ci_high
    }

    return y_pred_binary_optimal, roc_auc, bootstrapped_aucs, metrics
# Stratified Bootstrap CI calculation
def stratified_bootstrap_auc(y_true, y_pred, n_resamples=1000, confidence_level=0.95):
    indices = np.arange(len(y_true))
    pos_indices = indices[y_true == 1]
    neg_indices = indices[y_true == 0]
    
    auc_scores = []
    for _ in range(n_resamples):
        pos_sample = resample(pos_indices, replace=True, n_samples=len(pos_indices))
        neg_sample = resample(neg_indices, replace=True, n_samples=len(neg_indices))
        sample_indices = np.concatenate([pos_sample, neg_sample])
        
        if len(np.unique(y_true[sample_indices])) < 2:
            continue
        
        auc = roc_auc_score(y_true[sample_indices], y_pred[sample_indices])
        auc_scores.append(auc)
    
    lower = np.percentile(auc_scores, (1 - confidence_level) / 2 * 100)
    upper = np.percentile(auc_scores, (1 + confidence_level) / 2 * 100)
    
    return lower, upper, auc_scores

# Function to process ProvPath cross-validation results
def process_provpath_cv(file_path, true_label_col, pred_col, title_prefix):
    data = pd.read_csv(file_path)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for fold in sorted(data['fold'].unique()):
        fold_data = data[data['fold'] == fold]
        y_true = fold_data[true_label_col].values
        y_pred = fold_data[pred_col].values

        if len(np.unique(y_true)) < 2:
            print(f"Skipping fold {fold} due to only one class present in y_true.")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        roc_auc = roc_auc_score(y_true, y_pred)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {fold} (area = {roc_auc:.2f})')

    if len(tprs) == 0:
        print(f"No valid folds found for {title_prefix}. Skipping plotting.")
        return

    # Plotting the mean ROC curve and std dev
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (area = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=.8)

    tprs_upper = np.minimum(mean_tpr + std_auc, 1)
    tprs_lower = np.maximum(mean_tpr - std_auc, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'± 1 std. dev.')

    plt.title(title_prefix)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

# Function to process XGBoost cross-validation results
def process_xgboost_cv(file_path, true_label_col, pred_col, title_prefix):
    data = pd.read_csv(file_path)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    n_components = 100  # Assuming you want to use n_components=100
    comp_data = data[data['n_components'] == n_components]
    for fold in sorted(comp_data['fold'].unique()):
        fold_data = comp_data[comp_data['fold'] == fold]
        y_true = parse_json_like_list(fold_data[true_label_col]).iloc[0]
        y_pred = parse_json_like_list(fold_data[pred_col]).iloc[0]

        if len(np.unique(y_true)) < 2:
            print(f"Skipping fold {fold} with n_components {n_components} due to only one class present in y_true.")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        roc_auc = roc_auc_score(y_true, y_pred)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {fold} (area = {roc_auc:.2f}) PCA-component {n_components}')

    if len(tprs) == 0:
        print(f"No valid folds found for n_components {n_components}. Skipping plotting.")
        return

    # Plotting the mean ROC curve and std dev
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (area = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=.8)

    tprs_upper = np.minimum(mean_tpr + std_auc, 1)
    tprs_lower = np.maximum(mean_tpr - std_auc, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'± 1 std. dev.')

    plt.title(title_prefix)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

# Function to process CSV data and generate ROC curves
def process_csv(file_path, true_label_col, pred_col, title_prefix, is_cv=False, is_for_confusion_mat=False):
    if is_cv:
        if "prov-gigapath" in file_path:
            process_provpath_cv(file_path, true_label_col, pred_col, title_prefix)
        else:
            process_xgboost_cv(file_path, true_label_col, pred_col, title_prefix)
        return None, None, None, None

    data = pd.read_csv(file_path)
    if "XGBoost on UHE" in title_prefix:
        n_components = 100
        data = data[data['n_components'] == n_components]

    y_true = data[true_label_col].values
    y_pred = data[pred_col].values

    y_pred_binary_optimal, roc_auc, bootstrapped_aucs, metrics = analyze_predictions(y_true, y_pred, title_prefix)
    
    if not  is_for_confusion_mat:
        if y_pred_binary_optimal is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            plt.plot(fpr, tpr, lw=2, label=f'{title_prefix} (area = {roc_auc:.2f})')
            plt.title(title_prefix)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
    
    return y_true, y_pred_binary_optimal, roc_auc, metrics





# Set up the figure with a 2x3 grid of subplots and maximize subplot area
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs[0, 0].grid(False)
axs[1, 0].grid(False)

# Plot Confusion Matrix for Prov-GigaPath on UHE (Top Left)
y_true, y_pred_binary_optimal, _, _ = process_csv(
    "prov-gigapath/outputs/braf/braf/run-globalPool-unfreeze_traintype-UKE_epoch-5_blr-0.003_BS-32_wd-0.05_ld-0.95_drop-0.5_dropPR-0.1_feat-9/predictions.csv", 
    'y_true', 'y_pred', "Prov-GigaPath on UHE", False, True
)
cm = confusion_matrix(y_true, y_pred_binary_optimal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=axs[0, 0], cmap='viridis')
axs[0, 0].set_title('Confusion Matrix: Prov-GigaPath on UHE')
axs[0, 0].grid(False)

# Add letter 'a' to the top left corner of the first subplot
axs[0, 0].text(-0.1, 1.1, 'a)', transform=axs[0, 0].transAxes, fontsize=16, fontweight='bold', va='top')

# Plot ROC Curve for Prov-GigaPath on UHE (Top Middle)
plt.sca(axs[0, 1])  # Set current axis
process_csv(
    "prov-gigapath/outputs/braf/braf/run-globalPool-unfreeze_traintype-UKE_epoch-5_blr-0.003_BS-32_wd-0.05_ld-0.95_drop-0.5_dropPR-0.1_feat-9/predictions.csv", 
    'y_true', 'y_pred', "Prov-GigaPath on UHE", False
)
axs[0, 1].legend(loc="lower right", fontsize=8)

# Add letter 'b' to the top left corner of the second subplot
axs[0, 1].text(-0.1, 1.1, 'b)', transform=axs[0, 1].transAxes, fontsize=16, fontweight='bold', va='top')

# Plot ROC Curve for XGBoost on UHE (Top Right)
plt.sca(axs[0, 2])  # Set current axis
process_csv(
    "results/xgboost_predictions.csv", 
    'true_label', 'prediction', "XGBoost on UHE", False
)
axs[0, 2].legend(loc="lower right", fontsize=8)

# Add letter 'c' to the top left corner of the third subplot
axs[0, 2].text(-0.1, 1.1, 'c)', transform=axs[0, 2].transAxes, fontsize=16, fontweight='bold', va='top')

# Plot Confusion Matrix for XGBoost on UHE (Bottom Left)
y_true, y_pred_binary_optimal, _, _ = process_csv(
    "results/xgboost_predictions.csv", 
    'true_label', 'prediction', "XGBoost on UHE", False, True
)
cm = confusion_matrix(y_true, y_pred_binary_optimal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=axs[1, 0], cmap='viridis')
axs[1, 0].set_title('Confusion Matrix: XGBoost on UHE')
axs[1, 0].grid(False)

# Plot ROC Curves for Prov-GigaPath CV (Bottom Middle)
plt.sca(axs[1, 1])  # Set current axis
process_provpath_cv(
    "prov-gigapath/outputs/braf/braf/run-globalPool-unfreeze_traintype-CV_epoch-5_blr-0.003_BS-32_wd-0.05_ld-0.95_drop-0.5_dropPR-0.1_feat-9/predictions.csv", 
    'y_true', 'y_pred', "Prov-GigaPath CV"
)
axs[1, 1].legend(loc="lower right", fontsize=8)

# Plot ROC Curves for XGBoost CV (Bottom Right)
plt.sca(axs[1, 2])  # Set current axis
n_components = 100
data = pd.read_csv("results/cv_results.csv")
comp_data = data[data['n_components'] == n_components]

mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

for fold in sorted(comp_data['fold'].unique()):
    fold_data = comp_data[comp_data['fold'] == fold]
    y_true = parse_json_like_list(fold_data['y_true']).iloc[0]
    y_pred = parse_json_like_list(fold_data['y_pred']).iloc[0]

    if len(np.unique(y_true)) < 2:
        print(f"Skipping fold {fold} with n_components {n_components} due to only one class present in y_true.")
        continue

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)
    roc_auc = roc_auc_score(y_true, y_pred)
    aucs.append(roc_auc)
    axs[1, 2].plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {fold} (area = {roc_auc:.2f}) PCA-component {n_components}')

# Plotting the mean ROC curve and std dev for XGBoost CV
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

axs[1, 2].plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (area = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=.8)

tprs_upper = np.minimum(mean_tpr + std_auc, 1)
tprs_lower = np.maximum(mean_tpr - std_auc, 0)
axs[1, 2].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'± 1 std. dev.')

axs[1, 2].set_title('XGBoost CV')
axs[1, 2].set_xlabel('False Positive Rate')
axs[1, 2].set_ylabel('True Positive Rate')
axs[1, 2].legend(loc="lower right", fontsize=8)

# # Adjust the layout to increase space between subplots and reduce white space
# plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Save the combined figure as both JPEG and PDF files with reduced white space
plt.savefig('combined_figure.jpeg', format='jpeg', dpi=300, bbox_inches='tight')
# import matplotlib
# matplotlib.use('Agg')  # Use a non-interactive backend
plt.savefig('combined_figure.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()

plt.close()






# %%
