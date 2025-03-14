


#%%
import os
import pickle
import numpy as np
import pandas as pd
import timm
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve,precision_recall_curve, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import json

# if not 'CUDA_VISIBLE_DEVICES' in os.environ:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features_df= None 


#### Features extracted from ProvPath Model

tcga_sampled_patches_path ="prov-gigapath/outputs/braf/slide_representations/tcga_train_slide_representations.pkl"
tcga_val_sampled_patches_path ="prov-gigapath/outputs/braf/slide_representations/tcga_val_slide_representations.pkl"
uke_sampled_patches_path = "prov-gigapath/outputs/braf/slide_representations/uke_slide_representations.pkl"


tcga_sampled_data = pd.read_pickle(tcga_sampled_patches_path)
print("Loaded TCGA slide_representations: ",tcga_sampled_patches_path )

tcga_val_sampled_data = pd.read_pickle(tcga_val_sampled_patches_path)
print("Loaded TCGA validation slide_representations: ",tcga_val_sampled_patches_path )

uke_sampled_data = pd.read_pickle(uke_sampled_patches_path)
print("Loaded UKE slide_representations: ", uke_sampled_patches_path)

################################ Testing on UKE dataset  ############################

# Define training data
def convert_to_dataframe(data):
    # Convert list of dictionaries to DataFrame
    for entry in data:
        if isinstance(entry['label'], np.ndarray):
            entry['label'] = entry['label'].item()  # Convert numpy array to scalar
    df = pd.DataFrame(data)
    return df

tcga_df = convert_to_dataframe(tcga_sampled_data)
tcga_val_df = convert_to_dataframe(tcga_val_sampled_data)
uke_df = convert_to_dataframe(uke_sampled_data)


# Concatenate the TCGA and TCGA validation dataframes
tcga_df = pd.concat([tcga_df, tcga_val_df], ignore_index=True)


# Extract features, labels, and slide IDs for TCGA
X_train = np.vstack(tcga_df['features'].values)  #type: ignore
# print(X_train)
y_train = tcga_df['label'].values  # No need to concatenate
tcga_slide_ids = tcga_df['slide_id'].values
tcga_unique_slide_ids = np.unique(tcga_slide_ids) #type: ignore
tcga_slide_labels_df = tcga_df[['slide_id', 'label']].drop_duplicates()

# Extract features, labels, and slide IDs for TCGA validation
X_val = np.vstack(tcga_val_df['features'].values)  #type: ignore
y_val = tcga_val_df['label'].values  #
tcga_val_slide_ids = tcga_val_df['slide_id'].values
tcga_val_unique_slide_ids = np.unique(tcga_val_slide_ids) #type: ignore
tcga_val_slide_labels_df = tcga_val_df[['slide_id', 'label']].drop_duplicates()


# Extract features, labels, and slide IDs for UKE
X_test = np.vstack(uke_df['features'].values) #type: ignore
y_test = uke_df['label'].values  
uke_slide_ids = uke_df['slide_id'].values
uke_unique_slide_ids = np.unique(uke_slide_ids) #type: ignore
uke_slide_labels_df = uke_df[['slide_id', 'label']].drop_duplicates()
# Unique slide ids and corresponding labels for stratification
# unique_slides = np.unique(slide_ids) # type: ignore
unique_labels = np.array([y_train[tcga_slide_ids == slide].max() for slide in tcga_unique_slide_ids])
uke_unique_labels = np.array([y_test[uke_slide_ids == slide].max() for slide in uke_unique_slide_ids])

print("len of unique_labels: ", len(unique_labels)) 
print("Len of uke_unique_labels: ", len(uke_unique_labels))
print("Len tcga_slide_labels_df: ", len(tcga_slide_labels_df))
print("Len uke_slide_labels_df: ", len(uke_slide_labels_df))

print("Len tcga_val_slide_labels_df: ", len(tcga_val_slide_labels_df))

# print("uke_unique_slide_ids: ", uke_unique_slide_ids)
print("Len of tcga_unique_slide_ids: ", len(tcga_unique_slide_ids))
print("Len of uke_unique_slide_ids: ", len(uke_unique_slide_ids))

print("tcga_slide_labels_df: ", tcga_slide_labels_df)
print("uke_slide_labels_df: ", uke_slide_labels_df)


xgb_best_params = { 'colsample_bytree': 0.6,'learning_rate': 0.0005,'max_depth': 4,'min_child_weight': 8, 
    'n_estimators': 800, 'subsample': 0.4 , 'reg_alpha': 14.0, 'reg_lambda': 8.0, 'scale_pos_weight': 1.8450169491304928
}


##########################################################################################
# Dictionary to store cross-validation results


cv_results = {
    'n_components': [],
    'fold': [],
    'accuracy': [],
    'roc_auc': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'optimal_threshold': [],
    'y_true': [],
    'y_pred': []
}

test_results = {
    'n_components': [],
    'fold': [],
    'true_label': [],
    'prediction': []
}

results = {}
pca_components_list= [60,100,170]

# Initialize result storage
mean_cv_roc_aucs_list = []
test_set_roc_aucs_list = []

for n_components in pca_components_list:
    print(f"Training with PCA n_components={n_components}")

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_accuracies = []
    cv_roc_aucs = []
    cv_precisions = []
    cv_recalls = []
    cv_f1_scores = []

    fold = 0
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)
        
        # Apply PCA (if necessary)
        pca = PCA(n_components=n_components, random_state=42)  
        X_train_fold_reduced = pca.fit_transform(X_train_fold_scaled)
        X_val_fold_reduced = pca.transform(X_val_fold_scaled)

        xgb_model = xgb.XGBClassifier(
            random_state=42,
            early_stopping_rounds=5,
            n_jobs=15,
            eval_metric='auc',
            **xgb_best_params
        )

        xgb_model.fit(X_train_fold_reduced, y_train_fold, eval_set=[[X_val_fold_reduced, y_val_fold]], verbose=False)
        
        y_val_proba = xgb_model.predict_proba(X_val_fold_reduced)[:, 1]

        # Find optimal threshold for xgb        
        fpr, tpr, thresholds = roc_curve(y_val_fold, y_val_proba)
        distance = np.sqrt((1 - tpr)**2 + fpr**2)
        optimal_val_threshold = thresholds[np.argmin(distance)]

        # Make predictions with optimal threshold
        y_pred_val_optimal = (y_val_proba >= optimal_val_threshold).astype(int)
       
        # Slide Level metrics
        accuracy = accuracy_score(y_val_fold, y_pred_val_optimal)
        roc_auc = roc_auc_score(y_val_fold, y_val_proba)
        precision_val = precision_score(y_val_fold, y_pred_val_optimal)
        recall_val = recall_score(y_val_fold, y_pred_val_optimal)
        f1_val = f1_score(y_val_fold, y_pred_val_optimal)

        cv_accuracies.append(accuracy)
        cv_roc_aucs.append(roc_auc)
        cv_precisions.append(precision_val)
        cv_recalls.append(recall_val)
        cv_f1_scores.append(f1_val)

        # Save results for this fold
        cv_results['n_components'].append(n_components)
        cv_results['fold'].append(fold)
        cv_results['accuracy'].append(accuracy)
        cv_results['roc_auc'].append(roc_auc)
        cv_results['precision'].append(precision_val)
        cv_results['recall'].append(recall_val)
        cv_results['f1_score'].append(f1_val)
        cv_results['optimal_threshold'].append(optimal_val_threshold)
        cv_results['y_true'].append(json.dumps(y_val_fold.tolist()))
        cv_results['y_pred'].append(json.dumps(y_val_proba.tolist()))
        fold += 1

    mean_cv_accuracy = np.mean(cv_accuracies)
    std_cv_accuracy = np.std(cv_accuracies)
    mean_cv_roc_auc = np.mean(cv_roc_aucs)
    std_cv_roc_auc = np.std(cv_roc_aucs)
    mean_cv_precision = np.mean(cv_precisions)
    std_cv_precision = np.std(cv_precisions)
    mean_cv_recall = np.mean(cv_recalls)
    std_cv_recall = np.std(cv_recalls)
    mean_cv_f1 = np.mean(cv_f1_scores)
    std_cv_f1 = np.std(cv_f1_scores)

    print(f"Mean CV Accuracy: {mean_cv_accuracy} ± {std_cv_accuracy}")
    print(f"Mean CV ROC AUC : {mean_cv_roc_auc} ± {std_cv_roc_auc}")
    print(f"Mean CV Precision: {mean_cv_precision} ± {std_cv_precision}")
    print(f"Mean CV Recall: {mean_cv_recall} ± {std_cv_recall}")
    print(f"Mean CV F1 Score: {mean_cv_f1} ± {std_cv_f1}")


    mean_cv_roc_aucs_list.append(mean_cv_roc_auc)

    # Save mean results for this n_components value
    cv_results['n_components'].append(n_components)
    cv_results['fold'].append('mean')
    cv_results['accuracy'].append(mean_cv_accuracy)
    cv_results['roc_auc'].append(mean_cv_roc_auc)
    cv_results['precision'].append(mean_cv_precision)
    cv_results['recall'].append(mean_cv_recall)
    cv_results['f1_score'].append(mean_cv_f1)
    cv_results['optimal_threshold'].append(np.mean(cv_results['optimal_threshold'][-fold:]))
    cv_results['y_true'].append(y_val_fold.tolist())
    cv_results['y_pred'].append(y_val_proba.tolist())

    # Scale the full training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA on the full training set
    pca = PCA(n_components=n_components, random_state=42)
    X_train_reduced = pca.fit_transform(X_train_scaled)
    X_test_reduced = pca.transform(X_test_scaled)

    # Initialize XGBoost with the best parameters
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='auc',
        n_jobs=15,
        **xgb_best_params
    )

    # Train XGBoost on the full training set
    xgb_model.fit(X_train_reduced, y_train)

    # Predict with XGBoost on the test set
    y_pred_xgb_proba = xgb_model.predict_proba(X_test_reduced)[:, 1]
    y_test = np.array(y_test)
    y_pred_xgb_proba = np.array(y_pred_xgb_proba)

    # Find optimal threshold for xgb 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_xgb_proba)
    distance = np.sqrt((1 - tpr)**2 + fpr**2)
    optimal_threshold_xgb = thresholds[np.argmin(distance)]

    # Make predictions with optimal threshold
    y_pred_xgb_optimal = (y_pred_xgb_proba >= optimal_threshold_xgb).astype(int)

    accuracy_xgb_slides = accuracy_score(y_test, y_pred_xgb_optimal)
    classification_rep_xgb_slides = classification_report(y_test, y_pred_xgb_optimal, zero_division=0)
    roc_auc_xgb_slides = roc_auc_score(y_test, y_pred_xgb_proba)
    print(f"XGBoost Slide-Level ROC-AUC on Test Set: ", roc_auc_xgb_slides) 
    
    test_set_roc_aucs_list.append(roc_auc_xgb_slides)

    sensitivity = np.sum((y_pred_xgb_optimal == 1) & (y_test == 1)) / np.sum(y_test == 1)
    specificity = np.sum((y_pred_xgb_optimal == 0) & (y_test == 0)) / np.sum(y_test == 0)

    # Save predictions and true labels
    for i in range(len(y_test)):
        test_results['n_components'].append(n_components)
        test_results['fold'].append('test')
        test_results['true_label'].append(y_test[i])
        test_results['prediction'].append(y_pred_xgb_proba[i])


    test_results_df = pd.DataFrame(test_results)

    test_results_df.to_csv(f'results/xgboost_predictions.csv', index=False)

    
    print(f"Sensitivity with optimal threshold: {sensitivity:.3f}, Specificity with optimal threshold: {specificity:.3f}")

    results[n_components] = {
        'xgb_mean_cv_accuracy': mean_cv_accuracy,
        'xgb_mean_cv_roc_auc': mean_cv_roc_auc,
        'xgb_slide_accuracy': accuracy_xgb_slides,
        'xgb_slide_classification_report': classification_rep_xgb_slides,
        'xgb_slide_roc_auc': roc_auc_xgb_slides,
        'xgb_sensitivity': sensitivity,
        'xgb_specificity': specificity
    }

# Save cross-validation results to a CSV file
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv('results/cv_results.csv', index=False)

# Print evaluation results
for n_components, result in results.items():
    print(f"PCA n_components={n_components}")
    print(f"Mean CV Accuracy on training data: {result['xgb_mean_cv_accuracy']}")
    print(f"Mean CV ROC AUC on training data: {result['xgb_mean_cv_roc_auc']}")
    print(f"XGBoost Slide-Level Test Accuracy: {result['xgb_slide_accuracy']}")
    print(result['xgb_slide_classification_report'])        
    print(f"XGBoost Slide-Level ROC-AUC on Test Set: {result['xgb_slide_roc_auc']}")
    print(f"Sensitivity with optimal threshold: {result['xgb_sensitivity']:.3f}, Specificity with optimal threshold: {result['xgb_specificity']:.3f}")


# Plot the mean CV AUC and test set AUC
import matplotlib.pyplot as plt

# Plot Mean CV AUC
plt.figure(figsize=(10, 6))
plt.plot(pca_components_list, mean_cv_roc_aucs_list, marker='o', linestyle='None')
plt.title('Mean CV ROC AUC vs PCA Components')
plt.xlabel('PCA Components')
plt.ylabel('Mean CV ROC AUC')
plt.grid(True)
plt.savefig('mean_cv_auc_vs_pca_components.png')
plt.show()

# Plot Test Set AUC
plt.figure(figsize=(10, 6))
plt.plot(pca_components_list, test_set_roc_aucs_list, marker='o', color='orange', linestyle='None')
plt.title('Test Set ROC AUC vs PCA Components')
plt.xlabel('PCA Components')
plt.ylabel('Test Set ROC AUC')
plt.grid(True)
plt.savefig('test_set_auc_vs_pca_components.png')
plt.show()


# %%
