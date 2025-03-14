#!/usr/bin/env python
"""
This script performs feature extraction and machine learning experiments using 
features extracted from the ProvPath model. The extracted features (from TCGA 
and UKE slides) are loaded from pickle files, converted to DataFrames, and then 
used for training and evaluating an XGBoost classifier with various PCA dimensionality 
reductions. Hyperparameter optimization is performed using Optuna. 

By Mohamed Albahri, 2024.
"""

# =============================================================================
# Imports
# =============================================================================
import os
import glob
import pickle
import random
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import ast
import json
import optuna
from optuna.samplers import TPESampler

# Set logging level
logging.basicConfig(level=logging.DEBUG)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features_df = None

# =============================================================================
# Load Features from ProvPath Model
# =============================================================================
# Paths to features (slide representations) extracted from ProvPath Model
tcga_sampled_patches_path = "prov-gigapath/outputs/braf/slide_representations/braf/tcga_train_slide_representations.pkl"
tcga_val_sampled_patches_path = "prov-gigapath/outputs/braf/slide_representations/tcga_val_slide_representations.pkl"
uke_sampled_patches_path = "prov-gigapath/outputs/braf/slide_representations/uke_slide_representations.pkl"

tcga_sampled_data = pd.read_pickle(tcga_sampled_patches_path)
print("Loaded TCGA slide_representations: ", tcga_sampled_patches_path)

tcga_val_sampled_data = pd.read_pickle(tcga_val_sampled_patches_path)
print("Loaded TCGA validation slide_representations: ", tcga_val_sampled_patches_path)

uke_sampled_data = pd.read_pickle(uke_sampled_patches_path)
print("Loaded UKE slide_representations: ", uke_sampled_patches_path)

# =============================================================================
# Data Conversion and Preparation
# =============================================================================
def convert_to_dataframe(data):
    """
    Convert a list of dictionaries to a Pandas DataFrame.
    If the 'label' field is a numpy array, it converts it to a scalar.
    """
    for entry in data:
        if isinstance(entry['label'], np.ndarray):
            entry['label'] = entry['label'].item()  # Convert numpy array to scalar
    df = pd.DataFrame(data)
    return df

tcga_df = convert_to_dataframe(tcga_sampled_data)
tcga_val_df = convert_to_dataframe(tcga_val_sampled_data)
uke_df = convert_to_dataframe(uke_sampled_data)

# Concatenate the TCGA training and validation dataframes
tcga_df = pd.concat([tcga_df, tcga_val_df], ignore_index=True)

# Extract features, labels, and slide IDs for TCGA training data
X_train = np.vstack(tcga_df['features'].values)  # type: ignore
print(X_train)
y_train = tcga_df['label'].values
tcga_slide_ids = tcga_df['slide_id'].values
tcga_unique_slide_ids = np.unique(tcga_slide_ids)  # type: ignore
tcga_slide_labels_df = tcga_df[['slide_id', 'label']].drop_duplicates()

# Extract features, labels, and slide IDs for TCGA validation data
X_val = np.vstack(tcga_val_df['features'].values)  # type: ignore
y_val = tcga_val_df['label'].values
tcga_val_slide_ids = tcga_val_df['slide_id'].values
tcga_val_unique_slide_ids = np.unique(tcga_val_slide_ids)  # type: ignore
tcga_val_slide_labels_df = tcga_val_df[['slide_id', 'label']].drop_duplicates()

# Extract features, labels, and slide IDs for UKE data (test set)
X_test = np.vstack(uke_df['features'].values)  # type: ignore
y_test = uke_df['label'].values  
uke_slide_ids = uke_df['slide_id'].values
uke_unique_slide_ids = np.unique(uke_slide_ids)  # type: ignore
uke_slide_labels_df = uke_df[['slide_id', 'label']].drop_duplicates()

# Unique slide labels for stratification
unique_labels = np.array([y_train[tcga_slide_ids == slide].max() for slide in tcga_unique_slide_ids])
uke_unique_labels = np.array([y_test[uke_slide_ids == slide].max() for slide in uke_unique_slide_ids])

print("len of unique_labels: ", len(unique_labels))
print("Len of uke_unique_labels: ", len(uke_unique_labels))
print("Len tcga_slide_labels_df: ", len(tcga_slide_labels_df))
print("Len uke_slide_labels_df: ", len(uke_slide_labels_df))
print("Len tcga_val_slide_labels_df: ", len(tcga_val_slide_labels_df))
print("Len of tcga_unique_slide_ids: ", len(tcga_unique_slide_ids))
print("Len of uke_unique_slide_ids: ", len(uke_unique_slide_ids))
print("tcga_slide_labels_df: ", tcga_slide_labels_df)
print("uke_slide_labels_df: ", uke_slide_labels_df)

# =============================================================================
# Cross-Validation and Hyperparameter Tuning with Optuna
# =============================================================================
# Dictionary to store cross-validation results
cv_results = {
    'n_components': [],
    'fold': [],
    'accuracy': [],
    'roc_auc': [],
    'optimal_threshold': [],
    'y_true': [],
    'y_pred': []
}

pca_components_list = [100, 70, 175]
results = {}  # Store results for different PCA components

for n_components in pca_components_list:
    print(f"Training with PCA n_components={n_components}")
    
    # Define the objective function for hyperparameter tuning
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1200),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 15),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 2.5),
            'seed': 0
        }
        
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        aucs = []
        for train_index, val_index in skf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            # Scale the data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_fold_scaled = scaler.fit_transform(X_train_fold)
            X_val_fold_scaled = scaler.transform(X_val_fold)
            
            # Apply PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components, random_state=42)
            X_train_fold_reduced = pca.fit_transform(X_train_fold_scaled)
            X_val_fold_reduced = pca.transform(X_val_fold_scaled)
            
            clf = xgb.XGBClassifier(
                **params,
                random_state=42,
                eval_metric='auc',
                early_stopping_rounds=5,
                n_jobs=15
            )
            clf.fit(X_train_fold_reduced, y_train_fold, eval_set=[(X_val_fold_reduced, y_val_fold)], verbose=False)
            pred_proba = clf.predict_proba(X_val_fold_reduced)[:, 1]
            auc_value = roc_auc_score(y_val_fold, pred_proba)
            aucs.append(auc_value)
        
        mean_auc = np.mean(aucs)
        trial.set_user_attr(key="best_booster", value=clf)
        return mean_auc
    
    def callback(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])
                
    import optuna
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize")
    study.optimize(objective, n_trials=1000, callbacks=[callback])
    best_model = study.user_attrs["best_booster"]
    print("Best hyperparameters: ", study.best_params)
    print("Best AUC: ", study.best_value)
    xgb_best_params = study.best_params

    #######################################
    # Scale the full training set
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA on the full training set
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_reduced = pca.fit_transform(X_train_scaled)
    X_test_reduced = pca.transform(X_test_scaled)
    
    # Train XGBoost on the full training set
    best_model.fit(X_train_reduced, y_train)
    print("Training done..")
    
    # Predict on the test set using XGBoost
    y_pred_xgb_proba = best_model.predict_proba(X_test_reduced)[:, 1]
    y_test = np.array(y_test)
    y_pred_xgb_proba = np.array(y_pred_xgb_proba)

    # Find optimal threshold for XGBoost using precision-recall curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_xgb_proba)
    f1_scores = 2 * precision * recall / (precision + recall)
    optimal_threshold_xgb = thresholds[np.argmax(f1_scores)]
    y_pred_xgb_optimal = (y_pred_xgb_proba >= optimal_threshold_xgb).astype(int)

    assert len(y_test) == len(y_pred_xgb_proba), "The length of y_test and y_pred_xgb_proba must match"

    accuracy_xgb_slides = accuracy_score(y_test, y_pred_xgb_optimal)
    classification_rep_xgb_slides = classification_report(y_test, y_pred_xgb_optimal, zero_division=0)
    roc_auc_xgb_slides = roc_auc_score(y_test, y_pred_xgb_proba)
    print(f"XGBoost Slide-Level ROC-AUC on Test Set: {roc_auc_xgb_slides}")
          
    # Save predictions and true labels
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'prediction': y_pred_xgb_proba
    })
    predictions_df.to_csv('/projects/wispermed_rp18/braf-main/braf-main/traditional_ML/results/xgboost_predictions.csv', index=False)

    sensitivity = np.sum((y_pred_xgb_optimal == 1) & (y_test == 1)) / np.sum(y_test == 1)
    specificity = np.sum((y_pred_xgb_optimal == 0) & (y_test == 0)) / np.sum(y_test == 0)
    
    print(f"Sensitivity with optimal threshold: {sensitivity:.3f}, Specificity with optimal threshold: {specificity:.3f}")

    results[n_components] = {
        'xgb_slide_accuracy': accuracy_xgb_slides,
        'xgb_slide_classification_report': classification_rep_xgb_slides,
        'xgb_slide_roc_auc': roc_auc_xgb_slides,
    }
    
# Save cross-validation results for CI calculation
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv('/projects/wispermed_rp18/braf-main/braf-main/traditional_ML/results/cv_results.csv', index=False)

# Write evaluation results to a text file
with open('/projects/wispermed_rp18/braf-main/braf-main/traditional_ML/results/TestProvPath_slide_representation.txt', 'a') as f:
    for n_components, result in results.items():
        f.write(f"PCA n_components={n_components}\n")
        f.write(f"XGBoost Slide-Level Accuracy on Test Set: {result['xgb_slide_accuracy']}\n")
        f.write("XGBoost Slide-Level Classification Report:\n" + result['xgb_slide_classification_report'] + "\n")
        f.write(f"XGBoost Slide-Level ROC-AUC on Test Set: {result['xgb_slide_roc_auc']}\n")

# Print evaluation results
for n_components, result in results.items():
    print(f"PCA n_components={n_components}")
    print(f"XGBoost Slide-Level Test Accuracy: {result['xgb_slide_accuracy']}")
    print(result['xgb_slide_classification_report'])
    print(f"XGBoost Slide-Level ROC-AUC on Test Set: {result['xgb_slide_roc_auc']}")
