#%%
"""
Description:
    This script loads a pretrained model for BRAF mutation status prediction on TCGA slides 
    (test dataset) using the Prov-GigaPath foundation model alone (without additional fine-tuning). 
    Predictions are then used to calculate metrics (e.g., ROC AUC, accuracy, F1, precision, recall)
    for cross-validation.
    
    File paths for the checkpoint, predictions, and metrics are constructed using relative paths 
    to ensure reproducibility.
    
Author: Mohamed Albahri
Year: 2024
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import os
import sys
import argparse

# =============================================================================
# Third-Party Imports
# =============================================================================
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# =============================================================================
# Local Library Imports
# =============================================================================
# Fix import for interactive sessions: add the project root (two levels up)
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')))

from gigapath.classification_head import get_model
from datasets.slide_datatset import SlideDataset
from params import get_finetune_params
from task_configs.utils import load_task_config
from utils import seed_torch, get_exp_code, get_splits, get_loader, save_obj

# =============================================================================
# Prediction Function for Metrics Calculation
# =============================================================================
def predict_and_save(model, dataloader, device, args, fold=None):
    model.eval()
    predictions = []
    y_trues = []
    y_preds = []
    
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print('Using fp16 training')

    with torch.no_grad():
        for batch in dataloader:
            images, img_coords, labels, slide_ids = batch['imgs'], batch['coords'], batch['labels'], batch['slide_id']
            images = images.to(device, non_blocking=True)
            img_coords = img_coords.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                logits = model.forward(images, img_coords)
                probs = torch.softmax(logits, dim=1)[:, 1]

                y_trues.append(labels.cpu().item())
                y_preds.append(probs.cpu().item())

                pred_dict = {
                    'slide_id': slide_ids[0],
                    'y_true': labels.cpu().item(),
                    'y_pred': probs.cpu().item()
                }
                if fold is not None:
                    pred_dict['fold'] = fold

                predictions.append(pred_dict)
    
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    
    print("y_trues: ", y_trues)
    print("y_preds: ", y_preds)
    
    roc_auc = roc_auc_score(y_trues, y_preds)
    print(f"ROC AUC: {roc_auc}")

    fpr, tpr, thresholds = roc_curve(y_trues, y_preds)
    distance = np.sqrt((1 - tpr)**2 + fpr**2)
    optimal_threshold = thresholds[np.argmin(distance)]
    print(f"Optimal Threshold: {optimal_threshold}")
    
    y_pred_optimal = (y_preds >= optimal_threshold).astype(int)
    accuracy_optimal = accuracy_score(y_trues, y_pred_optimal)
    f1 = f1_score(y_trues, y_pred_optimal)
    precision_metric = precision_score(y_trues, y_pred_optimal)
    recall_metric = recall_score(y_trues, y_pred_optimal)
    
    print(f"Accuracy (optimal threshold): {accuracy_optimal}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision_metric}")
    print(f"Recall: {recall_metric}")
    
    metrics = {
        'roc_auc': roc_auc,
        'accuracy_optimal': accuracy_optimal,
        'f1_score': f1,
        'precision': precision_metric,
        'recall': recall_metric
    }
    if fold is not None:
        metrics['fold'] = fold

    print("Final Metrics: ", metrics)
    
    return predictions, metrics

# =============================================================================
# Main Function
# =============================================================================
def main():
    args = get_finetune_params()
    print(args)

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Set random seed.
    seed_torch(device, args.seed)

    # Load task configuration.
    print('Loading task configuration from: {}'.format(args.task_cfg_path))
    args.task_config = load_task_config(args.task_cfg_path)
    print(args.task_config)
    args.task = args.task_config.get('name', 'task')

    # Set experiment save directory (from args).
    args.save_dir = os.path.join(args.save_dir, args.task, args.exp_name)
    args.model_code, args.task_code, args.exp_code = get_exp_code(args)  # Get experiment code.
    args.save_dir = os.path.join(args.save_dir, args.exp_code)
    os.makedirs(args.save_dir, exist_ok=True)
    print('Experiment code: {}'.format(args.exp_code))
    print('Setting save directory: {}'.format(args.save_dir))

    # Set learning rate and effective batch size.
    eff_batch_size = args.batch_size * args.gc
    if args.lr is None or args.lr < 0:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.gc)
    print("effective batch size: %d" % eff_batch_size)

    # Set the split key.
    args.split_key = 'pat_id' if args.pat_strat else 'slide_id'
    print("args.split_key: ", args.split_key)

    # Set up the dataset split directory.
    args.split_dir = os.path.join(args.split_dir, args.task_code) if not args.pre_split_dir else args.pre_split_dir
    os.makedirs(args.split_dir, exist_ok=True)
    print('Setting split directory: {}'.format(args.split_dir))
  
    args.device = device  # Ensure device is set.

    # Load dataset CSV.
    dataset = pd.read_csv(args.dataset_csv)
    DatasetClass = SlideDataset
    CV_predictions = []
    CV_metrics = []

    # Start cross-validation (for each fold).
    for fold in range(args.folds):
        model = get_model(**vars(args))
        checkpoint_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../prov-gigapath/outputs/braf/braf/run-globalPool-unfreeze_traintype-CV_epoch-5_blr-0.003_BS-32_wd-0.05_ld-0.95_drop-0.5_dropPR-0.1_feat-9",
                f'fold_{fold}',
                "checkpoint.pt"
            )
        )
        model.load_state_dict(torch.load(checkpoint_path))
        model = model.to(device)

        train_splits, val_splits, test_splits = get_splits(dataset, fold=fold, **vars(args))
      
        # Instantiate dataset using SlideDataset.
        train_data = DatasetClass(dataset, args.root_path, train_splits, args.task_config, split_key=args.split_key)
        val_data = DatasetClass(dataset, args.root_path, val_splits, args.task_config, split_key=args.split_key) if len(val_splits) > 0 else None
        test_data = DatasetClass(dataset, args.root_path, test_splits, args.task_config, split_key=args.split_key) if len(test_splits) > 0 else None

        # Get the dataloader for the test set.
        _, _, test_loader = get_loader(train_data, val_data, test_data, **vars(args))
      
        # Perform predictions and compute metrics on the test set.
        predictions, metrics = predict_and_save(model, test_loader, device, args, fold)
        CV_predictions.extend(predictions)
        CV_metrics.append(metrics)

    # Use relative paths for saving predictions and metrics.
    predictions_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../prov-gigapath/outputs/braf/braf/run-globalPool-unfreeze_traintype-CV_epoch-5_blr-0.003_BS-32_wd-0.05_ld-0.95_drop-0.5_dropPR-0.1_feat-9",
            "predictions.csv"
        )
    )
    
    metrics_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../prov-gigapath/outputs/braf/braf/run-globalPool-unfreeze_traintype-CV_epoch-5_blr-0.003_BS-32_wd-0.05_ld-0.95_drop-0.5_dropPR-0.1_feat-9",
            "metrics.csv"
        )
    )
    pd.DataFrame(CV_predictions).to_csv(predictions_path, index=False)
    pd.DataFrame(CV_metrics).to_csv(metrics_path, index=False)
 
    print("Predictions saved to", predictions_path)

if __name__ == '__main__':
    main()
