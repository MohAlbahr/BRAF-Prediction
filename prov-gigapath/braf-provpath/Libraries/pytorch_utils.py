import numpy as np
import os
import pickle
from tqdm import tqdm
import wandb

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

import torch
import torch.nn.functional as F

from Libraries.utils import one_hot_encode
from Libraries.wandb_utils import plot_confusion_matrix, plot_roc_curve

def calculate_metrics(outputs, targets, num_classes, prefix):
    # convert to array if necessary
    if not isinstance(outputs, np.ndarray):
        outputs = np.array(outputs)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    
    # calculate other representations
    outputs_max = np.argmax(outputs, axis=1)
    targets_onehot = one_hot_encode(targets, num_classes)

    # caluclate metrics
    out_metrics = {}
    out_metrics[prefix + 'accuracy'] = accuracy_score(targets, outputs_max)
    out_metrics[prefix + 'balanced_accuracy'] = balanced_accuracy_score(targets, outputs_max)
    out_metrics[prefix + 'auroc'] = roc_auc_score(targets_onehot, outputs, average="macro")
    out_metrics[prefix + 'f1_score'] = f1_score(targets, outputs_max, average="macro", zero_division=0)
    out_metrics[prefix + 'precision'] = precision_score(targets, outputs_max, average="macro", zero_division=0)
    out_metrics[prefix + 'recall'] = recall_score(targets, outputs_max, average="macro", zero_division=0)

    return out_metrics

def test_classifier(model, test_loader, device, run_data_dir, calculate_wsi_metrics=False, **kwargs):
    # get number of classes
    num_classes = len(test_loader.dataset.dataset_object.class_list)

    # list for predictions and labels
    outputs = []
    targets = []

    # switch to evaluate mode
    model.eval()

    # SpotTune/MultiTune-specific
    if 'agent' in kwargs and kwargs['agent'] is not None:
        kwargs['agent'].eval()

    with torch.no_grad():
        for images, target in tqdm(test_loader, desc="Test split"):
            # Move to GPU if necessary
            images = images.to(device)
            target = target.to(device)

            # additional foward args
            forward_args = {}

            # SpotTune/MultiTune-specific
            if 'use_multitune' in kwargs:
                if 'agent' in kwargs and kwargs['agent'] is not None:
                    probs = kwargs['agent'](images)
                    action = F.gumbel_softmax(probs.view(probs.size(0), -1, 2), tau=5, hard=True)
                    policy = action[:,:,1]
                else:
                    policy = None
                forward_args['use_multitune'] = kwargs['use_multitune']
                forward_args['policy'] = policy

            # compute output
            
            if wandb.config['architecture'] == 'efficientnet':
                outputs_tuple = model(images , **forward_args)
                # output = outputs_tuple.logits if isinstance(outputs_tuple, torch.nn.Module) else outputs_tuple[0]
                output = outputs_tuple if isinstance(outputs_tuple, torch.Tensor) else outputs_tuple.logits

            else:
                # compute forward pass
                output = model(images, **forward_args)
            
            output_act = F.softmax(output, dim=1)

            # keep all results
            outputs.extend(output_act.detach().cpu().numpy())
            targets.extend(target.detach().cpu().numpy())

    # convert to array
    outputs = np.array(outputs)
    targets = np.array(targets)

    # save results
    with open(os.path.join(run_data_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump({'outputs': outputs, 'targets': targets}, f)
    
    # ------------------------------------
    # Patch-based metrics
    # ------------------------------------
    
    # calculate metrics
    out_metrics = calculate_metrics(outputs, targets, num_classes, prefix='test_')
    
    # log metrics to WandB
    for metric_name, metric_value in out_metrics.items():
        wandb.run.summary[metric_name] = metric_value

    # create confusion matrix
    wandb.log({'test_conf_mat' : wandb.Image(plot_confusion_matrix(outputs, targets, num_classes, test_loader.dataset.dataset_object.class_list))})

    # create ROC curve
    wandb.log({"test_roc_curve" : wandb.Image(plot_roc_curve(outputs, targets, num_classes, test_loader.dataset.dataset_object.class_list))})

    # ------------------------------------
    # WSI-based metrics
    # ------------------------------------

    if calculate_wsi_metrics:
        outputs_temp = outputs.copy()

        # list with WSI probabilities and classes
        wsi_probs = []
        wsi_classes = []

        for patch_class in test_loader.dataset.dataset_object.class_list:
            for file_name in test_loader.dataset.dataset_object.data_index_masks.keys():
                mask_len = len(test_loader.dataset.dataset_object.data_index_masks[file_name]['mask_class_index'][patch_class])
                if mask_len > 0:
                    # get probabilities from array
                    probs_logits = outputs_temp[:mask_len]
                    outputs_temp = outputs_temp[mask_len:]

                    # calculate mean
                    wsi_probs.append(np.mean(probs_logits, axis=0))

                    # get class index
                    wsi_classes.append(test_loader.dataset.dataset_object.class_list.index(patch_class))

        # convert to array
        wsi_probs = np.array(wsi_probs)

        # calculate metrics
        out_metrics = calculate_metrics(wsi_probs, wsi_classes, num_classes, prefix='test_wsi_')
        
        # log metrics to WandB
        for metric_name, metric_value in out_metrics.items():
            wandb.run.summary[metric_name] = metric_value

        # create confusion matrix
        wandb.log({'test_wsi_conf_mat' : wandb.Image(plot_confusion_matrix(wsi_probs, wsi_classes, num_classes, test_loader.dataset.dataset_object.class_list))})

        # create ROC curve
        wandb.log({"test_wsi_roc_curve" : wandb.Image(plot_roc_curve(wsi_probs, wsi_classes, num_classes, test_loader.dataset.dataset_object.class_list))})