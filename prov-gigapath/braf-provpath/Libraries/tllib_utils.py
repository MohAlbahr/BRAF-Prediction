import copy
import numpy as np
import time
import wandb

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import accuracy

# import timm
from Libraries.pytorch_utils import one_hot_encode, calculate_metrics

from tllib.utils.meter import AverageMeter, ProgressMeter

# ----------------------------------------------------------------------------------------------------
# Transfer-Learning-Library/examples/task_adaptation/image_classification/utils.py
# ----------------------------------------------------------------------------------------------------

def validate(val_loader, model, device, visualize=None) -> float:
    # source: https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/task_adaptation/image_classification/utils.py

    batch_time = AverageMeter('Time', ':4.2f')
    losses = AverageMeter('Loss', ':4.4e')
    top1 = AverageMeter('Acc@1', ':4.4f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1], prefix='Test: ')

    # list where to store all results
    outputs = []
    targets = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            if wandb.config['architecture'] == 'efficientnet':
                outputs_tuple = model(images)
                # Ensure that we're accessing the correct output
                output = outputs_tuple if isinstance(outputs_tuple, torch.Tensor) else outputs_tuple.logits
                output_act = F.softmax(output, dim=1)
            else:
                # compute forward pass
                output = model(images)
                output_act = F.softmax(output, dim=1)

            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output_act, target, task="multiclass", num_classes=wandb.config['num_classes'], top_k=1)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # keep all results
            outputs.extend(output_act.detach().cpu().numpy())
            targets.extend(target.detach().cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % wandb.config['print-freq'] == 0:
                progress.display(i)
                if visualize is not None:
                    visualize(images[0], "val_{}".format(i))

    # calculate metrics
    out_metrics = calculate_metrics(outputs, targets, wandb.config['num_classes'], 'val_')
    print('[Validation] Acc: %.4f, B.Acc: %.4f, AUROC: %.4f, F1: %.4f, Prec: %.4f, Rec: %.4f' 
        % (out_metrics['val_accuracy'], out_metrics['val_balanced_accuracy'], out_metrics['val_auroc'], out_metrics['val_f1_score'], out_metrics['val_precision'], out_metrics['val_recall']))

    return losses.avg, out_metrics