from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import roc_curve, auc, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay

from Libraries.utils import one_hot_encode

def plot_confusion_matrix(outputs, targets, num_classes, class_labels):
    # calculate other representations
    outputs_max = np.argmax(outputs, axis=1)
    
    # ------------------------------------
    # Plot confusion matrix
    # ------------------------------------

    fig, ax = plt.subplots(figsize=(6, 6), dpi=500)

    cm = confusion_matrix(
        targets,
        outputs_max,
        labels=range(num_classes),
    )
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels,
    )
    disp.plot(
        ax=ax,
        xticks_rotation='vertical',
    )

    fig.tight_layout()

    return fig

def plot_roc_curve(outputs, targets, num_classes, class_labels):
    # calculate other representations
    targets_onehot = one_hot_encode(targets, num_classes)

    if num_classes > 2:
        # ------------------------------------
        # Calculate micro-average ROC
        # ------------------------------------
        
        fpr, tpr, roc_auc = dict(), dict(), dict()

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(targets_onehot.ravel(), outputs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # ------------------------------------
        # Calculate macro-average ROC
        # ------------------------------------
        
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(targets_onehot[:, i], outputs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        fpr_grid = np.linspace(0.0, 1.0, 1000)

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)
        
        for i in range(num_classes):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])

        # Average it and compute AUC
        mean_tpr /= num_classes

        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # ------------------------------------
    # Plot ROC curve
    # ------------------------------------

    fig, ax = plt.subplots(figsize=(6, 6), dpi=500)

    if num_classes > 2:
        ax.plot(
            fpr["micro"],
            tpr["micro"],
            label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        ax.plot(
            fpr["macro"],
            tpr["macro"],
            label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(sns.color_palette('colorblind'))
        for class_id, color in zip(range(num_classes), colors):
            RocCurveDisplay.from_predictions(
                targets_onehot[:, class_id],
                outputs[:, class_id],
                name=f"ROC curve for {class_labels[class_id]}",
                color=color,
                ax=ax,
            )
    elif num_classes == 2:
        colors = sns.color_palette('colorblind')
        RocCurveDisplay.from_predictions(
            targets_onehot[:, 1],
            outputs[:, 1],
            name=f"ROC curve",
            color=colors[0],
            ax=ax,
        )
    else:
        raise ValueError('num_classes with value %i not supported.' % num_classes)

    ax.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    ax.axis("square")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend()

    fig.tight_layout()

    return fig