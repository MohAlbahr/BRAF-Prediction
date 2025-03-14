import numpy as np
import os

def one_hot_encode(x, n_classes):
    labels = np.zeros(shape=(len(x), n_classes))
    for i, x_i in enumerate(x):
        labels[i, x_i] = 1
    return labels

# ----------------------------------------------------------------------------------------------------
# Transfer-Learning-Library/examples/task_adaptation/image_classification/utils.py
# ----------------------------------------------------------------------------------------------------

class ModifiedLogger:
    # source: https://github.com/thuml/Transfer-Learning-Library/blob/0fdc06ca87c71fbf784d58e7388cf03a3f13bf00/tllib/utils/logger.py

    def __init__(self, root):
        self.root = root
        self.checkpoint_directory = os.path.join(self.root, 'checkpoints')

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.checkpoint_directory, exist_ok=True)

    def get_checkpoint_path(self, name):
        return os.path.join(self.checkpoint_directory, str(name) + ".pth")