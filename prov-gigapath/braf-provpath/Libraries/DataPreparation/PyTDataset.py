# %%
"""
Module: PyT Dataset for Stain Normalization and Patch Sampling
Description:
    This module defines the PyTDataset class for loading and processing patches from either the TCGA or UKE datasets.
    It includes functionality for stain normalization using staintools, brightness filtering, and patch retrieval.
    Also included is a PyTSubset class to create dataset subsets with weighted sampling.
    
Author: Mohamed Albahri
Year: 2024
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import os
import sys
import random
import argparse

# =============================================================================
# Third-Party Imports
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import staintools
from tqdm import tqdm

# =============================================================================
# Local Library Imports
# =============================================================================
# Fix import for interactive sessions
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')))

# Import stain filters from our local module
from Libraries.DataPreparation.DeepHistopathUtils import (
    filter_red_pen, filter_green_pen, filter_blue_pen, filter_grays
)

# =============================================================================
# Utility Functions
# =============================================================================
def calculate_brightness(img):
    """Calculate the average brightness of an image.
    
    Assumes image is in HWC format (height, width, channels).
    """
    if img.ndim == 3 and img.shape[2] == 3:
        gray_img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        return np.mean(gray_img)
    else:
        raise ValueError("Input image must have 3 channels (RGB).")

def check_image_dimensions(image):
    """Raise an error if image dimensions are not 224x224."""
    if image.shape[1] != 224 or image.shape[2] != 224:
        raise ValueError(f"Incorrect image dimensions: {image.shape}")

# =============================================================================
# PyTDataset Class Definition
# =============================================================================
class PyTDataset(Dataset):
    def __init__(self, source_dataset: str = 'tcga', data_split: str = 'all', subset_fraction: float = 1.0, 
                 subset_stratify: list = None, run_number: int = 0, transform=None, brightness_threshold: float = 0.8):
        """
        Initialize the PyTDataset.

        :param source_dataset: Identifier for the dataset ('tcga' or 'uke').
        :param data_split: Data split to use ('all', 'train', 'val', 'test').
        :param subset_fraction: Fraction of data to use.
        :param subset_stratify: List for stratification (if needed).
        :param run_number: Run number for reproducibility.
        :param transform: Transformations to apply to images.
        :param brightness_threshold: Threshold for brightness filtering.
        """
        data_class_args = {
            'data_split': data_split,
            'subset_fraction': subset_fraction,
            'subset_stratify': subset_stratify,
            'run_number': run_number,
        }
        self.data_split = data_split

        # Import the appropriate data class based on the source_dataset argument.
        if source_dataset.lower() == 'tcga':
            from Libraries.DataPreparation.DataTCGA import DataTCGA as DataClass
        elif source_dataset.lower() == 'uke':
            from Libraries.DataPreparation.DataUKE import DataUKE as DataClass
        else:
            raise Exception('This dataset is not supported yet.')
        
        # Create the data class object.
        self.dataset_object = DataClass(**data_class_args)

        self.transform = transform
        self.brightness_threshold = brightness_threshold

        # Load target image for stain normalization.
        target_image_path = os.path.join(os.path.dirname(__file__), 'source_stain_level1.png')
        self.target_image = staintools.read_image(target_image_path)

        # Initialize and fit the stain normalizer.
        self.normalizer = staintools.StainNormalizer(method='macenko')
        self.normalizer.fit(self.target_image)

        # Build target list (labels) based on the data object's class list.
        self.targets = []
        for patch_class in self.dataset_object.class_list:
            self.targets += [self.dataset_object.class_list.index(patch_class)] * self.dataset_object.n_per_class[patch_class]

        ## Compute weights for each patch.
        # self.weights = []
        # for patch_class in self.dataset_object.class_list:
        #     self.weights += [float(self.dataset_object.n_patches) / float(self.dataset_object.n_per_class[patch_class])] * self.dataset_object.n_per_class[patch_class]

        print("Number of patches before brightness filtering:", self.dataset_object.n_patches)

    def __len__(self):
        return self.dataset_object.n_patches
    
    def __getitem__(self, idx):
        try:
            slide_id, image, label, coordinates = self.dataset_object.get_item(idx)
            
            # Convert 16-bit images to 8-bit.
            if image.dtype == np.uint16: 
                image = (image // 256).astype(np.uint8)
            
            try:
                # Skip images that are too bright.
                if calculate_brightness(image) > self.brightness_threshold * 255.0:
                    return None, None, None, None
                else:
                    # Apply stain normalization.
                    image = self.normalizer.transform(image)
            except Exception as e:
                if "Empty tissue mask computed" in str(e):
                    print(f"Error during stain normalization: {e}")
                else:
                    raise e
                return None, None, None, None
            
            # Apply optional transformations.
            if self.transform:
                image = self.transform(image)
            
            # Convert image and labels to tensors.
            image = torch.from_numpy(np.array(image))
            label = torch.tensor(label, dtype=torch.long)
            coordinates = torch.from_numpy(np.array(coordinates)).float()

            return slide_id, image, label, coordinates
        except Exception as e:
            print(f"Error fetching item {idx}: {e}")
            return None, None, None, None

# =============================================================================
# PyTSubset Class Definition
# =============================================================================
class PyTSubset(Dataset):
    """
    A subset of a dataset. Adapted from the PyTorch Dataset class.
    """
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.targets = [self.dataset.targets[i] for i in self.indices]
        self.weights = [self.dataset.weights[i] for i in self.indices]

    def __getitem__(self, idx):
        if isinstance(idx, list):
            im, labels = self.dataset[[self.indices[i] for i in idx]]
        else:
            im, labels = self.dataset[self.indices[idx]]

        if self.transform:
            im = self.transform(im)
        
        return im, labels

    def __len__(self):
        return len(self.indices)

# =============================================================================
# Main Function
# =============================================================================
def main():
    # Instantiate PyTDataset with the TCGA dataset.
    ft_dataset = PyTDataset(
        source_dataset='tcga',
        data_split='all',
        subset_fraction=1.0,
        subset_stratify=None,
        run_number=0,
        transform=T.Resize((224, 224), antialias=True)
    )
    ft_weights = torch.DoubleTensor(ft_dataset.weights)
   
    def custom_collate(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        if len(batch) == 0:
            return [], torch.tensor([]), torch.tensor([]), torch.tensor([])
        return torch.utils.data.dataloader.default_collate(batch)

    def custom_collate_wrapper(batch):
        return custom_collate(batch)

    ft_sampler = WeightedRandomSampler(ft_weights, len(ft_weights))
    ft_loader = DataLoader(ft_dataset, batch_size=1024, shuffle=False, sampler=ft_sampler, collate_fn=custom_collate_wrapper)
    
    def show_batch(images, labels):
        fig, axs = plt.subplots(8, 8, figsize=(15, 8))
        for i, (img, lbl) in enumerate(zip(images, labels)):
            ax = axs[i // 8, i % 8]
            img = img.permute(1, 2, 0).numpy()  # Convert from tensor to numpy array
            img = img * 255  # Rescale to [0, 255]
            ax.imshow(img.astype(np.uint8))
            ax.set_title(f'Label: {lbl.item()}')
            ax.axis('off')
        plt.show()
   
    total_patches = 0
    for batch in ft_loader:
        images, labels = batch
        total_patches += len(images)

    print("Number of patches after brightness filtering:", total_patches)

if __name__ == '__main__':
    main()
# %%
