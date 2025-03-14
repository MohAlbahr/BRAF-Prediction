#%%
"""
Module: Data Loader and Transformations for PyT Dataset
Description:
    This module defines:
    - A helper class DatasetWithIndex to wrap a dataset with a custom collate function.
    - A function prepare_data_loader() to process the entire dataset (default: TCGA).
    - Transformation functions for training and validation.
    - A main() function demonstrating how to load a batch and visualize results.
    
    Note: Splitting into train/validation/test is available as commented-out code.
    
Author: Mohamed Albahri
Year: 2024
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
from collections import Counter
import os
import sys
import random

# =============================================================================
# Third-Party Imports
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as T
from tqdm import tqdm

# =============================================================================
# Local Library Imports
# =============================================================================
# Fix import for interactive sessions: add project root two levels up
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')))

from Libraries.DataPreparation.PyTDataset import PyTDataset

# =============================================================================
# DatasetWithIndex Class Definition
# =============================================================================
class DatasetWithIndex:
    def __init__(self, dataset):
        self.dataset = dataset
        self.start_idx = 0  # Initialize the starting index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def custom_collate(self, batch):
        # Filter out invalid samples (None or those containing None)
        filtered_batch = [item for item in batch if item is not None and None not in item]

        # Use start_idx to ensure we have a full batch
        while len(filtered_batch) < len(batch):
            sample = self.dataset[self.start_idx % len(self.dataset)]
            self.start_idx += 1  # Advance the index
            if sample is not None and None not in sample:
                filtered_batch.append(sample)

        slide_id, image, label, coordinates = zip(*filtered_batch)

        # Convert outputs to tensors (slide_id remains a list of strings)
        slide_id = list(slide_id)
        label = torch.stack(label)
        image = torch.stack(image)
        coordinates = torch.stack(coordinates)
        # Debug prints (uncomment if needed)
        # print(f"slide_id: {len(slide_id)}")
        # print(f"labels.shape: {label.shape}")
        # print(f"images.shape: {image.shape}")
        # print(f"coordinates.shape: {coordinates.shape}")
        return slide_id, image, label, coordinates

    def custom_collate_wrapper(self, batch):
        return self.custom_collate(batch)

# =============================================================================
# Data Loader Preparation Function
# =============================================================================
def prepare_data_loader(
        all_dataset: str,
        train_val_dataset: str,
        test_dataset: str,
        batch_size: int,
        num_workers: int,
        prov_transforms: bool,
        image_size: int,
        train_hflip: bool,
        train_vflip: bool,
        train_color_jitter: bool,
        train_subset_fraction: float = 1.0,
        val_subset_fraction: float = 1.0,
        test_subset_fraction: float = 1.0,
        subset_stratify: list = None,
        run_number: int = 0):
    
    # Get transformation pipelines for training and validation.
    train_transform = get_train_transform(prov_transforms, train_hflip, train_vflip, train_color_jitter)
    val_transform = get_val_transform(prov_transforms)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)
    
    ## Process the entire TCGA dataset at once.
    ## (Note: DataTCGA/DataUKE handle splitting internally, but you can also use the commented-out code below for explicit splitting.)
    whole_dataset = PyTDataset(
        source_dataset=all_dataset,  # e.g., 'tcga'
        data_split='all',
        subset_fraction=1.0,
        subset_stratify=None,  
        run_number=run_number,
        transform=val_transform)
    
    whole_dataset = DatasetWithIndex(whole_dataset)
    
    all_loader = DataLoader(
        whole_dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffling not necessary here
        num_workers=num_workers,
        # drop_last=True, 
        collate_fn=whole_dataset.custom_collate_wrapper
    )
    return all_loader

    # ## Use this if you want to split dataset into train/val/test:
    #
    # train_dataset = PyTDataset(
    #     source_dataset=train_val_dataset,
    #     data_split='train',
    #     subset_fraction=train_subset_fraction,
    #     subset_stratify=subset_stratify,
    #     run_number=run_number,
    #     transform=train_transform)
    #
    # val_dataset = PyTDataset(
    #     source_dataset=train_val_dataset,
    #     data_split='val',
    #     subset_fraction=val_subset_fraction,
    #     subset_stratify=subset_stratify,
    #     run_number=run_number,
    #     transform=val_transform)
    #
    # test_dataset = PyTDataset(
    #         source_dataset=test_dataset,
    #         data_split='test',
    #         subset_fraction=test_subset_fraction,
    #         subset_stratify=subset_stratify,
    #         run_number=run_number,
    #         transform=val_transform)
    #
    # train_weights = torch.DoubleTensor(train_dataset.weights)
    # train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
    # print("training dataset size: {}, validation dataset size: {}, test dataset size: {}".format(
    #     len(train_dataset), len(val_dataset), len(test_dataset) if test_dataset else None))
    #
    # train_dataset = DatasetWithIndex(train_dataset)
    # val_dataset = DatasetWithIndex(val_dataset)
    # test_dataset = DatasetWithIndex(test_dataset)
    #
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     drop_last=True,
    #     collate_fn=train_dataset.custom_collate_wrapper
    # )
    #
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     collate_fn=val_dataset.custom_collate_wrapper
    # )
    #
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     collate_fn=test_dataset.custom_collate_wrapper
    # )
    #
    # return train_loader, val_loader, test_loader

# =============================================================================
# Transformation Functions
# =============================================================================
def get_train_transform(prov_transforms=True, random_horizontal_flip=True, random_vertical_flip=True, random_color_jitter=True):
    transforms_list = []
    if prov_transforms:  
        transforms_list.append(T.ToPILImage())  # Convert NumPy array to PIL Image
        transforms_list.append(T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC))
        transforms_list.append(T.CenterCrop((224, 224)))
        transforms_list.append(T.ToTensor())
        transforms_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else: 
        if random_horizontal_flip:
            transforms_list.append(T.RandomHorizontalFlip())
        if random_vertical_flip: 
            transforms_list.append(T.RandomVerticalFlip())
        if random_color_jitter:
            transforms_list.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2))
    return T.Compose(transforms_list)

def get_val_transform(prov_transforms=True):
    transforms_list = []
    if prov_transforms:  
        transforms_list.append(T.ToPILImage())
        transforms_list.append(T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC))
        transforms_list.append(T.CenterCrop((224, 224)))
        transforms_list.append(T.ToTensor())
        transforms_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms_list)

# =============================================================================
# Main Function
# =============================================================================
def main():
    # Example: Process entire TCGA dataset at once (no explicit train/val/test split)
    all_loader = prepare_data_loader(
        all_dataset='tcga',
        train_val_dataset='tcga',
        test_dataset='tcga',
        batch_size=32,
        num_workers=4,
        image_size=224,
        train_hflip=True,
        train_vflip=True,
        train_color_jitter=True,
        train_subset_fraction=1.0,
        val_subset_fraction=1.0,
        test_subset_fraction=1.0,
        subset_stratify=None,
        prov_transforms=False,
        run_number=0
    )

    # Iterate over the loader and display one batch
    for loader in [all_loader]:
        imgs, labels = next(iter(loader))
        imgs = imgs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy().astype(np.int32)

        print('Min value:', imgs.min(), '/', 'Max value:', imgs.max())

        # Rescale images: perform channel-wise multiplication and addition
        for i, val in enumerate([0.21716536, 0.26081574, 0.20723464]):
            imgs[:, i, :, :] *= val
        for i, val in enumerate([0.70322989, 0.53606487, 0.66096631]):
            imgs[:, i, :, :] += val
        imgs *= 255.
        imgs = imgs.astype(np.int32)
        imgs = np.moveaxis(imgs, 1, -1)

        print(sorted(Counter(labels).items()))

        plt.figure(figsize=(8, 5))
        for idx, (patch, label) in enumerate(zip(imgs, labels)):
            ax = plt.subplot(4, 8, idx + 1)
            ax.set_title(label)
            ax.axis('off')
            ax.imshow(patch)
        plt.show()

if __name__ == '__main__':
    main()
# %%
