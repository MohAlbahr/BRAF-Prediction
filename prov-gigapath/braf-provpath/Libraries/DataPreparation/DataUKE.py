# %%
"""
Module: DataUKE Processing
Description: 
    This script defines the DataUKE class for processing UKE dataset slides,
    including creating masks, indexing, and providing utilities for retrieving
    patches. It also contains utility functions for image processing and a main()
    function to demonstrate the functionality.

Author: Mohamed Albahri
Year: 2024
"""

# =============================================================================
# Library Imports
# =============================================================================
import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tabulate
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes
from skimage import filters, morphology

# =============================================================================
# Local Library Imports
# =============================================================================
# Fix import for interactive sessions: add project root two levels up
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')))

from tiffslide import TiffSlide
from Libraries.DataPreparation.DeepHistopathUtils import (
    filter_red_pen, filter_green_pen, filter_blue_pen, filter_grays
)

# =============================================================================
# DataUKE Class Definition
# =============================================================================
class DataUKE:
    def __init__(self, data_split: str = 'all', subset_fraction: float = 1.0, subset_stratify: list = None, run_number: int = 0):
        """
        Initialize DataUKE for UKE dataset processing.

        :param data_split: Data split to prepare ('all', 'train', 'val', 'test').
        :param subset_fraction: Fraction of data to use.
        :param subset_stratify: List for stratification of class distributions.
        :param run_number: Run number for reproducibility.
        """
        # Define the class list and initialize patch count.
        self.class_list = ['no_braf', 'braf']
        self.n_patches = 0

        if data_split not in ['all', 'train', 'val', 'test']:
            raise ValueError('Data split %s not implemented.' % data_split)
       
        # Define directory paths
        self.index_dir = os.path.join("prov-gigapath", "braf-provpath", "Libraries", "DataPreparation", "index")
        self.dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/UKE"))

        # Load existing masks or prepare a new index if necessary
        if os.path.isfile(self.index_dir + '/uke_' + data_split + '_masks.pkl'):
            with open(self.index_dir + '/uke_' + data_split + '_masks.pkl', 'rb') as f:
                self.data_index_masks = pickle.load(f)
        else:
            self.__prepare_index(data_split=data_split)
        
        # Count total patches in the index
        self.__count_index()
        
        # Optionally reduce dataset size by using a subset fraction or stratification
        if subset_fraction != 1.0 or subset_stratify is not None:
            if subset_stratify is not None:
                frac_mul_min = np.min([x / y for x, y in zip(self.n_per_class.values(), subset_stratify)])
                max_n_per_class = {x: int(y * frac_mul_min) for x, y in zip(self.n_per_class.keys(), subset_stratify)}
            else:
                max_n_per_class = {x: y for x, y in self.n_per_class.items()}

            if subset_fraction != 1.0:
                max_n_per_class = {x: int(y * subset_fraction) for x, y in max_n_per_class.items()}

            self.__reduce_index(max_n_per_class, run_number)
            self.__count_index()  # recount after reduction
        
    def __prepare_index(self, data_split, patch_size_target=512, upp_target=0.25):
        """
        Prepare the index for the UKE dataset.

        :param data_split: Data split to prepare ('all', 'train', 'val', 'test').
        :param patch_size_target: Target patch size for tiling WSIs.
        :param upp_target: Relative magnitude based on scanning resolution.
        """
        slide_level = 0
        data_index_masks_all = {}
        data_index_masks_train = {}
        data_index_masks_val = {}
        data_index_masks_test = {}

        # Read mutation data from Excel file with labels
        braf_data = {}
        slide_data = pd.read_excel('../datasets/UKE/Label_Fix.xlsx', skiprows=[0])
        for row_idx in range(len(slide_data)):
            slide_file = slide_data.iloc[row_idx, 1].split('.')[0]
            braf_status_column = slide_data.iloc[row_idx, 2]
            if braf_status_column in [1, 2]:
                braf_data[slide_file] = 'braf'
            elif braf_status_column in [4]:
                braf_data[slide_file] = 'no_braf'
            else:
                print("Did not find mutation information for sample %s." % slide_file)
                continue

        # Read slide file paths from CSV (68 slides)
        data = pd.read_csv("/prov-gigapath/braf-provpath/Libraries/DataPreparation/UKE_slide_labels.csv")
        files_list = data["slide_path"]
        for file_path in tqdm(files_list, desc="Processing UKE slides"):
            file_name = os.path.basename(file_path)
            file_path_split = os.path.dirname(file_path).split(os.sep)
            slide_id = file_name.split('.')[0]
            sample_id = slide_id[:15]
            slide_id = str(slide_id).split('_')[0]

            if slide_id in braf_data:
                braf_status = braf_data[slide_id]
            else:
                print("Did not find mutation information for slide %s." % slide_id)
                continue

            with TiffSlide(file_path) as slide:
                slide_resolution = slide.properties['tiffslide.mpp-x']
                patch_size = int(patch_size_target * (upp_target / slide_resolution))
                slide_dims_original = (slide.properties[f'tiffslide.level[{slide_level}].height'], 
                                         slide.properties[f'tiffslide.level[{slide_level}].width'])
                slide_grid = tuple(x // patch_size for x in slide_dims_original)
                slide_thumbnail = np.array(slide.get_thumbnail(tuple(reversed(slide_grid))))
                slide_grid = slide_thumbnail.shape[0:2]
                slide_dims_rounded = tuple(x * patch_size for x in slide_grid)

            # Parameters for filtering
            smooth_sigma = 10
            thresh_val = 0.8
            min_tissue_size = 2000

            # Convert image to grayscale and threshold
            gray_img = rgb2gray(slide_thumbnail)
            bw_img = thresh_slide(gray_img, thresh_val, sigma=smooth_sigma)
            bw_fill = binary_fill_holes(bw_img)
            bw_remove = remove_small_objects(bw_fill, min_size=min_tissue_size)

            # Apply color filters using pen filters
            color_mask = (filter_red_pen(slide_thumbnail) & 
                          filter_green_pen(slide_thumbnail) & 
                          filter_blue_pen(slide_thumbnail) & 
                          filter_grays(slide_thumbnail))
            
            mask_all = bw_remove & color_mask

            mask_class_index = {class_name: [] for class_name in self.class_list}
            tumor_binary_flatten = mask_all.flatten(order='F')
            mask_class_index[braf_status] = np.where(tumor_binary_flatten == 1)[0].tolist()
                            
            key = file_path_split[-1] + '/' + file_name
            data_index_masks_all[key] = {
                'slide_level': slide_level,
                'patch_size': patch_size,
                'slide_dims_original': slide_dims_original,
                'slide_grid': slide_grid,
                'slide_dims_rounded': slide_dims_rounded,
                'mask_class_index': mask_class_index,
            }
        
        # Save complete masks
        with open(self.index_dir + '/uke_all_masks.pkl', 'wb') as f:
            pickle.dump(data_index_masks_all, f)

        # Split files randomly (with fixed seed) into train, validation, test sets
        file_names_list = [os.path.basename(x) for x in data_index_masks_all.keys()]
        file_names_train, file_names_val = train_test_split(file_names_list, test_size=0.4, random_state=42, shuffle=True)
        file_names_val, file_names_test = train_test_split(file_names_val, test_size=0.5, random_state=42, shuffle=True)

        for key, value in data_index_masks_all.items():
            file_name = os.path.basename(key)
            if file_name in file_names_train:
                data_index_masks_train[key] = value
            elif file_name in file_names_val:
                data_index_masks_val[key] = value
            elif file_name in file_names_test:
                data_index_masks_test[key] = value

        with open(self.index_dir + '/uke_train_masks.pkl', 'wb') as f:
            pickle.dump(data_index_masks_train, f)
        with open(self.index_dir + '/uke_val_masks.pkl', 'wb') as f:
            pickle.dump(data_index_masks_val, f)
        with open(self.index_dir + '/uke_test_masks.pkl', 'wb') as f:
            pickle.dump(data_index_masks_test, f)

        if data_split == 'all':
            self.data_index_masks = data_index_masks_all
        elif data_split == 'train':
            self.data_index_masks = data_index_masks_train
        elif data_split == 'val':
            self.data_index_masks = data_index_masks_val
        elif data_split == 'test':
            self.data_index_masks = data_index_masks_test

    def get_item(self, idx):
        """
        Retrieve a patch and its label given a global index.

        :param idx: Index of the patch.
        :return: Tuple (file_name, patch, label, coordinates).
        """
        file_name, patch_class, patch_index = self.__get_index(idx)
        mask_class_index = self.data_index_masks[file_name]['mask_class_index']
        slide_grid = self.data_index_masks[file_name]['slide_grid']
        patch_size = self.data_index_masks[file_name]['patch_size']

        patch_pos = mask_class_index[patch_class][patch_index]
        idx_y, idx_x = np.unravel_index(patch_pos, slide_grid, order='F')

        with TiffSlide(os.path.join(self.dataset_dir, file_name)) as slide:
            patch = slide.read_region((idx_x * patch_size, idx_y * patch_size), 0, (patch_size, patch_size), as_array=True)

        if patch.dtype == np.uint16: 
            patch = (patch // 256).astype(np.uint8)
        
        label = self.class_list.index(patch_class)
        coordinates = [idx_x, idx_y]
    
        return file_name, patch, label, coordinates

    def __get_index(self, idx):
        """
        Convert a global patch index into a tuple (file_name, patch_class, local index).

        :param idx: Global patch index.
        :return: Tuple (file_name, patch_class, local index).
        """
        if idx < 0:
            raise Exception("The specified index is negative.") 
        
        for patch_class in self.class_list:
            for file_name in self.data_index_masks.keys():
                mask_len = len(self.data_index_masks[file_name]['mask_class_index'][patch_class])
                if idx < mask_len:
                    return file_name, patch_class, idx
                else:
                    idx -= mask_len

        raise Exception("The specified index was not found.")

    def __count_index(self):
        """
        Count the total number of patches per class.
        """
        self.n_per_class = {x: 0 for x in self.class_list}
        for file_name in self.data_index_masks.keys():
            for tissue_class in self.class_list:
                self.n_per_class[tissue_class] += len(self.data_index_masks[file_name]['mask_class_index'][tissue_class])
        self.n_patches = sum(self.n_per_class.values())

    def __reduce_index(self, max_n_per_class, run_number):
        """
        Reduce the index to a maximum number of patches per class.

        :param max_n_per_class: Dictionary with maximum number of patches per class.
        :param run_number: Run number for reproducibility.
        """
        for patch_class in self.class_list:
            delete_per_class = [True] * (self.n_per_class[patch_class] - max_n_per_class[patch_class]) + \
                               [False] * max_n_per_class[patch_class]
            random.Random(run_number).shuffle(delete_per_class)
            for file_name in self.data_index_masks.keys():
                mask_len = len(self.data_index_masks[file_name]['mask_class_index'][patch_class])
                delete_vector = [delete_per_class.pop() for _ in range(mask_len)]
                delete_idx = np.nonzero(delete_vector)[0]
                for i in delete_idx[::-1]:
                    del self.data_index_masks[file_name]['mask_class_index'][patch_class][i]

    def print_dataset_stats(self, show_masks=True):
        """
        Print dataset statistics (number of patches per class per file).
        Optionally, display the slide thumbnail with overlayed masks.
        """
        data = []
        for file_name in self.data_index_masks.keys():
            row = [file_name]
            for tissue_class in self.class_list:
                row.append(len(self.data_index_masks[file_name]['mask_class_index'][tissue_class]))
            data.append(row)
        display(HTML(tabulate.tabulate(data, headers=['file'] + self.class_list, tablefmt='html')))

        if show_masks:
            for file_name in self.data_index_masks.keys():
                slide_grid = self.data_index_masks[file_name]['slide_grid']
                mask_img = np.zeros(slide_grid)
                for tissue_class in self.class_list:
                    for patch_pos in self.data_index_masks[file_name]['mask_class_index'][tissue_class]:
                        idx_y, idx_x = np.unravel_index(patch_pos, slide_grid, order='F')
                        mask_img[idx_y, idx_x] = 255

                with TiffSlide(os.path.join(self.dataset_dir, file_name)) as slide:
                    slide_thumbnail = np.array(slide.get_thumbnail(tuple(reversed(slide_grid))))
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(file_name)
                ax1.imshow(slide_thumbnail)
                ax2.imshow(mask_img)
                fig.tight_layout()
                fig.show()
                plt.show()
                plt.close(fig)

# =============================================================================
# Utility Functions
# =============================================================================
def rgb2gray(img):
    """Convert an RGB image to grayscale."""
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])

def calculate_brightness(img):
    """Calculate the average brightness of an image."""
    return np.mean(rgb2gray(img))

def thresh_slide(gray, thresh_val, sigma=13):
    """Threshold a grayscale image to a binary image."""
    if not (0 <= gray.min() <= 255 and 0 <= gray.max() <= 255):
        raise ValueError("Input image values should be in the range [0, 255].")
    smooth = filters.gaussian(gray, sigma=sigma)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min()) * 255.0
    binary_img = smooth < thresh_val * 255.0
    return binary_img

def fill_tissue_holes(bw_img):
    """Fill holes in a binary tissue image."""
    return binary_fill_holes(bw_img)

def remove_small_tissue(bw_img, min_size=10000):
    """Remove small objects from a binary tissue image."""
    return morphology.remove_small_objects(bw_img, min_size=min_size, connectivity=8)

def filter_fat_tissue(gray_img, fat_thresh=0.9):
    """Filter out fat tissue from a grayscale image."""
    fat_mask = gray_img > fat_thresh
    gray_img[fat_mask] = 0
    return gray_img

# =============================================================================
# Main Execution Function
# =============================================================================
def main():
    """
    Main function demonstrating usage of the DataUKE class.
    Initializes the dataset, prints statistics, and displays example patches.
    """
    uke = DataUKE(data_split='all')
    print('Number of patches:', uke.n_patches, uke.n_per_class)
    uke.print_dataset_stats()

if __name__ == "__main__":
    main()
