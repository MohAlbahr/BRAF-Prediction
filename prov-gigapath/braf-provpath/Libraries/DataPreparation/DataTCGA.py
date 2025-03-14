# %%
"""
Module for processing TCGA data for BRAF mutation prediction.

This script includes:
 - Data indexing and mask creation for slides.
 - Utility functions for image processing (e.g., thresholding, hole-filling).
 - A DataTCGA class for managing the dataset.
 - A main() function for demo/testing purposes.

Author: Mohamed Albahri
Year: 2024
"""

# =============================================================================
# Library Imports
# =============================================================================
import glob
import gzip
import json
import os
import pickle
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes
from skimage import filters, morphology

# =============================================================================
# Local Library Imports
# =============================================================================
# Fix import for interactive session
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')))
    
from Libraries.DataPreparation.DeepHistopathUtils import (
    filter_red_pen, filter_green_pen, filter_blue_pen, filter_grays
)
from tiffslide import TiffSlide

# =============================================================================
# Environment Setup
# =============================================================================
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# DataTCGA Class Definition
# =============================================================================
class DataTCGA:
    def __init__(self, data_split: str = 'all', subset_fraction: float = 1.0, subset_stratify: list = None, run_number: int = 0):
        """
        Initialize the DataTCGA object.
        
        :param data_split: Data split to use ('all', 'train', 'val', 'test').
        :param subset_fraction: Fraction of data to use.
        :param subset_stratify: List for stratification.
        :param run_number: Run number (for reproducibility).
        """
        self.class_list = ['no_braf', 'braf']  # List of classes
        self.slide_level = 0
        
        if data_split not in ['all', 'train', 'val', 'test']:
            raise ValueError('Data split %s not implemented.' % data_split)

        # Set directory paths
        self.class_dir = os.path.dirname(os.path.realpath(__file__))
        self.index_dir = os.path.abspath(os.path.join(self.class_dir, 'index'))

        # Load existing masks or prepare new index if needed
        mask_file = os.path.join(self.index_dir, f'tcga_{data_split}_masks.pkl')
        if os.path.isfile(mask_file):
            with open(mask_file, 'rb') as f:
                self.data_index_masks = pickle.load(f)
        else:
            self.__prepare_index(data_split=data_split)

        # Set dataset (images) directory
        self.dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/TCGA-SKCM'))

        # Count total patches
        self.__count_index()

        # Optionally reduce dataset size using a subset fraction or stratification
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
        Prepare the index for the dataset.
        
        :param data_split: Data split to prepare ('all', 'train', 'val', 'test').
        :param patch_size_target: Target patch size for WSIs tiling.
        :param upp_target: Relative magnitude based on scanning resolution.
        """
        data_index_masks_all = {}
        data_index_masks_train = {}
        data_index_masks_val = {}
        data_index_masks_test = {}

        # Read mutation data from MAF files
        braf_data = {}
        for file in tqdm(glob.glob('../datasets/TCGA-SKCM/MAF/**/*.maf.gz', recursive=True), desc="Reading MAF files"):
            with gzip.open(file, 'r') as f:
                file_header = [f.readline().strip().decode("utf-8") for _ in range(7)]
            tumor_aliquot = file_header[6].split(' ')[1]
            data = pd.read_csv(file, compression='gzip', sep='\t', header=0, skiprows=7, low_memory=False)
            braf_data[tumor_aliquot] = 'braf' if 'BRAF' in data.Hugo_Symbol.unique() else 'no_braf'
        print("Maf reading done!")

        # Read biospecimen JSON
        with open('../datasets/TCGA-SKCM/MAF/biospecimen.cases_selection.2023-11-16.json', 'r') as f:
            biospecimen_data = json.load(f)

        # Map slides to aliquots
        sample_aliquot_assignment = {}
        for case in biospecimen_data:
            for sample in case['samples']:
                for portion in sample['portions']:
                    if 'slides' in portion and 'analytes' in portion:
                        slide_list, aliquot_list = [], []
                        for slide in portion['slides']:
                            slide_list.append(slide['submitter_id'])
                        for analyte in portion['analytes']:
                            for aliquot in analyte['aliquots']:
                                if aliquot['aliquot_id'] in braf_data:
                                    aliquot_list.append(aliquot['aliquot_id'])
                        if len(slide_list) == 0 or len(aliquot_list) == 0:
                            continue  # skip if missing info
                        elif len(slide_list) == 1 and len(aliquot_list) == 1:
                            sample_aliquot_assignment[slide_list[0][:15]] = aliquot_list[0]
                        else:
                            raise Exception('Ambiguous assignment for slides %s and aliquots %s.' % (slide_list, aliquot_list))

        # Get list of slide files
        files_list = sorted(glob.glob(self.dataset_dir + '/**/*.svs'))
        print("Slide reading starts now:")
        for file_path in tqdm(files_list, desc="Processing slides"):
            file_name = os.path.basename(file_path)
            file_path_split = os.path.dirname(file_path).split(os.sep)
            slide_id = file_name.split('.')[0]
            sample_id = slide_id[:15]

            if sample_id in sample_aliquot_assignment:
                braf_status = braf_data[sample_aliquot_assignment[sample_id]]
            else:
                print("Did not find mutation info for sample %s." % sample_id)
                continue

            with TiffSlide(file_path) as slide:
                slide_resolution = slide.properties['tiffslide.mpp-x']
                patch_size = int(patch_size_target * (upp_target / slide_resolution))
                slide_dims_original = (slide.properties['tiffslide.level[' + str(self.slide_level) + '].height'], 
                                         slide.properties['tiffslide.level[' + str(self.slide_level) + '].width'])
                slide_grid = tuple(x // patch_size for x in slide_dims_original)
                slide_thumbnail = np.array(slide.get_thumbnail(tuple(reversed(slide_grid))))
                slide_grid = slide_thumbnail.shape[0:2]
                slide_dims_rounded = tuple(x * patch_size for x in slide_grid)

                # Filtering parameters
                smooth_sigma = 10
                thresh_val = 0.8
                min_tissue_size = 2000

                # Process image: grayscale, threshold, fill holes, remove small objects
                gray_img = rgb2gray(slide_thumbnail)
                bw_img = thresh_slide(gray_img, thresh_val, sigma=smooth_sigma)
                bw_fill = fill_tissue_holes(bw_img)
                bw_remove = remove_small_tissue(bw_fill, min_tissue_size)

            # Apply color filters using pen filters
            color_mask = filter_red_pen(slide_thumbnail) & \
                         filter_green_pen(slide_thumbnail) & \
                         filter_blue_pen(slide_thumbnail) & \
                         filter_grays(slide_thumbnail)
            
            mask_all = bw_remove & color_mask

            mask_class_index = {class_name: [] for class_name in self.class_list}
            mask_class_index[braf_status] = []
            tumor_binary_flatten = mask_all.flatten(order='F')
            mask_class_index[braf_status] = np.where(tumor_binary_flatten == 1)[0].tolist()
                            
            data_index_masks_all[file_path_split[-1] + '/' + file_name] = {
                'slide_level': self.slide_level,
                'patch_size': patch_size,
                'slide_dims_original': slide_dims_original,
                'slide_grid': slide_grid,
                'slide_dims_rounded': slide_dims_rounded,
                'mask_class_index': mask_class_index,
            }
        
        # Save all masks to file
        with open(os.path.join(self.index_dir, 'tcga_' + data_split + '_masks.pkl'), 'wb') as f:
            pickle.dump(data_index_masks_all, f)

        # Split dataset into train, validation, test
        file_names_list = [os.path.basename(x) for x in data_index_masks_all.keys()]
        file_names_train, file_names_val = train_test_split(file_names_list, test_size=0.3, random_state=42, shuffle=True)
        file_names_val, file_names_test = train_test_split(file_names_val, test_size=0.5, random_state=42, shuffle=True)

        data_index_masks_train = {}
        data_index_masks_val = {}
        data_index_masks_test = {}

        for key, value in data_index_masks_all.items():
            file_name = os.path.basename(key)
            if file_name in file_names_train:
                data_index_masks_train[key] = value
            elif file_name in file_names_val:
                data_index_masks_val[key] = value
            elif file_name in file_names_test:
                data_index_masks_test[key] = value

        with open(os.path.join(self.index_dir, 'tcga_train_masks.pkl'), 'wb') as f:
            pickle.dump(data_index_masks_train, f)
        with open(os.path.join(self.index_dir, 'tcga_val_masks.pkl'), 'wb') as f:
            pickle.dump(data_index_masks_val, f)
        with open(os.path.join(self.index_dir, 'tcga_test_masks.pkl'), 'wb') as f:
            pickle.dump(data_index_masks_test, f)

        # Set the appropriate masks based on data_split
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
        Retrieve a patch and its label based on a global index.
        
        :param idx: Global index of the patch.
        :return: Tuple (file_name, patch, label, coordinates)
        """
        file_name, patch_class, patch_index = self.__get_index(idx)
        mask_class_index = self.data_index_masks[file_name]['mask_class_index']
        slide_grid = self.data_index_masks[file_name]['slide_grid']
        patch_size = self.data_index_masks[file_name]['patch_size']
        patch_pos = mask_class_index[patch_class][patch_index]
        idx_y, idx_x = np.unravel_index(patch_pos, slide_grid, order='F')

        with TiffSlide(os.path.join(self.dataset_dir, file_name)) as slide:
            patch = slide.read_region((idx_x * patch_size, idx_y * patch_size), self.slide_level, (patch_size, patch_size), as_array=True)

        if patch.dtype == np.uint16:
            patch = (patch // 256).astype(np.uint8)
        
        label = self.class_list.index(patch_class)
        coordinates = [idx_x, idx_y]
    
        return file_name, patch, label, coordinates

    def __get_index(self, idx):
        """
        Convert a global patch index into (file_name, patch_class, local_index).
        
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
        Count the total number of patches for each class.
        """
        self.n_per_class = {x: 0 for x in self.class_list}
        for file_name in self.data_index_masks.keys():
            for tissue_class in self.class_list:
                self.n_per_class[tissue_class] += len(self.data_index_masks[file_name]['mask_class_index'][tissue_class])
        self.n_patches = sum(self.n_per_class.values())

    def __reduce_index(self, max_n_per_class, run_number):
        """
        Reduce the index to a maximum number of patches per class.

        :param max_n_per_class: Dict with maximum patches per class.
        :param run_number: Run number for reproducibility.
        """
        for patch_class in self.class_list:
            delete_per_class = [True] * (self.n_per_class[patch_class] - max_n_per_class[patch_class]) + [False] * max_n_per_class[patch_class]
            random.Random(run_number).shuffle(delete_per_class)

            for file_name in self.data_index_masks.keys():
                mask_len = len(self.data_index_masks[file_name]['mask_class_index'][patch_class])
                delete_vector = [delete_per_class.pop() for _ in range(mask_len)]
                delete_idx = np.nonzero(delete_vector)[0]
                for i in delete_idx[::-1]:
                    del self.data_index_masks[file_name]['mask_class_index'][patch_class][i]

    def print_dataset_stats(self, show_masks=True):
        """
        Print statistics on the dataset: number of patches per file per class.
        Optionally, display masks overlayed on slide thumbnails.
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
    """Apply Gaussian smoothing and thresholding to a grayscale image."""
    smooth = filters.gaussian(gray, sigma=sigma)
    return smooth > thresh_val

def fill_tissue_holes(bw_img):
    """Fill holes in a binary tissue image."""
    return binary_fill_holes(bw_img)

def remove_small_tissue(bw_img, min_size=10000):
    """Remove small objects from a binary tissue image."""
    return morphology.remove_small_objects(bw_img, min_size=min_size, connectivity=8)

def filter_fat_tissue(gray_img, fat_thresh=0.9):
    """Filter out fat tissues from a grayscale image."""
    fat_mask = gray_img > fat_thresh
    gray_img[fat_mask] = 0
    return gray_img

# =============================================================================
# Main Execution Function
# =============================================================================
def main():
    """
    Main function to demonstrate DataTCGA usage:
    - Initializes the DataTCGA object.
    - Prints dataset statistics.
    - Displays example patches per class.
    """
    tcga = DataTCGA(data_split='all')
    print('Number of patches:', tcga.n_patches, tcga.n_per_class)
    tcga.print_dataset_stats()

    generator = random.Random(0)
    c_pos = 0

    for key, value in tcga.n_per_class.items():
        patches, patches_label = [], []

        # Retrieve 16 random patches for the current class
        for i in range(16):
            _, patch, patch_label, _ = tcga.get_item(generator.randint(c_pos, c_pos + value - 1))
            patches.append(patch)
            patches_label.append(patch_label)

        # Plot patches for the current class
        fig = plt.figure(figsize=(8, 2.5))
        fig.suptitle(key)
        for idx, (label, patch) in enumerate(zip(patches_label, patches)):
            ax = plt.subplot(2, 8, idx + 1)
            ax.set_title(label)
            ax.axis('off')
            ax.imshow(patch)
        plt.show()

        # Shift position to next class
        c_pos += value

# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":
    main()

# %%
