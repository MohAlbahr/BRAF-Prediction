import numpy as np
from tiffslide import TiffSlide
import torch
from torch.utils.data import Dataset

class SlideDataset(Dataset):
    def __init__(self, slide_file, slide_grid, slide_level, patch_size, worklist):
        self.slide_file = slide_file
        self.slide_grid = slide_grid
        self.slide_level = slide_level
        self.patch_size = patch_size
        self.worklist = worklist

    def __len__(self):
        return len(self.worklist)

    def __getitem__(self, idx):
        # idx to list if necessary
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get patch indices from worklist
        patch_idx = self.worklist[idx]

        # get patch position
        idx_y, idx_x = np.unravel_index(patch_idx, self.slide_grid, order='F')
        
        # read patch for classification
        with TiffSlide(self.slide_file) as slide:
            patch = slide.read_region((idx_x * self.patch_size, idx_y * self.patch_size), self.slide_level, (self.patch_size, self.patch_size), as_array=True)

        return patch, patch_idx