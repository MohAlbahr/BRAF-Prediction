#!/usr/bin/env python
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#
#  Original: https://github.com/microsoft/hi-ml/blob/main/hi-ml-cpath/src/health_cpath/preprocessing/create_tiles_dataset.py
#  ------------------------------------------------------------------------------------------
#
# Modified by Mohamed Albahri, 2024.
# This code has been modified to tile slides to a specific tile size computed dynamically
# so that tiles match a required mpp (microns per pixel).
#  ------------------------------------------------------------------------------------------

# =============================================================================
# Standard Library Imports
# =============================================================================
import sys
import os
import functools
import logging
import shutil
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

# =============================================================================
# Third-Party Imports
# =============================================================================
import numpy as np
import pandas as pd
import PIL
from matplotlib import collections, patches, pyplot as plt
from monai.data import Dataset
from monai.data.wsi_reader import WSIReader
from openslide import OpenSlide
from tqdm import tqdm
from gigapath.preprocessing.data import tiling
from gigapath.preprocessing.data.foreground_segmentation import LoadROId, segment_foreground

# =============================================================================
# Original Utility Functions
# =============================================================================
def select_tiles(foreground_mask: np.ndarray, occupancy_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Exclude tiles that are mostly background based on estimated occupancy."""
    if occupancy_threshold < 0. or occupancy_threshold > 1.:
        raise ValueError("Tile occupancy threshold must be between 0 and 1")
    occupancy = foreground_mask.mean(axis=(-2, -1), dtype=np.float16)
    return (occupancy > occupancy_threshold).squeeze(), occupancy.squeeze()  # type: ignore

def get_tile_descriptor(tile_location: Sequence[int]) -> str:
    """Format the XY tile coordinates into a tile descriptor."""
    return f"{tile_location[0]:05d}x_{tile_location[1]:05d}y"

def get_tile_id(slide_id: str, tile_location: Sequence[int]) -> str:
    """Format the slide ID and XY tile coordinates into a unique tile ID."""
    return f"{slide_id}.{get_tile_descriptor(tile_location)}"

def save_image(array_chw: np.ndarray, path: Path) -> PIL.Image:
    """Save an image array in (C, H, W) format to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array_hwc = np.moveaxis(array_chw, 0, -1).astype(np.uint8).squeeze()
    pil_image = PIL.Image.fromarray(array_hwc)
    pil_image.convert('RGB').save(path)
    return pil_image

def check_empty_tiles(tiles: np.ndarray, std_th: int = 5, extreme_value_portion_th: float = 0.5) -> np.ndarray:
    """Determine if a tile is empty. Hacky."""
    b, c, h, w = tiles.shape
    flattned_tiles = tiles.reshape(b, c, h * w)
    std_rgb = flattned_tiles[:, :, :].std(axis=2)
    std_rgb_mean = std_rgb.mean(axis=1)
    low_std_mask = std_rgb_mean < std_th
    extreme_value_count = ((flattned_tiles == 0)).sum(axis=2)
    extreme_value_proportion = extreme_value_count / (h * w)
    extreme_value_mask = extreme_value_proportion.max(axis=1) > extreme_value_portion_th
    return low_std_mask | extreme_value_mask

def generate_tiles(slide_image: np.ndarray, tile_size: int, foreground_threshold: float, occupancy_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Split the foreground of an input slide image into tiles."""
    image_tiles, tile_locations = tiling.tile_array_2d(slide_image, tile_size=tile_size, constant_values=255)
    logging.info(f"image_tiles.shape: {image_tiles.shape}, dtype: {image_tiles.dtype}")
    logging.info(f"Tiled {slide_image.shape} to {image_tiles.shape}")
    foreground_mask, _ = segment_foreground(image_tiles, foreground_threshold)
    selected, occupancies = select_tiles(foreground_mask, occupancy_threshold)
    n_discarded = (~selected).sum()
    logging.info(f"Percentage tiles discarded: {n_discarded / len(selected) * 100:.2f}")
    # FIXME: this uses too much memory
    # empty_tile_bool_mask = check_empty_tiles(image_tiles)
    # selected = selected & (~empty_tile_bool_mask)
    # n_discarded = (~selected).sum()
    # logging.info(f"Percentage tiles discarded after filtering empty tiles: {n_discarded / len(selected) * 100:.2f}")
    # logging.info(f"Before filtering: min y: {tile_locations[:, 0].min()}, max y: {tile_locations[:, 0].max()}, min x: {tile_locations[:, 1].min()}, max x: {tile_locations[:, 1].max()}")
    image_tiles = image_tiles[selected]
    tile_locations = tile_locations[selected]
    occupancies = occupancies[selected]
    if len(tile_locations) == 0:
        logging.warn("No tiles selected")
    else:
        logging.info(f"After filtering: min y: {tile_locations[:, 0].min()}, max y: {tile_locations[:, 0].max()}, min x: {tile_locations[:, 1].min()}, max x: {tile_locations[:, 1].max()}")
    return image_tiles, tile_locations, occupancies, n_discarded

def get_tile_info(sample: Dict["SlideKey", Any], occupancy: float, tile_location: Sequence[int], rel_slide_dir: Path) -> Dict["TileKey", Any]:
    """Map slide information and tiling outputs into a tile-specific information dictionary."""
    slide_id = sample["slide_id"]
    descriptor = get_tile_descriptor(tile_location)
    rel_image_path = f"{rel_slide_dir}/{descriptor}.png"
    tile_info = {
        "slide_id": slide_id,
        "tile_id": get_tile_id(slide_id, tile_location),
        "image": rel_image_path,
        "label": sample.get("label", None),
        "tile_x": tile_location[0],
        "tile_y": tile_location[1],
        "occupancy": occupancy,
        "metadata": {"slide_" + key: value for key, value in sample["metadata"].items()}
    }
    return tile_info

def format_csv_row(tile_info: Dict["TileKey", Any], keys_to_save: Iterable["TileKey"], metadata_keys: Iterable[str]) -> str:
    """Format a tile information dictionary as a CSV row."""
    tile_slide_metadata = tile_info.pop("metadata")
    fields = [str(tile_info[key]) for key in keys_to_save]
    fields.extend(str(tile_slide_metadata[key]) for key in metadata_keys)
    dataset_row = ','.join(fields)
    return dataset_row

def load_image_dict(sample: dict, level: int, margin: int, foreground_threshold: Optional[float] = None) -> Dict["SlideKey", Any]:
    """Load an image from a metadata dictionary."""
    loader = LoadROId(WSIReader(backend="OpenSlide"), level=level, margin=margin, foreground_threshold=foreground_threshold)
    img = loader(sample)
    return img

def save_thumbnail(slide_path, output_path, size_target=1024):
    """Save a thumbnail of the slide image."""
    with OpenSlide(str(slide_path)) as openslide_obj:
        scale = size_target / max(openslide_obj.dimensions)
        thumbnail = openslide_obj.get_thumbnail([int(m * scale) for m in openslide_obj.dimensions])
        thumbnail.save(output_path)
        logging.info(f"Saving thumbnail {output_path}, shape {thumbnail.size}")

def visualize_tile_locations(slide_sample, output_path, tile_info_list, tile_size, origin_offset):
    """Overlay tile locations on a slide thumbnail and save the visualization."""
    slide_image = slide_sample["image"]
    downscale_factor = slide_sample["scale"]
    fig, ax = plt.subplots()
    ax.imshow(slide_image.transpose(1, 2, 0))
    rects = []
    for tile_info in tile_info_list:
        xy = ((tile_info["tile_x"] - origin_offset[0]) / downscale_factor,
              (tile_info["tile_y"] - origin_offset[1]) / downscale_factor)
        rects.append(patches.Rectangle(xy, tile_size, tile_size))
    pc = collections.PatchCollection(rects, match_original=True, alpha=0.5, edgecolor="black")
    pc.set_array(np.array([100] * len(tile_info_list)))
    ax.add_collection(pc)
    fig.savefig(output_path)
    plt.close()

def is_already_processed(output_tiles_dir):
    """Check if a slide's tiles have already been processed."""
    if not output_tiles_dir.exists():
        return False
    if len(list(output_tiles_dir.glob("*.png"))) == 0:
        return False
    dataset_csv_path = output_tiles_dir / "dataset.csv"
    try:
        df = pd.read_csv(dataset_csv_path)
    except:
        return False
    return len(df) > 0

# =============================================================================
# Modified Tiling Code (My Modified Code)
# =============================================================================

def compute_tile_size(slide_path, patch_size_target, upp_target):
    """
    Compute tile size dynamically based on slide MPP or resolution.
    :param slide_path: Path to the slide image.
    :param patch_size_target: Target patch size.
    :param upp_target: Desired microns per pixel (MPP).
    :return: Computed tile size.
    """
    slide = openslide.OpenSlide(slide_path)
    mpp_x = slide.properties.get('openslide.mpp-x')
    mpp_y = slide.properties.get('openslide.mpp-y')
    if mpp_x and mpp_y:
        mpp_x = float(mpp_x)
        mpp_y = float(mpp_y)
    else:
        x_resolution = slide.properties.get('tiff.XResolution')
        y_resolution = slide.properties.get('tiff.YResolution')
        resolution_unit = slide.properties.get('tiff.ResolutionUnit')
        if x_resolution and y_resolution:
            x_resolution = float(x_resolution)
            y_resolution = float(y_resolution)
            if resolution_unit == 'centimeter':
                mpp_x = 10000 / x_resolution
                mpp_y = 10000 / y_resolution
            elif resolution_unit == 'inch':
                mpp_x = x_resolution
                mpp_y = y_resolution
            else:
                raise ValueError(f"Unknown resolution unit: {resolution_unit} for slide {slide_path}")
        else:
            raise ValueError(f"Slide {slide_path} does not have MPP or resolution information.")
    print(mpp_x)
    slide_resolution = (mpp_x + mpp_y) / 2
    tile_size = int(patch_size_target * (upp_target / slide_resolution))
    return tile_size

def process_and_check_slide(sample, level, margin, foreground_threshold, occupancy_threshold, output_dir, thumbnail_dir, tile_progress, patch_size_target, upp_target):
    """
    Process a slide: compute tile size, tile the slide, and check the resulting CSV files.
    :param sample: Slide sample dictionary.
    :param level: Magnification level.
    :param margin: Margin around foreground.
    :param foreground_threshold: Threshold for foreground segmentation.
    :param occupancy_threshold: Threshold for tile occupancy.
    :param output_dir: Directory to save tile images and CSV.
    :param thumbnail_dir: Directory for thumbnails.
    :param tile_progress: Whether to display progress.
    :param patch_size_target: Target patch size.
    :param upp_target: Desired MPP.
    :return: Output tiles directory.
    """
    slide_path = sample["image"]
    tile_size = compute_tile_size(slide_path, patch_size_target, upp_target)
    print(f"Processing {slide_path} with tile size {tile_size}")
    slide_dir = process_slide(
        sample=sample,
        level=level,
        margin=margin,
        tile_size=tile_size,
        foreground_threshold=foreground_threshold,
        occupancy_threshold=occupancy_threshold,
        output_dir=output_dir,
        thumbnail_dir=thumbnail_dir,
        tile_progress=tile_progress
    )
    dataset_csv_path = slide_dir / "dataset.csv"
    dataset_df = pd.read_csv(dataset_csv_path)
    assert len(dataset_df) > 0
    failed_csv_path = slide_dir / "failed_tiles.csv"
    failed_df = pd.read_csv(failed_csv_path)
    assert len(failed_df) == 0
    print(f"Slide {slide_path} has been tiled. {len(dataset_df)} tiles saved to {slide_dir}.")

def merge_dataset_csv_files(dataset_dir: Path) -> Path:
    """Combine all '*/dataset.csv' files into a single CSV file."""
    full_csv = dataset_dir / "dataset.csv"
    with full_csv.open('w') as full_csv_file:
        first_file = True
        for slide_csv in tqdm(dataset_dir.glob("*/dataset.csv"), desc="Merging dataset.csv", unit='file'):
            logging.info(f"Merging slide {slide_csv}")
            content = slide_csv.read_text()
            if not first_file:
                content = content[content.index('\n') + 1:]
            full_csv_file.write(content)
            first_file = False
    return full_csv

def main():
    # Main entry point for processing slides to create a tiles dataset.
    # All commented-out code is preserved.
    
    # Load slides dataset from CSV and set directories. (chnage as needed)
    slides_csv_path = "prov-gigapath/dataset_csv/braf/slide_labels.csv"
    root_dir = "datasets/TCGA-SKCM"
    root_output_dir = "prov-gigapath/data/tcga_data"
    image_type = ".svs"
    tcga_data_dir = "prov-gigapath/data/tcga_data"
    slides_csv_path = "Libraries/DataPreparation/slide_labels.csv"
   
    patch_size_target = 256
    upp_target = 0.5 
    level = 0
    margin = 0  
    foreground_threshold = None   
    occupancy_threshold = 0.1
    parallel = True 
    overwrite = False      
    n_slides = None
   
    directories = os.listdir(tcga_data_dir)
    slides_df = pd.read_csv(slides_csv_path)
    process_all = False
   
    if process_all: 
        slides_df = pd.read_csv(slides_csv_path)
        slides_dataset = []
        for index, row in slides_df.iterrows():
            slide_path = os.path.join(root_dir, row['slide_id'])
            slides_dataset.append({"image": slide_path + image_type, "slide_id": str(row['slide_id']).split('/')[0], "label": row['label'], "metadata": {}})
    else:
        print("Unique directories: ", len(np.unique(directories)))
        Images_to_process = []
        for direct in directories:
            num_files = len(os.listdir(os.path.join(tcga_data_dir, direct)))
            if num_files < 10:
                Images_to_process.append(direct)
        print("Number of to-process data: ", len(Images_to_process))
        slides_dataset = []
        for index, row in slides_df.iterrows():
            slide_path = os.path.join(root_dir, row['slide_id'])
            for slide_id in Images_to_process:
                if slide_id in slide_path:
                    slides_dataset.append({
                        "image": slide_path + image_type,
                        "slide_id": str(slide_path).split('/')[5],
                        "label": row['label'],
                        "metadata": {}
                    })
        print(slides_dataset)
        print("Total number: ", len(slides_dataset))

    #############################################
    # Custom dataset to tile only some slides,
    # not the whole at once
    #############################################
    
    def tile_one_slide(slide_file: str = '', save_dir: str = '', level: int = 0, tile_size: int = 256):
        """
        This function is used to tile a single slide and save the tiles to a directory.
        """
        slide_id = os.path.basename(slide_file)
        save_dir = Path(save_dir)
        print(f"Processing slide {slide_file} at level {level} with tile size {tile_size}. Saving to {save_dir}.")
        slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": {}}
        slide_dir = process_slide(
            sample=slide_sample,
            level=level,
            margin=0,
            tile_size=tile_size,
            foreground_threshold=None,
            occupancy_threshold=0.1,
            output_dir=save_dir,
            thumbnail_dir=save_dir / "thumbnails",
            tile_progress=True,
        )
        dataset_csv_path = slide_dir / "dataset.csv"
        dataset_df = pd.read_csv(dataset_csv_path)
        assert len(dataset_df) > 0
        failed_csv_path = slide_dir / "failed_tiles.csv"
        failed_df = pd.read_csv(failed_csv_path)
        assert len(failed_df) == 0
        print(f"Slide {slide_file} has been tiled. {len(dataset_df)} tiles saved to {slide_dir}.")

    for slide_info in slides_dataset:
        slide_file = slide_info['image']
        slide_id = slide_info['slide_id']
        label = slide_info['label']
        save_dir = os.path.join(root_output_dir, slide_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tile_size = compute_tile_size(slide_file, patch_size_target, upp_target)
        tile_one_slide(slide_file, save_dir, level, tile_size)

    print("All slides have been processed.")

if __name__ == "__main__":
    main()
