#!/usr/bin/env python
"""

This script processes the dataset using the Prov-GigaPath foundation model,
extracts tiles embeddings from histopathological images, and saves the embeddings
to HDF5 files (for each slide). 

Author: Mohamed Albahri
Year: 2024

"""

import os
import sys
import argparse

# Fix import path for interactive sessions
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')))

import numpy as np
import h5py
import torch
import wandb
import timm
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from Libraries.DataPreparation.PyTDataloader import prepare_data_loader
from Libraries.utils import ModifiedLogger
from RunConfig import RunConfig
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def group_and_process_tiles(data_loader, tile_encoder, device):
    """
    Group image tiles by slide and extract features using the tile encoder.

    Args:
        data_loader: DataLoader yielding tuples (slide_id, images, labels, coordinates).
        tile_encoder: Pretrained tile encoder model.
        device: Torch device for computation.

    Returns:
        slide_representations: Dictionary mapping slide_id to a dict with 'features' and 'coords'.
        slide_labels: Dictionary mapping slide_id to the corresponding label.
    """
    grouped_tiles = {}
    slide_representations = {}
    slide_labels = {}

    tile_encoder.eval()  # Set model to evaluation mode
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for slide_ids, images, labels, coordinates in tqdm(data_loader, total=len(data_loader), desc="Processing Tiles"):
            images = images.to(device)  # Move batch to device
            tile_features = tile_encoder(images).detach().cpu()  # Extract features

            for i in range(len(slide_ids)):
                sid = slide_ids[i]
                if sid not in grouped_tiles:
                    grouped_tiles[sid] = {"features": [], "coords": []}
                grouped_tiles[sid]["features"].append(tile_features[i])
                grouped_tiles[sid]["coords"].append(coordinates[i])
                # Assume all tiles from a slide share the same label
                slide_labels[sid] = labels[i].item()

    # Convert lists to numpy arrays for each slide
    for sid in grouped_tiles:
        features = np.stack(grouped_tiles[sid]["features"])
        coords = np.stack(grouped_tiles[sid]["coords"])
        slide_representations[sid] = {'features': features, 'coords': coords}

    return slide_representations, slide_labels

def save_slide_embeddings(data_loader, tile_encoder, device, save_dir):
    """
    Process the dataset and save slide embeddings to HDF5 files.

    Args:
        data_loader: DataLoader for the dataset.
        tile_encoder: Pretrained tile encoder model.
        device: Torch device.
        save_dir: Directory where HDF5 files will be saved.
    """
    slide_reps, slide_labels = group_and_process_tiles(data_loader, tile_encoder, device)

    for slide_id, data in slide_reps.items():
        # Generate filename based on slide_id
        h5_path = os.path.join(save_dir, f'{str(slide_id).split("/")[0]}.h5')
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('features', data=data['features'])
            hf.create_dataset('labels', data=np.array([slide_labels[slide_id]]))
            hf.create_dataset('coords', data=data['coords'])
    print(f'Embeddings saved to {save_dir}')

def main(args):
    """
    Main execution pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Initialize custom logger from wandb configuration
    logger = ModifiedLogger(wandb.config['run_data_dir'])
    cudnn.benchmark = True

    # Prepare data loader using parameters from wandb config
    data_loader = prepare_data_loader(
        all_dataset=wandb.config['train_val_dataset'],
        train_val_dataset=wandb.config['train_val_dataset'],
        test_dataset=wandb.config['test_dataset'],
        batch_size=wandb.config['batch_size'],
        num_workers=wandb.config['workers'],
        prov_transforms=wandb.config['prov_transforms'],
        image_size=wandb.config['image_size'],
        train_hflip=wandb.config['hflip'],
        train_vflip=wandb.config['vflip'],
        train_color_jitter=wandb.config['color-jitter'],
        train_subset_fraction=wandb.config['train_subset_fraction'],
        val_subset_fraction=wandb.config['val_subset_fraction'],
        test_subset_fraction=wandb.config['test_subset_fraction'],
        subset_stratify=wandb.config['subset_stratify'],
        run_number=wandb.config['run_number']
    )

    # Load tile encoder model (Prov-GigaPath) via timm
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    tile_encoder.to(device)
    tile_encoder.eval()

    # Use the provided save_dir argument; if not specified, use default from config
    save_dir = args.save_dir if args.save_dir else os.path.join(wandb.config['run_data_dir'], "slide_embeddings")

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Process the dataset and save slide embeddings
    save_slide_embeddings(data_loader, tile_encoder, device, save_dir)
    
    ## Use this if you need the splitted source Dataset
    # save_slide_embeddings(train_loader,  tile_encoder, device, save_dir)
    # save_slide_embeddings(val_loader,    tile_encoder, device, save_dir)
    # save_slide_embeddings(test_loader,   tile_encoder, device, save_dir)

# =============================================================================
# Command-Line Interface
# =============================================================================
if __name__ == '__main__':
    # First, parse additional arguments (e.g., save_dir) that are not handled by RunConfig.
    parser = argparse.ArgumentParser(description="script for tiles embeddings generation.", add_help=True)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save slide embeddings (HDF5 files).")
    # Parse known args; RunConfig will parse the remaining arguments.
    args, remaining = parser.parse_known_args()

    # Update sys.argv to contain only the remaining arguments so that RunConfig doesn't see unrecognized args.
    sys.argv = [sys.argv[0]] + remaining
    # Initialize RunConfig with default_config_name 'prov_config' (only implemented configuration)
    
    config = RunConfig(root_dir=os.path.dirname(os.path.realpath(__file__)), default_config_name='prov_config')

    # Initialize wandb using configuration from RunConfig
    wandb.init(
        project="braf",
        entity="fh-dortmund",
        dir=config.get_root_dir(),
        name=config.get_name(),
        config=config.get_config(),
    )

    # Run the main pipeline with additional arguments
    main(args)

    # Finish the wandb run (necessary for proper logging termination)
    wandb.finish()
