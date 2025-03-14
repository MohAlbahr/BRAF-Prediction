#%%

from gigapath.pipeline import tile_one_slide
import huggingface_hub
import os
import pandas as pd
# assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

# local_dir = os.path.join(os.path.expanduser("~"), ".cache/")
# huggingface_hub.hf_hub_download("prov-gigapath/prov-gigapath", filename="sample_data/PROV-000-000001.ndpi", local_dir=local_dir, force_download=True)

root_dir= "/local/work/rp5-histology/TCGA SKCM/"

data= pd.read_csv("/projects/wispermed_rp18/braf-main/prov-gigapath/prov-gigapath/braf-provpath/Libraries/DataPreparation/slide_labels.csv")

for image in data["slide_id"]:

    slide_path = os.path.join(root_dir, image)
    
    save_dir = "/projects/wispermed_rp18/tcga_tils"
    
    print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides. Please make sure to use the appropriate level for the 0.5 MPP")
    tile_one_slide(slide_path, save_dir=save_dir, level=0)
 
print("NOTE: tiling dependency libraries can be tricky to set up. Please double check the generated tile images.")
# %%


import os
import pandas as pd
import openslide
from gigapath.pipeline import tile_one_slide

# Root directory where slide images are stored
root_dir = "/local/work/rp5-histology/TCGA SKCM/"

# Load slide data
data = pd.read_csv("/projects/wispermed_rp18/braf-main/prov-gigapath/prov-gigapath/braf-provpath/Libraries/DataPreparation/slide_labels.csv")

# Parameters
upp_target = 0.5  # Target microns per pixel
patch_size_target = 256  # Target patch size in pixels

for image in data["slide_id"]:
    slide_path = os.path.join(root_dir, image)
    
    # Open the slide
    slide = openslide.OpenSlide(slide_path)
    
    # Retrieve the resolution at level 0
    mpp_x = float(slide.properties.get('openslide.mpp-x', '0'))
    mpp_y = float(slide.properties.get('openslide.mpp-y', '0'))
    
    # Ensure the MPP is valid
    if mpp_x == 0 or mpp_y == 0:
        print(f"Error: Unable to retrieve MPP for slide {slide_path}.")
        continue
    
    slide_resolution = (mpp_x + mpp_y) / 2  # Average MPP
    
    # Calculate the tile size based on the target MPP
    tile_size = int(patch_size_target * (upp_target / slide_resolution))
    
    save_dir = "/projects/wispermed_rp18/tcga_tils"
    
    print(f"Processing slide {slide_path} with tile size {tile_size} (target MPP: {upp_target}, slide MPP: {slide_resolution})")
    
    # Tile the slide
    tile_one_slide(slide_path, save_dir=save_dir, level=0, tile_size=tile_size)
