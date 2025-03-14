
#%%

import openslide
import numpy as np
def find_level_for_target_mpp(slide_path, target_mpp):
    """
    Find the level in the slide that corresponds to the target MPP.

    Parameters:
    slide_path (str): Path to the slide file.
    target_mpp (float): Target microns per pixel (MPP).

    Returns:
    int: Level number that corresponds to the target MPP or None if not found.
    """
    slide = openslide.OpenSlide(slide_path)

    print(slide.properties)

    # Retrieve resolution information from properties
    x_resolution = float(slide.properties.get('tiff.XResolution'))
    y_resolution = float(slide.properties.get('tiff.YResolution'))
    # mpp_x = slide.properties.get('openslide.mpp-x')
    # mpp_y = slide.properties.get('openslide.mpp-y')
    # print(type(mpp_x), " ...", mpp_y)
    resolution_unit = slide.properties.get('tiff.ResolutionUnit')

    ### Convert resolution to microns per pixel (MPP)
    if resolution_unit == 'centimeter':
        mpp_x = 10000 / x_resolution
        mpp_y = 10000 / y_resolution
        print(type(mpp_x), " ...", mpp_y)

    else:
        print("Resolution unit is not in centimeters. Adjust the calculation accordingly.")
        return None

    # Check if MPP information is available
    if not mpp_x or not mpp_y:
        print("Could not calculate MPP due to missing or invalid resolution information.")
        return None

    # Iterate through each level and calculate MPP
    for level in range(slide.level_count):
        # Calculate MPP for the current level
        level_mpp_x = np.float64(mpp_x) * slide.level_downsamples[level]
        level_mpp_y = np.float64(mpp_y) * slide.level_downsamples[level]

        # Check if this level's MPP is close to the target MPP
        if abs(level_mpp_x - target_mpp) < 0.1 and abs(level_mpp_y - target_mpp) < 0.1:
            print(f"Level {level} corresponds to approximately {target_mpp} MPP.")
            return level

    print(f"No level corresponds to approximately {target_mpp} MPP.")
    return None



find_level_for_target_mpp(slide_path= "/local/work/rp5-histology/UKE/0f20ade4-aecf-4361-abe5-f7a549fcac9e/0f20ade4-aecf-4361-abe5-f7a549fcac9e_Wholeslide_EnhancedColors_Extended.tif", target_mpp=0.5)
# %%


## Alternative implementation

# def find_level_for_target_mpp(slide_path, target_mpp):
#     """
#     Find the level in the slide that corresponds to the target MPP.

#     Parameters:
#     slide_path (str): Path to the slide file.
#     target_mpp (float): Target microns per pixel (MPP).

#     Returns:
#     int: Level number that corresponds to the target MPP or None if not found.
#     """
#     slide = openslide.OpenSlide(slide_path)

#     print("Slide Properties:")
#     for key, value in slide.properties.items():
#         print(f"{key}: {value}")

#     # Retrieve base MPP from properties
#     mpp_x = float(slide.properties.get('openslide.mpp-x', 0))
#     mpp_y = float(slide.properties.get('openslide.mpp-y', 0))

#     if mpp_x == 0 or mpp_y == 0:
#         print("Base MPP not found in slide properties.")
#         return None

#     base_mpp = (mpp_x + mpp_y) / 2

#     # Calculate the target downsampling factor
#     target_downsample = target_mpp / base_mpp
#     print("target_downsample", target_downsample)
#     # Find the closest level to the target MPP
#     closest_level = None
#     min_diff = float('inf')
    
#     for level in range(slide.level_count):
#         level_downsample = slide.level_downsamples[level]
#         level_mpp = base_mpp * level_downsample

#         diff = abs(level_mpp - target_mpp)
#         if diff < min_diff:
#             min_diff = diff
#             closest_level = level

#     if closest_level is not None:
#         print(f"Level {closest_level} corresponds to approximately {target_mpp} MPP.")
#     else:
#         print(f"No level corresponds to approximately {target_mpp} MPP.")
    
#     return closest_level

# slide_path = "/local/work/rp5-histology/TCGA SKCM/fc206b0f-811d-476a-9896-757208b217a8/TCGA-3N-A9WC-01Z-00-DX1.C833FCAB-6329-4F90-88E5-CFDA0948047B.svs"
# target_mpp = 0.5

# find_level_for_target_mpp(slide_path, target_mpp)

# %%


## compute the tile size for a given mpp in one slide

import openslide
import numpy as np

def compute_tile_size(slide_path, patch_size_target, upp_target):
    slide = openslide.OpenSlide(slide_path)
    
    # Check for MPP information
    mpp_x = slide.properties.get('openslide.mpp-x')
    mpp_y = slide.properties.get('openslide.mpp-y')
    
    if mpp_x and mpp_y:
        mpp_x = float(mpp_x)
        mpp_y = float(mpp_y)
    else:
        # Check for resolution information
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
                mpp_x =  x_resolution
                mpp_y = y_resolution
            else:
                raise ValueError(f"Unknown resolution unit: {resolution_unit} for slide {slide_path}")
        else:
            raise ValueError(f"Slide {slide_path} does not have MPP or resolution information.")
    print(mpp_x)
    slide_resolution = (mpp_x + mpp_y) / 2  # Average resolution if x and y resolutions are different
    tile_size = int(patch_size_target * (upp_target / slide_resolution))
    return tile_size

slide_path = "/local/work/rp5-histology/TCGA SKCM/0a6ef6d7-4e0b-4b23-90a7-da39e0eaf5af/TCGA-D3-A2JL-06Z-00-DX1.3258F79C-866E-4AC5-BB16-F4DF65E9DFC2.svs"
target_mpp=0.5

compute_tile_size(slide_path, 256, target_mpp)

# %%
