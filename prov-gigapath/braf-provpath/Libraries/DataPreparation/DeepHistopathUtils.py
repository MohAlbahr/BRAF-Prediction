# Source: https://github.com/deroneriksson/python-wsi-preprocessing/blob/master/deephistopath/wsi/filter.py

import datetime
import numpy as np

ADDITIONAL_NP_STATS = False

def np_info(np_arr, name=None, elapsed=None):
    """
    Display information (shape, type, max, min, etc) about a NumPy array.

    Args:
        np_arr: The NumPy array.
        name: The (optional) name of the array.
        elapsed: The (optional) time elapsed to perform a filtering operation.
    """

    if name is None:
        name = "NumPy Array"
    if elapsed is None:
        elapsed = "---"

    if ADDITIONAL_NP_STATS is False:
        print("%-20s | Time: %-14s    Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
    else:
        # np_arr = np.asarray(np_arr)
        max = np_arr.max()
        min = np_arr.min()
        mean = np_arr.mean()
        is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
        print("%-20s | Time: %-14s Min: %6.2f    Max: %6.2f    Mean: %6.2f    Binary: %s    Type: %-7s Shape: %s" % (
            name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))

class Time:
    """
    Class for displaying elapsed time.
    """

    def __init__(self):
        self.start = datetime.datetime.now()

    def elapsed_display(self):
        time_elapsed = self.elapsed()
        print("Time elapsed: " + str(time_elapsed))

    def elapsed(self):
        self.end = datetime.datetime.now()
        time_elapsed = self.end - self.start
        return time_elapsed

def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool",
               display_np_info=False):
    """
    Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
    red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.

    Args:
        rgb: RGB image as a NumPy array.
        red_lower_thresh: Red channel lower threshold value.
        green_upper_thresh: Green channel upper threshold value.
        blue_upper_thresh: Blue channel upper threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] > red_lower_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] < blue_upper_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Red", t.elapsed())
    return result


def filter_red_pen(rgb, output_type="bool", display_np_info=False):
    """
    Create a mask to filter out red pen marks from a slide.

    Args:
        rgb: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing the mask.
    """
    t = Time()
    result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
             filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
             filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
             filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
             filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
             filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
             filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
             filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
             filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Red Pen", t.elapsed())
    return result


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool",
                 display_np_info=False):
    """
    Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
    red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
    Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
    lower threshold value rather than a blue channel upper threshold value.

    Args:
        rgb: RGB image as a NumPy array.
        red_upper_thresh: Red channel upper threshold value.
        green_lower_thresh: Green channel lower threshold value.
        blue_lower_thresh: Blue channel lower threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] > green_lower_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Green", t.elapsed())
    return result


def filter_green_pen(rgb, output_type="bool", display_np_info=False):
    """
    Create a mask to filter out green pen marks from a slide.

    Args:
        rgb: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing the mask.
    """
    t = Time()
    result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
             filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
             filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
             filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
             filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
             filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
             filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
             filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
             filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
             filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
             filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
             filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
             filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
             filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
             filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Green Pen", t.elapsed())
    return result


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool",
                display_np_info=False):
    """
    Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
    red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.

    Args:
        rgb: RGB image as a NumPy array.
        red_upper_thresh: Red channel upper threshold value.
        green_upper_thresh: Green channel upper threshold value.
        blue_lower_thresh: Blue channel lower threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Blue", t.elapsed())
    return result


def filter_blue_pen(rgb, output_type="bool", display_np_info=False):
    """
    Create a mask to filter out blue pen marks from a slide.

    Args:
        rgb: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing the mask.
    """
    t = Time()
    result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
             filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
             filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
             filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
             filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
             filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
             filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
             filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
             filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
             filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
             filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
             filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Blue Pen", t.elapsed())
    return result


def filter_grays(rgb, tolerance=15, output_type="bool", display_np_info=False):
    """
    Create a mask to filter out pixels where the red, green, and blue channel values are similar.

    Args:
        np_img: RGB image as a NumPy array.
        tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
    """
    t = Time()
    (h, w, c) = rgb.shape

    rgb = rgb.astype(int)
    rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)

    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Grays", t.elapsed())
    return result