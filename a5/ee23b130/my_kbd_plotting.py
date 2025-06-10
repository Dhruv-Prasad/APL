# Imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.colors as mcolors

import numpy as np
from collections import defaultdict
from numpy.typing import ArrayLike
from typing import Tuple, Dict, Any


def get_freqs(string: str, presses_reqd, dictionary: Dict) -> Tuple[float, int]:
    """Function to: Populate an dictionary with frequency of key presses \n
    Returns max frequency
    """
    max_freq = 0
    for character in string:
        if character in presses_reqd:
            for key_pressed in presses_reqd[character]:
                dictionary[key_pressed] += 1
                max_freq = max(max_freq, dictionary[key_pressed])
    return max_freq


def get_widths(
    layout: Dict[str, dict[str, Any]], widths: Dict[str, float]
) -> Tuple[float, float, float, float]:
    """Function to get width of keys based on usual key size and space available
    Returns range of x and y values of plot
    """

    # Default widths of keys provided there is space
    shifts_width = 2
    space_width = 3.5
    normal_width = 1

    # I need these so I can set xlim and ylim while plotting
    overall_x_max = 0
    overall_x_min = 100
    overall_y_max = 0
    overall_y_min = 100

    # Map each key to it's row (y coordinate)
    # Get the range of positions specified
    rows = defaultdict(list)
    for key, data in layout.items():
        rows[data["pos"][1]].append((key, data["pos"]))
        overall_x_max = max(data["pos"][0], overall_x_max)
        overall_x_min = min(data["pos"][0], overall_x_min)
        overall_y_max = max(data["pos"][1], overall_y_max)
        overall_y_min = min(data["pos"][1], overall_y_min)

    # Increase the max x and y to account for key size
    overall_x_max += 1
    overall_y_max += 1

    # Sort the rows by x coordinate
    for row in rows.values():
        row.sort(key=lambda x: x[1][0])

    # Find width of each key
    for row in rows.values():
        for i in range(len(row) - 1):
            # I try to make each key of a certain size unless the next key in row is too close
            # Some special keys are larger than normal ones
            key = row[i][0]
            if key == "Shift_L" or key == "Shift_R" or key == "Enter":
                widths[key] = min(row[i + 1][1][0] - row[i][1][0], shifts_width)
            elif key == "Space":
                widths[key] = min(row[i + 1][1][0] - row[i][1][0], space_width)
            else:
                widths[key] = min(row[i + 1][1][0] - row[i][1][0], normal_width)
        # Handle last key in row
        # Increase the max x value if required due to larger size of some special keys
        key = row[len(row) - 1][0]
        if key == "Shift_L" or key == "Shift_R" or key == "Enter":
            widths[key] = shifts_width
            overall_x_max = max(row[len(row) - 1][1][0] + shifts_width, overall_x_max)
        elif key == "Space":
            widths[key] = space_width
            overall_x_max = max(row[len(row) - 1][1][0] + shifts_width, overall_x_max)
        else:
            widths[key] = normal_width
    return (overall_x_min, overall_x_max, overall_y_min, overall_y_max)


def plot_kbd(
    ax: plt.Axes,
    kbd: Dict[str, dict[str, Any]],
    widths: Dict[str, float],
    plot_lims: Tuple[float, float, float, float],
):
    """Function to plot keyboard given a layout and key widths"""
    (xmin, xmax, ymin, ymax) = plot_lims
    ax.set_xlim(xmin - 0.2, xmax + 0.2)
    ax.set_ylim(ymin - 2, ymax + 2)
    for key in kbd.keys():
        width = widths[key]
        height = 1  # Assumed as 1
        ax.add_patch(
            mpatch.FancyBboxPatch(
                kbd[key]["pos"],
                width - 0.2,
                height - 0.2,
                boxstyle="round,pad=0.1",
                fc="none",
                ec=(0, 0, 0),
            )
        )
        # Add the key name centered inside the box
        x_center = (kbd[key]["pos"][0]) + (width / 2)
        y_center = (kbd[key]["pos"][1]) + (height / 2)
        ax.text(
            x_center - 0.1,
            y_center,
            key,
            ha="center",
            va="top",
            fontsize=6,
            bbox=dict(facecolor="none", alpha=0.5, edgecolor="none"),
        )


# Function defs
def calculate_distance(pos1: Tuple, pos2: Tuple) -> float:
    """Function to find distance between two points"""
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** (0.5)


def gaussian_2d(
    x: float, y: float, x0: float, y0: float, amplitude: float, sigma: float
) -> float:
    """Function to calculate value at x,y of gaussian distribution at x0,y0"""
    return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def make_empty_grid(
    xmin: float, xmax: float, ymin: float, ymax: float, division_size: float
) -> Tuple[int, int, ArrayLike]:
    """Function to make an empty grid with the dimensions of the keyboard
    Returns dimensions of grid and the grid
    """
    num_of_x_divisions = int((xmax - xmin + 0.4) / division_size)
    num_of_y_divisions = int((ymax - ymin + 0.4) / division_size)
    grid_freq = np.zeros((num_of_x_divisions, num_of_y_divisions), float)
    return num_of_x_divisions, num_of_y_divisions, grid_freq


def nearest_idx(x: float, xmin: float, division_size: float) -> int:
    """Calculate the nearest point in a grid based on division size."""
    x_idx = (x - xmin) / division_size
    return int(x_idx)


# Function to generate continuous heatmap with Gaussian smoothing
def create_continuous_heatmap(
    kbd: Dict[str, dict[str, Any]],
    key_press_frequency: Dict[str, int],
    widths: Dict[str, float],
    max_freq: int,
    plot_lims: Tuple[float, float, float, float],
):
    """Function to make a heatmap of key press frequencies using gaussian to blend colors
    Returns 2D ndarray of smoothened frequencies
    """
    (xmin, xmax, ymin, ymax) = plot_lims

    # Get separate lists of key centre positions and frequencies
    key_positions_list = []
    key_freqs_list = []

    for key, _ in kbd.items():
        width = widths[key]
        # I assume key height to always be 1 (i.e. y coords are always integral)
        height = 1
        x_center = (kbd[key]["pos"][0]) + (width / 2)
        y_center = (kbd[key]["pos"][1]) + (height / 2)
        key_positions_list.append((x_center, y_center))
        key_freqs_list.append(
            key_press_frequency[key] / max_freq if key_press_frequency[key] > 0 else 0
        )

    # Convert key positions and frequencies from list to ndarray
    key_positions_ndarray = np.array(key_positions_list)
    key_freqs_ndarray = np.array(key_freqs_list)

    # Make grid of constant division size
    division_size = 0.05
    num_xs, num_ys, grid_freq = make_empty_grid(xmin, xmax, ymin, ymax, division_size)

    # Specify constant parameters for gaussian
    sigma = 7

    # Specify radius in transformed (grid) coordinates of cirle to apply gaussian in
    radius = 100

    # Fill the grid with key frequencies at the appropriate positions using Gaussian smoothing
    for i, (x_center, y_center) in enumerate(key_positions_ndarray):
        # Find the closest grid point to the key position
        x_center_i = nearest_idx(x_center, xmin=xmin, division_size=division_size)
        y_center_i = nearest_idx(y_center, xmin=ymin, division_size=division_size)
        # Iterate through square (if possible) around key press position of side 2*radius
        for xi in range(max(0, x_center_i - radius), min(num_xs, x_center_i + radius)):
            for yi in range(
                max(0, y_center_i - radius), min(num_ys, y_center_i + radius)
            ):
                # Check if point is in circle
                if calculate_distance((xi, yi), (x_center_i, y_center_i)) < radius:
                    gauss_freq = gaussian_2d(
                        xi, yi, x_center_i, y_center_i, key_freqs_ndarray[i], sigma
                    )
                    grid_freq[xi, yi] += gauss_freq  # Add Gaussian value to the grid
    return grid_freq


def draw_map(
    ax: plt.Axes,
    kbd: Dict[str, dict[str, Any]],
    widths: Dict[str, float],
    plot_lims: Tuple[float, float, float, float],
    freqs: Dict[str, int],
    max_freq,
):
    """Function to draw heatmap, to be called by main file"""
    (xmin, xmax, ymin, ymax) = plot_lims
    # Make a custom colormap object from blue to orange from default cmap 'Spectral'
    Spectral = plt.get_cmap("Spectral", 1000)
    # Reverse the colormap and change start and end color
    newcmap = mcolors.ListedColormap(Spectral(np.linspace(0.9, 0.1, 800)))

    # Create continuous heatmap grid with Gaussian smoothing
    grid_freq = create_continuous_heatmap(
        kbd=kbd,
        key_press_frequency=freqs,
        widths=widths,
        max_freq=max_freq,
        plot_lims=plot_lims,
    )

    # Plot the continuous heatmap
    # Use bicubic interpolation for further smoothing
    ax.imshow(
        grid_freq.T,
        extent=(xmin - 0.2, xmax + 0.2, ymin - 0.2, ymax + 0.2),
        origin="lower",
        cmap=newcmap,
        aspect="auto",
        interpolation="bicubic",
    )
