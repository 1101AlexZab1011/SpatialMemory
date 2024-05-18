from matplotlib import pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np


def adjust_color_brightness(color: str | tuple) -> str:
    """
    Adjusts the brightness of a given color.

    This function takes a color specified either as a name (e.g., 'darkblue') or as a hexadecimal value (e.g., '#FFDD44')
    and returns a modified version of it. If the original color is dark, a brighter version of the color is returned.
    If the original color is bright, a darker version is returned.

    Parameters:
    color (Union[str, tuple]): The color to adjust. Can be a string with the color name or hex value, or a tuple representing RGBA.

    Returns:
    str: The adjusted color in hexadecimal format.

    Example:
    >>> adjust_color_brightness('darkblue')
    '#0000ee'
    >>> adjust_color_brightness('#FFDD44')
    '#bba833'
    """
    # Convert the color to RGBA format, which returns (r, g, b, a) with values in [0, 1]
    rgba = mcolors.to_rgba(color)

    # Calculate the brightness of the color using a common formula
    brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]

    # Determine if the color is bright or dark
    is_dark = brightness < 0.5

    # Adjust the color brightness
    if is_dark:
        # Make the color brighter
        adjusted_color = mcolors.to_hex([min(1, c * 1.5) for c in rgba[:3]] + [rgba[3]])
    else:
        # Make the color darker
        adjusted_color = mcolors.to_hex([c * 0.7 for c in rgba[:3]] + [rgba[3]])

    return adjusted_color


def generate_cmap(*colors: str) -> mcolors.ListedColormap:
    """Generate a ListedColormap using the provided colors.

    This function generates a `ListedColormap` with a color gradient
    between each pair of adjacent colors in the `colors` list.

    Args:
    *colors : str
        List of hexadecimal color strings.

    Returns:
    matplotlib.colors.ListedColormap
        ListedColormap with a gradient between the provided colors.

    Examples:
    >>> generate_cmap("#FF0000", "#00FF00", "#0000FF")
    ListedColormap(['#ff0000', '#00ff00', '#0000ff'], N=256)

    """
    def crange(a: float, b: float, N: int) -> np.ndarray:
        """Return a range of numbers from a to b, with N values."""
        if a < b:
            return np.arange(a, b, (abs(a - b))/N)
        elif a > b:
            return np.arange(b, a, (abs(a - b))/N)[::-1]
        else:
            return a*np.ones(N)

    N = 256
    all_vals = list()
    for from_color, to_color in zip(colors[:-1], colors[1:]):
        vals = np.ones((N, 4))
        a1, a2, a3 = mcolors.hex2color(from_color)
        b1, b2, b3 = mcolors.hex2color(to_color)
        vals[:, 0] = crange(a1, b1, N)
        vals[:, 1] = crange(a2, b2, N)
        vals[:, 2] = crange(a3, b3, N)
        all_vals.append(vals)

    return mcolors.ListedColormap(np.vstack(all_vals))


def get_most_visible_color(cmap):
    """
    Returns the hex code of the most visible color on the given colormap
    when viewed on a grayscale background.

    Args:
        cmap (str or Colormap): The colormap to analyze.

    Returns:
        str: The hex code of the most visible color.
    """
    # Ensure cmap is a Colormap instance
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Generate an array of points along the colormap
    colors = cmap(np.linspace(0, 1, 256))

    # Convert colors to grayscale using the luminosity method
    # This method better represents human perception
    grayscale = 0.21 * colors[:, 0] + 0.72 * colors[:, 1] + 0.07 * colors[:, 2]

    # Find the color with the maximum contrast to the midpoint of the grayscale spectrum
    # The midpoint is 0.5 in normalized grayscale (0 is black, 1 is white)
    contrast = np.abs(grayscale - 0.5)
    max_contrast_index = np.argmax(contrast)

    # Convert the most visible color to hex format
    most_visible_color = colors[max_contrast_index]
    hex_color = matplotlib.colors.rgb2hex(most_visible_color[:3])

    return hex_color
