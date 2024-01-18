from matplotlib import pyplot as plt
import numpy as np
from bbtoolkit.structures.geometry import TexturedPolygon


def plot_polygon(polygon: TexturedPolygon, ax: plt.Axes = None, **kwargs) -> plt.Figure:
    """
    Plots a given polygon on a matplotlib figure.

    Args:
        polygon (TexturedPolygon): The polygon to be plotted.
        ax (plt.Axes, optional): The axes object to draw the plot onto. If None, a new figure and axes object are created. Defaults to None.
        **kwargs: Arbitrary keyword arguments to be passed to the plot function.

    Returns:
        plt.Figure: The figure object with the plotted polygon.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if isinstance(polygon, TexturedPolygon) and polygon.texture.color is not None:
        kwargs.setdefault('color', polygon.texture.color)
    else:
        kwargs.setdefault('color', 'tab:red')

    ax.plot(*polygon.exterior.xy, **kwargs)

    if polygon.interiors:
        for interior in polygon.interiors:
            ax.plot(*interior.xy, **kwargs)

    return fig


def plot_arrow(angle: float, x: float, y: float, ax: plt.Axes = None, **kwargs) -> plt.Figure:
    """
    Plot an arrow at a specified angle with given x and y coordinates.

    Args:
        angle (float): Angle in radians (from 0 to 2*pi).
        x (float): X-coordinate of the arrow.
        y (float): Y-coordinate of the arrow.
        ax (plt.Axes, optional): Optional axis for the plot. If not provided, a new plot is created.
        **kwargs: Keyword arguments passed to the matplotlib.pyplot.arrow function.

    Returns:
        plt.Figure: The figure on which the arrow is plotted.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    kwargs.setdefault('head_width', 1)
    kwargs.setdefault('head_length', 1)
    kwargs.setdefault('fc', 'red')
    kwargs.setdefault('ec', 'red')

    # Calculate the arrow components
    arrow_length = 0.2
    dx = arrow_length * np.cos(angle)
    dy = arrow_length * np.sin(angle)

    # Plot the arrow
    ax.arrow(x, y, dx, dy, **kwargs)

    return fig