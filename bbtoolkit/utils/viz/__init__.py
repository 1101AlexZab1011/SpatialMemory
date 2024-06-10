import logging
import os
import re
import cv2
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import numpy as np
from bbtoolkit.structures.geometry import TexturedPolygon


def show_figure(fig: matplotlib.figure.Figure) -> None:
    """
    Displays a given matplotlib figure using the canvas manager of a newly created dummy figure.

    This function creates a dummy matplotlib figure and utilizes its canvas manager to display the input figure (`fig`).
    It effectively transfers the canvas of the dummy figure to the input figure, allowing the input figure to be displayed
    using the matplotlib backend currently in use.

    Parameters:
    fig (matplotlib.figure.Figure): The matplotlib figure to be displayed.

    Returns:
    None
    """
    # Create a dummy figure
    dummy = plt.figure()

    # Retrieve the manager of the dummy figure
    new_manager = dummy.canvas.manager

    # Assign the input figure to the new manager's canvas
    new_manager.canvas.figure = fig

    # Set the canvas of the input figure to the new manager's canvas
    fig.set_canvas(new_manager.canvas)


def plot_polygon(polygon: TexturedPolygon, ax: plt.Axes = None, fill: bool = True, **kwargs) -> plt.Figure:
    """
    Plots a given polygon on a matplotlib figure.

    Args:
        polygon (TexturedPolygon): The polygon to be plotted.
        ax (plt.Axes, optional): The axes object to draw the plot onto. If None, a new figure and axes object are created. Defaults to None.
        fill (bool, optional): If True, fills the polygon with its texture color at 50% transparency. Defaults to True.
        **kwargs: Arbitrary keyword arguments to be passed to the plot function.

    Returns:
        plt.Figure: The figure object with the plotted polygon.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Determine the color to use for the polygon's border and fill
    if isinstance(polygon, TexturedPolygon) and polygon.texture.color is not None:
        border_color = polygon.texture.color
    else:
        border_color = 'tab:red'

    # Set default color for the border if not specified in kwargs
    kwargs.setdefault('color', border_color)

    # Plot the exterior of the polygon
    ax.plot(*polygon.exterior.xy, **kwargs)

    # Fill the polygon if fill is True
    if fill:
        # Create a fill color with 50% transparency
        # fill_color = border_color + '80'  # Assuming border_color is a hex color, '80' corresponds to 50% opacity
        ax.fill(*polygon.exterior.xy, color=border_color, alpha=0.5)

    # Plot and fill interiors if any
    if polygon.interiors:
        for interior in polygon.interiors:
            ax.plot(*interior.xy, **kwargs)
            ax.fill(*interior.xy, color=ax.get_facecolor())

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


def create_circular_layout(N: int, d: float = 5, figsize: tuple[int, int] = (5, 5)) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Creates a circular layout for a matplotlib figure with N axes.

    Args:
        N (int): The number of axes to create in the circular layout.
        d (int): The radius of the circular layout.
        figsize (tuple[int, int]): The size of the figure.

    Returns:
        tuple[plt.Figure, list[plt.Axes]]: The figure and the axes.
    """
    fig = plt.figure(figsize=figsize)
    axs = []
    gs = GridSpec(int(2*d*1000 + 500), int(2*d*1000 + 500), figure=fig)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)

    max_a, max_b = 0, 0

    for i in range(N):
        # Calculate the angle for each subplot

        # Set the position of the subplot
        a = int((d*np.cos(theta[i]) + d) * 500)
        b = int((d*np.sin(theta[i]) + d) * 500)

        if a > max_a:
            max_a = a
        if b > max_b:
            max_b = b

        ax = fig.add_subplot(gs[a:a+1000, b:b+1000], projection='3d')

        axs.append(ax)
    axs = [
            fig.add_subplot(gs[
                max_a//2: max_a//2 + 1000,
                max_b//2: max_b//2 + 1000
            ], polar=True)
        ] + axs

    return fig, axs


def plot_3d_matrix(
    matrix: np.ndarray,
    sparcity: int = 1,
    cmap: str = 'viridis',
    ax: plt.Axes = None,
    title: str = None,
    **kwargs
) -> plt.Figure:
    """
    Plots a 3D matrix using a scatter plot to visualize the distribution of values within the matrix.

    This function allows for the visualization of a 3D numpy array by plotting points in a 3D space. The density of points can be adjusted using the `sparcity` parameter, and the color map can be customized.

    Args:
        matrix (np.ndarray): The 3D numpy array to be visualized.
        sparcity (int, optional): The interval at which points are sampled from the matrix to be plotted. Defaults to 1, meaning every point is plotted.
        cmap (str, optional): The colormap used to color the points based on their value. Defaults to 'viridis'.
        ax (plt.Axes, optional): A matplotlib 3D axes object to plot on. If None, a new figure and axes object are created. Defaults to None.
        title (str, optional): The title of the plot. If None, a default title is set. Defaults to None.
        **kwargs: Additional keyword arguments passed to `ax.scatter`.

    Returns:
        plt.Figure: The matplotlib figure object containing the plot.

    Example:
        >>> matrix = np.random.rand(10, 10, 10)
        >>> plot_3d_matrix(matrix, sparcity=2, cmap='hot')
        <Figure ...>
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    x, y, z = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]), np.arange(matrix.shape[2]), indexing='ij')

    l, m, n = matrix.shape
    x, y, z = np.meshgrid(
        np.arange(0, l, sparcity),
        np.arange(0, m, sparcity),
        np.arange(0, n, sparcity),
        indexing='ij'
    )
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    w = matrix[::sparcity, ::sparcity, ::sparcity].flatten()

    indices = np.where(
            np.sqrt((x + y + z)**2) < np.sqrt(l**2 + m**2 + n**2),
    )

    x = x[indices]
    y = y[indices]
    z = z[indices]
    w = w[indices]
    ax.scatter(x, y, z, c=w, cmap=cmap, **kwargs)

    if title is None:
        ax.set_title('3D Distribution (Every {}th Voxel)'.format(sparcity))
    else:
        ax.set_title(title)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim(0, l)
    ax.set_ylim(0, m)
    ax.set_zlim(0, n)

    ax.view_init(elev=45, azim=45)

    return fig
