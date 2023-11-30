from matplotlib import pyplot as plt
from bbtoolkit.structures.geometry import TexuredPolygon


def plot_polygon(polygon: TexuredPolygon, ax: plt.Axes = None, **kwargs) -> plt.Figure:
    """
    Plots a given polygon on a matplotlib figure.

    Args:
        polygon (TexuredPolygon): The polygon to be plotted.
        ax (plt.Axes, optional): The axes object to draw the plot onto. If None, a new figure and axes object are created. Defaults to None.
        **kwargs: Arbitrary keyword arguments to be passed to the plot function.

    Returns:
        plt.Figure: The figure object with the plotted polygon.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if isinstance(polygon, TexuredPolygon) and polygon.texture.color is not None:
        kwargs.setdefault('color', polygon.texture.color)
    else:
        kwargs.setdefault('color', 'tab:red')

    ax.plot(*polygon.exterior.xy, **kwargs)

    if polygon.interiors:
        for interior in polygon.interiors:
            ax.plot(*interior.xy, **kwargs)

    return fig