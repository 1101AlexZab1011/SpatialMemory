import numpy as np


def triple_arange(start: float, stop: float, step: float = 1) -> np.ndarray:
    """
    Generate a tripled range of values within a specified range.

    This function generates an array of values that combines three ranges. The central range is the original
    range, and the other two ranges are offset by the size of the original range.

    Args:
        start (float): The start of the original range (inclusive).
        stop (float): The end of the original range (exclusive).
        step (float, optional): The step size between values (default is 1).

    Returns:
        np.ndarray: An array containing a tripled range of values.

    """
    x = np.arange(start, stop, step)
    triple_x = np.zeros((3 * x.size,))
    triple_x[:x.size] = x - x.size
    triple_x[x.size:2*x.size] = x
    triple_x[2*x.size:] = x + x.size
    return triple_x