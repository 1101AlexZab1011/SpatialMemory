import numpy as np


def triple_gaussian(
    amplitude: float,
    width: np.ndarray,
    x0: int,
    offset: int,
    std: float
) -> np.ndarray:
    """
    Generate three Gaussian functions with specified parameters.

    Args:
        amplitude (float): The maximum amplitude (height) of the Gaussian curves.
        width (np.ndarray): An array of values representing the input domain (e.g., x-values).
        x0 (int): The center of the final curve (mean).
        offset (int): An offset value that define modes (x0 ± offset) of final curve.
        std (float): The standard deviation of the Gaussian curves, controlling its width.

    Returns:
        np.ndarray: An array of values representing the Gaussian function evaluated at each point in the 'width' array.

    Example:
        >>> amplitude = 1.0
        >>> width = np.linspace(0, 10, 100)
        >>> x0 = 5
        >>> offset = 2
        >>> std = 1.0
        >>> result = gaussian(amplitude, width, x0, offset, std)
    """
    return amplitude * (
        np.exp(
            -((width - x0)/std)**2
        ) +
        np.exp(
            -((width - x0 - offset)/std)**2
        ) +
        np.exp(
            -((width - x0 + offset)/std)**2
        )
    )