import numpy as np

def cart2pol(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts Cartesian coordinates to polar coordinates.

    Parameters:
        x (np.ndarray): Array of x-coordinates in Cartesian space.
        y (np.ndarray): Array of y-coordinates in Cartesian space.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the converted polar coordinates (rho, phi).
            - rho (np.ndarray): Array of radial distances.
            - phi (np.ndarray): Array of angular positions in radians.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts polar coordinates to Cartesian coordinates.

    Parameters:
        rho (np.ndarray): Array of radial distances.
        phi (np.ndarray): Array of angular positions in radians.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the converted Cartesian coordinates (x, y).
            - x (np.ndarray): Array of x-coordinates in Cartesian space.
            - y (np.ndarray): Array of y-coordinates in Cartesian space.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


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