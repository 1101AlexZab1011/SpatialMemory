import numpy as np

from bbtoolkit.utils.math import circular_gaussian


def gaussian_kernel_1d(
    n_neurons: int,
    sigma: float = .05,
    center: float = .5,
    amplitude: float = 1.
) -> np.ndarray:
    """
    Generates a 1-dimensional Gaussian kernel over a circular array.

    Args:
        n_neurons (int): The number of neurons in the circular array.
        sigma (float, optional): The standard deviation of the Gaussian distribution, scaled to the circle's circumference. Defaults to .05.
        center (float, optional): The center of the Gaussian distribution, represented as a fraction of the circle's circumference. Defaults to .5.
        amplitude (float, optional): The maximum amplitude of the Gaussian distribution. Defaults to 1.

    Returns:
        np.ndarray: An array representing the Gaussian kernel across the circular array of neurons.
    """

    return circular_gaussian(
        center*2*np.pi,
        n_neurons,
        amplitude,
        angular_sigma=sigma*2*np.pi
    )


def gaussian_kernel_2d(m: int, n: int, sigma: float | tuple[float, float] = None) -> np.ndarray:
    """
    Creates an m x n matrix filled with Gaussian values centered in the matrix, with the option to specify separate sigma values for each dimension.

    Args:
        m (int): The number of rows in the matrix.
        n (int): The number of columns in the matrix.
        sigma (float | tuple[float, float], optional): The standard deviation of the Gaussian distribution for each dimension. If a single float is provided, it is used for both dimensions. If a tuple is provided, each value is used for the corresponding dimension. If None, sigma is set to a tenth of the minimum of m and n for both dimensions.

    Returns:
        np.ndarray: An m x n Gaussian matrix.
    """
    if sigma is None:
        sigma = (min(m, n) / 10.0, min(m, n) / 10.0)
    elif isinstance(sigma, (int, float)):
        sigma = (sigma, sigma)  # Use the same sigma for both dimensions if a single float is provided

    # Unpack sigma values for each dimension
    sigma_x, sigma_y = sigma

    center_x, center_y = (m - 1) / 2.0, (n - 1) / 2.0

    # Create meshgrid for matrix indices
    x, y = np.meshgrid(np.arange(n), np.arange(m))

    # Gaussian formula adjusted for different sigma values in each dimension
    g = np.exp(-(((x - center_y) ** 2 / (2.0 * sigma_y ** 2)) +
                 ((y - center_x) ** 2 / (2.0 * sigma_x ** 2))))

    return g


def ricker_kernel_2d(m: int, n: int, a: float = None) -> np.ndarray:
    """
    Creates a matrix populated with the Ricker wavelet (also known as the "Mexican hat" wavelet).

    Args:
        m (int): The number of rows in the matrix.
        n (int): The number of columns in the matrix.
        a (float, optional): The parameter controlling the width of the Ricker wavelet. If None, it's set to a tenth of the minimum of m and n.

    Returns:
        np.ndarray: An m x n matrix with the Ricker wavelet applied.
    """
    # Create an m x n matrix
    matrix = np.zeros((m, n))

    if a is None:
        a = min(m, n) // 10

    # Calculate the center of the matrix
    center_x, center_y = m // 2, n // 2

    # Iterate over the matrix to apply the Ricker wavelet function
    for i in range(m):
        for j in range(n):
            # Calculate distances from the center
            x = i - center_x
            y = j - center_y
            # Calculate the Ricker wavelet value
            factor = (x**2 + y**2) / a**2
            matrix[i, j] = (1 - factor) * np.exp(-factor / 2)

    return matrix


def gaussian_kernel_3d(l: int, m: int, n: int, sigma: float | tuple[float, float, float] = None) -> np.ndarray:
    """
    Creates a 3D Gaussian matrix with specified dimensions and standard deviation.

    Args:
        l (int): The size of the first dimension.
        m (int): The size of the second dimension.
        n (int): The size of the third dimension.
        sigma (float | tuple[float, float, float], optional): The standard deviation of the Gaussian distribution. If a single float is provided, it is used for all dimensions. If a tuple is provided, each value is used for the corresponding dimension. If None, sigma is set to a tenth of the minimum dimension size.

    Returns:
        np.ndarray: A 3D Gaussian matrix.
    """
    if sigma is None:
        sigma = (min(l, m, n) / 10.0, min(l, m, n) / 10.0, min(l, m, n) / 10.0)
    elif isinstance(sigma, (int, float)):
        sigma = (sigma, sigma, sigma)  # Use the same sigma for all dimensions if a single float is provided

    # Unpack sigma values for each dimension
    sigma_x, sigma_y, sigma_z = sigma

    center_x, center_y, center_z = (l - 1) / 2.0, (m - 1) / 2.0, (n - 1) / 2.0
    x = np.arange(0, l, 1)
    y = np.arange(0, m, 1)
    z = np.arange(0, n, 1)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    # Gaussian formula adjusted for different sigma values in each dimension
    g = np.exp(-(((x - center_x) ** 2 / (2.0 * sigma_x ** 2)) +
                 ((y - center_y) ** 2 / (2.0 * sigma_y ** 2)) +
                 ((z - center_z) ** 2 / (2.0 * sigma_z ** 2))))
    return g
