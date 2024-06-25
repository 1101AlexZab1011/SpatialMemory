import numpy as np

from bbtoolkit.utils.indextools import create_index_matrix, parity_reorder, reorder_doubled_array, shifted_1d
from bbtoolkit.utils.math.tensor_algebra import duplicate_along_axes


def get_attractor_weights(
    kernel: np.ndarray
) -> np.ndarray:
    """
    Computes attractor weights for a given kernel.

    Args:
        kernel (np.ndarray): The input kernel as a numpy array.

    Returns:
        np.ndarray: The computed attractor weights as a numpy array.
    """
    return reorder_doubled_array(
        duplicate_along_axes(kernel[create_index_matrix(kernel.shape)]),
        shifted_1d(*(np.arange(i) for i in kernel.shape))
    ).transpose(
        parity_reorder(
            tuple(range(2*kernel.ndim))
        )
    ).reshape(
        (side := np.prod(kernel.shape), side)
    )
