from typing import Callable
import numpy as np


def operation3d(a: np.ndarray, b: np.ndarray, operation: Callable[[np.ndarray, np.ndarray], np.ndarray], return_2d: bool = False) -> np.ndarray:
    """
    Operation between two 2-dimensional tensors.
    (m, n) - (k, n) = (k, m, n)

    Args:
        a (np.ndarray): First tensor of shape (m, n).
        b (np.ndarray): Second tensor of shape (k, n).
        operation (Callable[[np.ndarray, np.ndarray], np.ndarray]): Operation to perform on the tensors. (This operation will be performed k times to a and each row of b.)
        return_2d (bool): If True, returns a 2-dimensional tensor. Default is False.

    Returns:
        np.ndarray: Subtracted tensor of shape (k, m, n).
    """
    m, n = a.shape
    k, _ = b.shape
    c = operation(a[None, :, :], b[:, None, :])

    if return_2d:
        return c.reshape(-1, c.shape[-1])

    return c.reshape(k, m, n)


def operation3d_batched(a: np.ndarray, b: np.ndarray, operation: Callable[[np.ndarray, np.ndarray], np.ndarray], return_2d: bool = False) -> np.ndarray:
    """
    Operation between two batched 2-dimensional tensors.
    (m, n) - (k, n) = (k, m, n)

    Args:
        a (np.ndarray): First tensor of shape (m, n).
        b (np.ndarray): Second tensor of shape (k, n).
        operation (Callable[[np.ndarray, np.ndarray], np.ndarray]): Operation to perform on the tensors. (This operation will be performed k times to a and each row of b.)
        return_2d (bool): If True, returns a 2-dimensional tensor. Default is False.

    Returns:
        np.ndarray: Subtracted tensor of shape (k, m, n).
    """
    n_batches, m, n = a.shape
    n_batches_, k, _ = b.shape

    if n_batches != n_batches_:
        raise ValueError(f'Number of batches of a ({n_batches}) and b ({n_batches_}) must be equal.')

    c = operation(a[:, None, :, :], b[:, :, None, :])

    if return_2d:
        return c.reshape(n_batches, -1, c.shape[-1])

    return c.reshape(n_batches, k, m, n)


def sub3d(
    a: np.ndarray,
    b: np.ndarray,
    return_2d: bool = False,
    batch_first: bool = False
) -> np.ndarray:
    """
    Subtracts two 2-dimensional tensors.
    (m, n) - (k, n) = (k, m, n)

    Args:
        a (np.ndarray): First tensor of shape (m, n).
        b (np.ndarray): Second tensor of shape (k, n).
        return_2d (bool): If True, returns a 2-dimensional tensor. Default is False.
        batch_first (bool): If True, the 1st dimension is considered as batch. Default is False.

    Returns:
        np.ndarray: Subtracted tensor of shape (k, m, n).
    """
    operator = operation3d_batched if batch_first else operation3d
    return operator(a, b, lambda a, b: a - b, return_2d)


def cross3d(
    a: np.ndarray,
    b: np.ndarray,
    return_2d: bool = False,
    batch_first: bool = False
) -> np.ndarray:
    """
    Cross product of two 2-dimensional tensors.
    (m, n) - (k, n) = (k, m, n)

    Args:
        a (np.ndarray): First tensor of shape (m, n).
        b (np.ndarray): Second tensor of shape (k, n).
        return_2d (bool): If True, returns a 2-dimensional tensor. Default is False.
        batch_first (bool): If True, the 1st dimension is considered as batch. Default is False.

    Returns:
        np.ndarray: Subtracted tensor of shape (k, m, n).
    """
    operator = operation3d_batched if batch_first else operation3d
    return operator(a, b, lambda a, b: np.cross(a, b), return_2d)


def divide3d(
    a: np.ndarray,
    b: np.ndarray,
    return_2d: bool = False,
    batch_first: bool = False
) -> np.ndarray:
    """
    Cross product of two 2-dimensional tensors.
    (m, n) - (k, n) = (k, m, n)

    Args:
        a (np.ndarray): First tensor of shape (m, n).
        b (np.ndarray): Second tensor of shape (k, n).
        return_2d (bool): If True, returns a 2-dimensional tensor. Default is False.
        batch_first (bool): If True, the 1st dimension is considered as batch. Default is False.
        batch_first (bool): If True, the 1st dimension is considered as batch. Default is False.

    Returns:
        np.ndarray: Subtracted tensor of shape (k, m, n).
    """
    operator = operation3d_batched if batch_first else operation3d
    return operator(a, b, lambda a, b: a / b, return_2d)


def duplicate_along_axes(matrix: np.ndarray) -> np.ndarray:
    """
    Duplicates elements of the input matrix along its axes.

    Args:
        matrix (np.ndarray): A numpy array to be duplicated along its axes.

    Returns:
        np.ndarray: The resulting array after duplication along its axes.
    """
    reshape = list()
    tile = list()
    for dim in matrix.shape:
        reshape.extend([1, dim])
        tile.extend([dim, 1])
    reshaped = matrix.reshape(reshape)
    return np.tile(reshaped, tile)
