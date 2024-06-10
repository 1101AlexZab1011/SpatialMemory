import numpy as np


def remove_slice(slices: list[slice], i: int) -> list[slice]:
    """
    Removes a slice from a list of slices and adjusts the remaining slices accordingly.

    Args:
        slices (list[slice]): A list of slice objects.
        i (int): The index of the slice to be removed.

    Returns:
        list[slice]: A list of adjusted slice objects after removing the specified slice.

    Example:
        >>> remove_slice([slice(0, 5), slice(5, 10), slice(10, 15)], 1)
        [slice(0, 5), slice(5, 10)]
    """
    lengths = [slice_.stop - slice_.start for slice_ in slices]
    lengths.pop(i)
    cumulative_lengths = np.cumsum(lengths).tolist()

    return [
        slice(from_, to)
        for from_, to in zip([0] + cumulative_lengths[:-1], cumulative_lengths)
    ]


def generate_index_shapes(shape: int) -> list[tuple[int, ...]]:
    """
    Generates a list of shapes for indexing, each shape tailored to iterate over one dimension.

    Args:
        shape (int): The size of each dimension in the array.

    Returns:
        list[tuple[int, ...]]: A list of tuples, where each tuple represents the shape for indexing a specific dimension.
    """
    return [(1,) * i + (shape[i],) + (1,) * (len(shape) - i - 1) for i in range(len(shape))]


def generate_tile_shapes(shape: tuple[int, ...]) -> tuple[tuple[int, ...]]:
    """
    Generates a tuple of shapes for tiling, where each shape is designed for broadcasting over one dimension.

    Args:
        shape (tuple[int, ...]): The original shape of the array.

    Returns:
        tuple[tuple[int, ...]]: A tuple of tuples, each representing the shape for tiling across a specific dimension.
    """
    return tuple(tuple(1 if j == i else shape[j] for j in range(len(shape))) for i in range(len(shape)))


def create_index_matrix(shape: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    """
    Creates a tuple of index matrices for each dimension, centered around the middle of the dimension.

    Args:
        shape (tuple[int, ...]): The shape of the array for which to create index matrices.

    Returns:
        tuple[np.ndarray, ...]: A tuple of numpy arrays, each being an index matrix for a dimension.
    """
    # Calculate the center indices
    center = [dim//2 for dim in shape]
    indices = [
        np.arange(dim).reshape(*index_shape) - center
        for dim, center, index_shape in zip(shape, center, generate_index_shapes(shape))
    ]

    return tuple(
        np.tile(
            index, tile_shape
        )
        for index, tile_shape in zip(indices, generate_tile_shapes(shape))
    )


def select_data(data: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Selects data from an array using provided indices, supporting both direct and advanced indexing.

    Args:
        data (np.ndarray): The data array from which to select.
        indices (np.ndarray): The indices for selection.

    Returns:
        np.ndarray: The selected data.
    """
    # Check dimensions
    if isinstance(indices, (tuple, list)):
        indices_ndim = len(indices)
        indices = [*indices]
    else:
        indices_ndim = indices.ndim
        indices = [indices]

    if indices_ndim >= data.ndim:
        # Directly use indices for selection
        return data[*indices]
    else:
        # Prepare a tuple for advanced indexing
        idx = [slice(None)] * (data.ndim - indices_ndim) + indices
        return data[tuple(idx)]


def wrap_indices(indices: tuple[int, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Wraps indices around the given shape, effectively implementing periodic boundary conditions.

    Args:
        indices (tuple[int, ...]): The indices to wrap.
        shape (tuple[int, ...]): The shape of the array to wrap indices around.

    Returns:
        tuple[int, ...]: The wrapped indices.

    Raises:
        ValueError: If the length of indices and shape do not match.
    """
    if len(indices) != len(shape):
        raise ValueError('Indices and shape must have the same length')
    return tuple((((i % s) + s) % s for i, s in zip(indices, shape)))
