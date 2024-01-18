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