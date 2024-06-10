from abc import ABC

from bbtoolkit.utils.indextools import wrap_indices


class AbstractIndexer(ABC):
    """
    Abstract base class for operations based on the shape of an array.

    Attributes:
        shape (Tuple[int, ...]): The shape of the array to operate on.
    """

    def __init__(self, shape: tuple[int, ...]):
        """
        Initializes the ShapeBasedIndexer with the specified shape.

        Args:
            shape (Tuple[int, ...]): The shape of the array to operate on.
        """
        self.shape = shape
        self.validate_shape()

    def validate_shape(self):
        """
        Validates the shape to ensure it meets specific criteria (e.g., non-empty, positive dimensions).
        This method can be overridden by subclasses if they require specific validations.
        """
        if not all(isinstance(dim, int) and dim > 0 for dim in self.shape):
            raise ValueError("All dimensions must be positive integers.")


class IteratorIndexer(AbstractIndexer):
    """
    An iterator for indexing through an array shape, iterating over every possible index.

    Attributes:
        shape (tuple[int, ...]): The shape of the array to iterate over.
    """
    def __init__(self, shape: tuple[int, ...]):
        """
        Initializes the IteratorIndexer with the specified shape.

        Args:
            shape (tuple[int, ...]): The shape of the array to iterate over.
        """
        super().__init__(shape)
        self.ranges = [range(dim) for dim in shape]
        self.current = [0] * len(shape)  # Initialize current index to start of each dimension
        self.started = False  # To handle the first increment

    def __iter__(self):
        """
        Returns the iterator object itself.
        """
        return self

    def __next__(self):
        """
        Returns the next index in the iteration, wrapping around at the end of each dimension.

        Returns:
            tuple[int, ...]: The next index in the iteration.

        Raises:
            StopIteration: When the iteration is complete.
        """
        if not self.started:
            self.started = True
            if all(size == 0 for size in self.shape):  # Handle empty shape
                raise StopIteration
            return tuple(self.current)  # Return the first index if not started

        for i in range(len(self.shape) - 1, -1, -1):
            if self.current[i] < self.shape[i] - 1:
                self.current[i] += 1
                for j in range(i + 1, len(self.shape)):
                    self.current[j] = 0
                return tuple(self.current)
            elif i == 0:
                raise StopIteration
            self.current[i] = 0  # Reset current index at dimension i and continue


class WrapperIndexer(AbstractIndexer):
    """
    A wrapper for indexing, providing wrapped indices according to the array's shape.

    Attributes:
        shape (tuple[int, ...]): The shape of the array for which to provide wrapped indices.
    """
    def __init__(self, shape: tuple[int, ...]):
        """
        Initializes the WrapperIndexer with the specified shape.

        Args:
            shape (tuple[int, ...]): The shape of the array for which to provide wrapped indices.
        """
        super().__init__(shape)

    def __getitem__(self, item: tuple[int, ...]):
        """
        Returns wrapped indices for the given item.

        Args:
            item (tuple[int, ...]): The indices to wrap.

        Returns:
            tuple[int, ...]: The wrapped indices.
        """
        return wrap_indices(item, self.shape)
