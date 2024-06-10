import numpy as np
from bbtoolkit.utils.indextools.indexers import AbstractIndexer, WrapperIndexer


class AttractorIndexer(AbstractIndexer):
    """
    An indexer that shifts indices around a central attractor point, applying periodic boundary conditions.

    Attributes:
        shape (tuple[int, ...]): The shape of the array for which to shift indices.
    """
    def __init__(self, shape: tuple[int, ...]):
        """
        Initializes the AttractorIndexer with the specified shape.

        Args:
            shape (tuple[int, ...]): The shape of the array for which to shift indices.
        """
        super().__init__(shape)
        self.indexer = WrapperIndexer(shape)

    def __getitem__(self, item: tuple[int, ...]):
        """
        Returns shifted indices for the given item, applying periodic boundary conditions.

        Args:
            item (tuple[int, ...]): The indices to shift.

        Returns:
            np.ix_: The shifted indices suitable for numpy indexing.
        """
        return np.ix_(*[
            (np.arange(dim) - shift) % dim
            for shift, dim in zip(self.indexer[item], self.shape)
        ])


class InverseAttractorIndexer(AbstractIndexer):
    """
    An indexer that provides inverse attractor indices, applying periodic boundary conditions without shifting.

    Attributes:
        shape (tuple[int, ...]): The shape of the array for which to provide indices.
    """
    def __init__(self, shape: tuple[int, ...]):
        """
        Initializes the InverseAttractorIndexer with the specified shape.

        Args:
            shape (tuple[int, ...]): The shape of the array for which to provide indices.
        """
        super().__init__(shape)
        self.indexer = WrapperIndexer(shape)  # Assuming WrapperIndexer is defined elsewhere

    def __getitem__(self, item: tuple[int, ...]):
        """
        Returns indices for the given item, applying periodic boundary conditions without shifting.

        Args:
            item (tuple[int, ...]): The indices to provide.

        Returns:
            np.ix_: The indices suitable for numpy indexing.
        """
        return np.ix_(*[
            (np.arange(dim) ) % dim
            for dim in self.shape
        ])
