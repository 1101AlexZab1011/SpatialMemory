from abc import ABC, abstractmethod

import numpy as np

from bbtoolkit.structures.attractors.indexers import AttractorIndexer, InverseAttractorIndexer
from bbtoolkit.utils.indextools import create_index_matrix, select_data, wrap_indices


class AbstractAttractorState(ABC):
    """
    An abstract class representing the state of an attractor system. This class defines the basic structure and required methods for any attractor state implementation.

    Attributes:
        kernel (np.ndarray): The kernel array representing the attractor's influence pattern.
        state (np.ndarray): The current state of the system.
        indexer (AttractorIndexer): An indexer object to handle wrapping and indexing logic.
        inplace (bool): If True, operations modify the state in place. Defaults to False.
        weights (np.ndarray, optional): An array of weights for the state transformation. If None, no weighting is applied.
    """
    def __init__(
        self,
        kernel: np.ndarray,
        state: np.ndarray,
        indexer: AttractorIndexer,
        inplace: bool = False,
        weights: np.ndarray = None,
    ):
        """
        Initializes the AbstractAttractorState with a kernel, state, and optional weights.

        Args:
            kernel (np.ndarray): The kernel array.
            state (np.ndarray): The current state array.
            indexer (AttractorIndexer): The indexer object for handling indexing.
            inplace (bool, optional): Whether to modify the state in place. Defaults to False.
            weights (np.ndarray, optional): An array of weights for state transformation. Defaults to None.
        """
        self.kernel = kernel
        self.state = state
        self.indexer = indexer
        self.inplace = inplace
        self.weights = weights

    @property
    @abstractmethod
    def shape(self):
        """
        Abstract property that should return the shape of the kernel.

        Returns:
            tuple[int, ...]: The shape of the kernel array.
        """
        pass

    @property
    @abstractmethod
    def ndim(self):
        """
        Abstract property that should return the number of dimensions of the kernel.

        Returns:
            int: The number of dimensions of the kernel array.
        """
        pass

    @abstractmethod
    def __getitem__(self, indices: np.ndarray):
        """
        Abstract method for indexing into the attractor state.

        Args:
            indices (np.ndarray): The indices to access.

        Returns:
            np.ndarray: The result of applying the kernel and indexer to the state at the given indices.
        """
        pass

    @abstractmethod
    def __matmul__(self, other: np.ndarray):
        """
        Abstract method for transforming an array based on the attractor state.

        Args:
            other (np.ndarray): The array to be transformed by the attractor state.

        Returns:
            np.ndarray: The transformed array.
        """
        pass

    @abstractmethod
    def values(self) -> np.ndarray:
        """
        Abstract method that should return the weighted state if weights are provided, otherwise some default state representation.

        Returns:
            np.ndarray: The weighted state or a default state representation.
        """
        pass

class AbstractAttractor(ABC):
    """
    An abstract class representing an attractor system. This class defines the basic structure and required methods for any attractor implementation.

    Attributes:
        kernel (np.ndarray): The kernel array representing the attractor's influence pattern.
        inplace (bool): If True, operations modify states in place. Defaults to False.
        precompute (bool): If True, precomputes weights for the attractor. Defaults to False.
        shape (tuple[int, ...], optional): The shape of the attractor system. If None, derived from the kernel. Defaults to None.
    """
    def __init__(
        self,
        kernel: np.ndarray,
        inplace: bool = False,
        precompute: bool = False,
        shape: tuple[int, ...] = None,
    ):
        """
        Initializes the AbstractAttractor with a kernel and optional inplace modification, precomputation settings, and shape.

        Args:
            kernel (np.ndarray): The kernel array.
            inplace (bool, optional): Whether to modify states in place. Defaults to False.
            precompute (bool, optional): Whether to precompute weights for the attractor. Defaults to False.
            shape (tuple[int, ...], optional): The shape of the attractor system. If None, derived from the kernel.
        """

        if shape is None:
            shape = kernel.shape

        self.inplace = inplace
        self.precompute = precompute
        self.indexer = AttractorIndexer(shape)
        self.weights = None
        self.kernel = kernel

    @property
    @abstractmethod
    def kernel(self):
        """
        Abstract property that should return the current kernel array.

        Returns:
            np.ndarray: The current kernel array.
        """
        pass

    @property
    @abstractmethod
    def shape(self):
        """
        Abstract property that should return the shape of the attractor system.

        Returns:
            tuple[int, ...]: The shape of the attractor system.
        """
        pass

    @property
    @abstractmethod
    def ndim(self):
        """
        Abstract property that should return the number of dimensions of the attractor system.

        Returns:
            int: The number of dimensions of the attractor system.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Abstract method that should return the length of the attractor system, typically the number of elements or states.

        Returns:
            int: The length of the attractor system.
        """
        pass

    @abstractmethod
    def __call__(self, *args):
        """
        Abstract method that should allow the attractor object to be called like a function, typically to apply the attractor transformation to some input.

        Args:
            *args: Variable length argument list for the attractor transformation.

        Returns:
            Any: The result of the attractor transformation.
        """
        pass

    @abstractmethod
    def get_weights(self):
        """
        Abstract method that should compute and return the weights matrix based on the kernel, used for transforming states.

        Returns:
            np.ndarray: The weights matrix derived from the kernel.
        """
        pass


class SelfAttractorState(AbstractAttractorState):
    """
    Represents the state of a self-attractor system, allowing for interaction with a kernel and state array.
    """
    def __init__(
        self,
        kernel: np.ndarray,
        state: np.ndarray,
        indexer: AttractorIndexer,
        inplace: bool = False,
        weights: np.ndarray = None,
    ):
        """
        Initializes the SelfAttractorState with a kernel, state, and optional weights.

        Args:
            kernel (np.ndarray): The kernel array.
            state (np.ndarray): The current state array.
            indexer (AttractorIndexer): The indexer object for handling indexing.
            inplace (bool, optional): Whether to modify the state in place. Defaults to False.
            weights (np.ndarray, optional): An array of weights for state transformation. Defaults to None.
        """
        super().__init__(
            kernel,
            state,
            indexer,
            inplace,
            weights
        )

    @property
    def shape(self):
        """
        Returns the shape of the kernel.

        Returns:
            tuple: The shape of the kernel array.
        """
        return self.kernel.shape

    @property
    def ndim(self):
        """
        Returns the number of dimensions of the kernel.

        Returns:
            int: The number of dimensions of the kernel array.
        """
        return self.kernel.ndim

    def __getitem__(self, indices: np.ndarray):
        """
        Allows indexing into the attractor state, applying the kernel and indexer logic.

        Args:
            indices (np.ndarray): The indices to access.

        Returns:
            np.ndarray: The result of applying the kernel and indexer to the state at the given indices.
        """
        indices = wrap_indices(indices, self.kernel.shape)
        ratio = self.kernel[*indices]

        return select_data(self.state, self.indexer[indices])*ratio

    def __matmul__(self, other: np.ndarray):
        """
        Implements the matrix multiplication operation, transforming the 'other' array based on the attractor state.

        Args:
            other (np.ndarray): The array to be transformed by the attractor state.

        Returns:
            np.ndarray: The transformed array.
        """
        if not self.inplace:
            other = other.copy()

        for indices in zip(*np.nonzero(self.kernel)):
            other += self[indices]

        return other

    def values(self) -> np.ndarray:
        """
        Returns the weighted state if weights are provided, otherwise the result of matrix multiplication with a zero array.

        Returns:
            np.ndarray: The weighted state or the result of the attractor operation on a zero array.
        """
        if self.weights is not None:
            state = self.state.copy()
            original_shape = state.shape
            state = np.reshape(state, (-1, self.weights.shape[-1]))
            state = np.transpose(state, (1, 0))
            state = self.weights@state
            state = np.transpose(state, (1, 0))
            return np.reshape(state, original_shape)
        else:
            return self@np.zeros_like(self.state)


class SelfAttractor(AbstractAttractor):
    """
    Represents a self-attractor system capable of transforming states based on a kernel.
    """
    def __init__(
        self,
        kernel: np.ndarray,
        inplace: bool = False,
        precompute: bool = False
    ):
        """
        Initializes the SelfAttractor with a kernel and optional inplace modification and precomputation settings.

        Args:
            kernel (np.ndarray): The kernel array representing the attractor's influence pattern.
            inplace (bool, optional): Whether to modify states in place. Defaults to False.
            precompute (bool, optional): Whether to precompute weights for the attractor. Defaults to False.
        """
        super().__init__(
            kernel,
            inplace,
            precompute
        )

    @property
    def kernel(self):
        """
        Property to get or set the kernel. Setting the kernel also updates the shifted kernel and precomputes weights if enabled.

        Returns:
            np.ndarray: The current kernel array.
        """
        return self._kernel

    @kernel.setter
    def kernel(self, value: np.ndarray):
        """
        Sets the kernel array and updates the shifted kernel and precomputes weights if enabled.

        Args:
            value (np.ndarray): The new kernel array.
        """
        self._kernel = value
        self._kernel_shifted = self.kernel[create_index_matrix(self.kernel.shape)]

        if self.precompute:
            weights = self.get_weights()
            shape = weights.shape
            side = np.prod(shape[:len(shape)//2])
            self.weights = weights.reshape(side, side)

    @property
    def shape(self):
        """
        Returns the shape of the kernel.

        Returns:
            tuple: The shape of the kernel array.
        """
        return self.kernel.shape

    @property
    def ndim(self):
        """
        Returns the number of dimensions of the kernel.

        Returns:
            int: The number of dimensions of the kernel array.
        """
        return self.kernel.ndim

    def __len__(self):
        """
        Returns the length of the kernel.

        Returns:
            int: The length of the kernel array.
        """
        return len(self.kernel)

    def __call__(self, state: np.ndarray) -> SelfAttractorState:
        """
        Allows the object to be called like a function, returning a new SelfAttractorState based on the provided state.

        Args:
            state (np.ndarray): The state array to be transformed by the attractor.

        Returns:
            SelfAttractorState: A new SelfAttractorState object initialized with the current kernel and the provided state.
        """
        if not self.inplace:
            state = state.copy()

        return SelfAttractorState(self._kernel_shifted, state, self.indexer, self.inplace, self.weights)

    def get_weights(self) -> np.ndarray:
        """
        Computes and returns the weights matrix based on the kernel, used for transforming states.

        Returns:
            np.ndarray: The weights matrix derived from the kernel.
        """
        weights_matrix = np.zeros(list(self.shape) + list(self.shape))
        inverse_indexer = InverseAttractorIndexer(self.shape)
        state = self(np.ones(self.shape))

        for indices in zip(*np.nonzero(self._kernel_shifted)):
            weights_matrix[*inverse_indexer[indices], *state.indexer[indices]] = state[indices]

        return weights_matrix


class RingAttractorState(AbstractAttractorState):
    """
    Represents the state of a ring attractor system with multiple kernels and states.
    """
    def __init__(
        self,
        kernel: tuple[np.ndarray, ...],
        state: tuple[np.ndarray, ...],
        indexer: AttractorIndexer,
        inplace: bool = False,
        weights: tuple[np.ndarray, ...] = None
    ):
        """
        Initializes the RingAttractorState with kernels, states, and optional weights.

        Args:
            kernel (tuple[np.ndarray, ...]): The kernel arrays.
            state (tuple[np.ndarray, ...]): The current state arrays.
            indexer (AttractorIndexer): The indexer object for handling indexing.
            inplace (bool, optional): Whether to modify the states in place. Defaults to False.
            weights (tuple[np.ndarray, ...], optional): Arrays of weights for state transformation. Defaults to None.
        """
        super().__init__(
            kernel,
            state,
            indexer,
            inplace,
            weights
        )
        self.state = list(self.state)
        self.state = tuple([self.state[-1]] + self.state[:-1])

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the first kernel.

        Returns:
            tuple[int, ...]: The shape of the first kernel array.
        """
        return self.kernel[0].shape

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions of the first kernel.

        Returns:
            int: The number of dimensions of the first kernel array.
        """
        return self.kernel[0].ndim

    def __len__(self) -> int:
        """
        Returns the number of kernels/states in the attractor system.

        Returns:
            int: The length of the kernels/states tuple.
        """
        return len(self.kernel)

    def __getitem__(self, indices: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Allows indexing into the ring attractor state, applying the kernels and indexer logic.

        Args:
            indices (np.ndarray): The indices to access.

        Returns:
            tuple[np.ndarray, ...]: The result of applying the kernels and indexer to the states at the given indices.
        """
        indices = wrap_indices(indices, self.shape)
        kernels = list(self.kernel)

        return tuple(
            select_data(
                state, self.indexer[indices]
            )*kernel[*indices]
            for state, kernel in zip(self.state, kernels)
        )

    def __matmul__(self, other: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        """
        Implements the matrix multiplication operation, transforming the 'other' tuple of arrays based on the attractor states.

        Args:
            other (tuple[np.ndarray, ...]): The tuple of arrays to be transformed by the attractor states.

        Returns:
            tuple[np.ndarray, ...]: The transformed tuple of arrays.
        """
        if not self.inplace:
            other = tuple(inst.copy() for inst in other)

        for indices in zip(*np.nonzero(np.add(*self.kernel))):
            updates = self[indices]
            other = list(state + update for state, update in zip(other, updates))

        return other

    def values(self) -> np.ndarray:
        """
        Returns the weighted states if weights are provided, otherwise the result of matrix multiplication with a tuple of zero arrays.

        Returns:
            np.ndarray: The weighted states or the result of the attractor operation on a tuple of zero arrays.
        """
        if self.weights is not None:
            state = [s.copy() for s in self.state]
            original_shape = state[0].shape
            state = [s.reshape(-1, self.weights[0].shape[-1]) for s in state]
            state = [np.transpose(s, (1, 0)) for s in state]
            state = [weight@s for weight, s in zip(self.weights, state)]
            state = [np.transpose(s, (1, 0)) for s in state]
            state = [s.reshape(original_shape) for s in state]
            return state
        else:
            return self@tuple(np.zeros_like(state) for state in self.state)


class RingAttractor(AbstractAttractor):
    """
    Represents a ring attractor system capable of transforming states based on multiple kernels.

    Attributes:
        inplace (bool): If True, operations modify states in place. Defaults to False.
        precompute (bool): If True, precomputes weights for the attractor. Defaults to False.
        kernel (tuple[np.ndarray, ...]): The kernel arrays representing the attractor's influence patterns.
    """
    def __init__(
        self,
        kernel: tuple[np.ndarray, ...],
        inplace: bool = False,
        precompute: bool = False
    ):
        """
        Initializes the RingAttractor with kernels and optional inplace modification and precomputation settings.

        Args:
            kernel (tuple[np.ndarray, ...]): The kernel arrays.
            inplace (bool, optional): Whether to modify states in place. Defaults to False.
            precompute (bool, optional): Whether to precompute weights for the attractor. Defaults to False.
        """
        super().__init__(
            kernel,
            inplace,
            precompute,
            kernel[0].shape
        )

    @property
    def kernel(self) -> tuple[np.ndarray, ...]:
        """
        Property to get or set the kernels. Setting the kernels also updates the shifted kernels and precomputes weights if enabled.

        Returns:
            tuple[np.ndarray, ...]: The current tuple of kernel arrays.
        """
        return self._kernel

    @kernel.setter
    def kernel(self, value: tuple[np.ndarray, ...]):
        """
        Sets the kernel arrays and updates the shifted kernels and precomputes weights if enabled.

        Args:
            value (tuple[np.ndarray, ...]): The new tuple of kernel arrays.
        """
        self._kernel = value
        self._kernel_shifted = tuple(kernel[create_index_matrix(kernel.shape)] for kernel in value)

        if self.precompute:
            weights = self.get_weights()
            shape = weights[0].shape
            side = np.prod(shape[:len(shape)//2])
            self.weights = tuple(weight.reshape(side, side) for weight in weights)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the first kernel.

        Returns:
            tuple[int, ...]: The shape of the first kernel array.
        """
        return self.kernel[0].shape

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions of the first kernel.

        Returns:
            int: The number of dimensions of the first kernel array.
        """
        return self.kernel[0].ndim

    def __len__(self) -> int:
        """
        Returns the number of kernels in the attractor system.

        Returns:
            int: The length of the kernels tuple.
        """
        return len(self.kernel)

    def __call__(self, *weights: tuple[np.ndarray, ...]) -> RingAttractorState:
        """
        Allows the object to be called like a function, returning a new RingAttractorState based on the provided weights.

        Args:
            *weights (tuple[np.ndarray, ...]): The weights arrays to be transformed by the attractor.

        Returns:
            RingAttractorState: A new RingAttractorState object initialized with the current kernels and the provided weights.
        """
        if not self.inplace:
            weights = tuple(weight.copy() for weight in weights)
        return RingAttractorState(self._kernel_shifted, weights, self.indexer, self.inplace, self.weights)

    def get_weights(self) -> tuple[np.ndarray, ...]:
        """
        Computes and returns the weights matrices based on the kernels, used for transforming states.

        Returns:
            tuple[np.ndarray, ...]: The tuple of weights matrices derived from the kernels.
        """
        inverse_indexer = InverseAttractorIndexer(self.shape)

        weights_matrices = [np.zeros(list(self.shape) + list(self.shape)) for _ in range(len(self.kernel))]
        state = self(*[np.ones(self.shape) for _ in range(len(self.kernel))])

        for indices in zip(*np.nonzero(np.add(*self._kernel_shifted))):
            state_matrices = state[indices]

            for i in range(len(weights_matrices)):
                weights_matrices[i][*inverse_indexer[indices], *state.indexer[indices]] = state_matrices[i]

        return tuple(weights_matrices)
