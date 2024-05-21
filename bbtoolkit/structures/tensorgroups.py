from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Generator, KeysView, Literal, ValuesView
from matplotlib import pyplot as plt
from netgraph import Graph
import numpy as np
import pandas as pd

from bbtoolkit.utils.datautils import Copyable, WritablePickle



class BaseTensor(Copyable):
    """
    Abstract class for representing multidimensional data

    Attributes:
        weights (np.ndarray): A NumPy array containing the weights of a tensor.
    """
    def __init__(self, weights: np.ndarray):
        """
        Initialize a BaseTensor object with weights.

        Args:
            weights (np.ndarray): A NumPy array containing the weights of a tensor.
        """
        self.weights = weights


class NamedTensor(BaseTensor):
    """
    A data class representing a named tensor.

    Attributes:
        name (str): The name of the tensor.
        weights (np.ndarray): The numerical data of the tensor, stored in a numpy array.
    """
    def __init__(self, name: str, weights: np.ndarray):
        super().__init__(weights)
        self.name = name


class DirectedTensor(BaseTensor):
    """
    Represents the weights between two layers in a neural network.

    Attributes:
        from_ (str): The name or identifier of the source layer.
        to (str): The name or identifier of the target layer.
        weights (np.ndarray): A NumPy array containing the weights connecting the source and target layers.

    Example:
    ```python
    # Creating a DirectedTensor object
    weights = DirectedTensor(from_='hidden_layer', to='output_layer', weights=np.array([[0.5, 0.3], [0.1, 0.8]]))

    # Accessing attributes
    print(weights.from_)  # Output: 'hidden_layer'
    print(weights.to)      # Output: 'output_layer'
    print(weights.weights) # Output: array([[0.5, 0.3],
                            #                [0.1, 0.8]])
    ```
    """
    def __init__(self, from_: str, to: str, weights: np.ndarray):
        """
        Initialize a DirectedTensor object with weights.

        Args:
            from_ (str): The name or identifier of the source layer.
            to (str): The name or identifier of the target layer.
            weights (np.ndarray): A NumPy array containing the weights connecting the source and target layers.
        """
        super().__init__(weights)
        self.from_ = from_
        self.to = to

    def __copy__(self):
        """
        Create a shallow copy of the DirectedTensor object.
        """
        return DirectedTensor(self.from_, self.to, self.weights.copy())


def dict2directed_tensor(data: dict[str, dict[str, np.ndarray]]) -> Generator[DirectedTensor, None, None]:
    """
    Converts a dictionary of dictionaries of weights to a generator of DirectedTensor objects

    Args:
        data (dict[str, dict[str, np.ndarray]]): Dictionary of dictionaries of weights

    Yields:
        Generator[DirectedTensor, None, None]: Generator of DirectedTensor objects
    """
    for from_ in data:
        for to in data[from_]:
            yield DirectedTensor(from_, to, data[from_][to])


class AbstractTensorGroup(WritablePickle, Copyable, ABC):
    @property
    @abstractmethod
    def data(self):
        """
        Abstract property to get data of tensor group
        """
        ...

    @abstractmethod
    def add_tensor(self, tensor: 'AbstractTensorGroup'):
        """
        Abstract method to add tensor to tensor group

        Args:
            tensor (AbstractTensorGroup): Tensor to add to tensor group
        """
        ...

    @abstractmethod
    def remove_tensor(self, *args, **kwargs):
        """
        Abstract method to remove tensor from tensor group

        Args:
            *args, **kwargs: Any additional parameters for removing tensor
        """
        ...

    @abstractmethod
    def __getitem__(self, key: Any) -> Any:
        """
        Abstract method to get item from tensor group

        Args:
            key (Any): Key to get item from tensor group
        """
        ...

    @abstractmethod
    def __iter__(self) -> Generator:
        """
        Abstract method to iterate over tensor group

        Yields:
            Generator: Generator of tensor group
        """
        ...

    @abstractmethod
    def operation_with(
        self,
        other: 'AbstractTensorGroup',
        operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *args, **kwargs
    ) -> 'AbstractTensorGroup':
        """
        Abstract method to perform operations between two tensor groups

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to perform operation with
            operation (Callable): Concrete operation to perform. Depends on the details of tensor groups structure
            args, kwargs: Any additional parameters for performing operation
        """
        ...

    @abstractmethod
    def __add__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to add tensor group

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to add to current tensor group
        """
        ...

    @abstractmethod
    def __iadd__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to add tensor group inplace

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to add to current tensor group
        """
        ...

    @abstractmethod
    def __sub__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to subtract tensor group

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to subtract from current tensor group
        """
        ...

    @abstractmethod
    def __isub__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to subtract tensor group inplace

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to subtract from current tensor group
        """
        ...

    @abstractmethod
    def __mul__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to multiply tensor group

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to multiply with current tensor group
        """
        ...

    @abstractmethod
    def __imul__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to multiply tensor group inplace

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to multiply with current tensor group
        """
        ...

    @abstractmethod
    def __div__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to divide tensor group

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to divide from current tensor group
        """
        ...

    @abstractmethod
    def __idiv__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to divide tensor group inplace

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to divide from current tensor group
        """
        ...


    @abstractmethod
    def __floordiv__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to floor divide tensor group

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to floor divide from current tensor group
        """
        ...

    @abstractmethod
    def __ifloordiv__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to floor divide tensor group inplace

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to floor divide from current tensor group
        """
        ...

    @abstractmethod
    def __mod__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to mod tensor group

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to mod from current tensor group
        """
        ...

    @abstractmethod
    def __imod__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to mod tensor group inplace

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to mod from current tensor group
        """
        ...

    @abstractmethod
    def __pow__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to raise tensor group

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to raise from current tensor group
        """
        ...

    @abstractmethod
    def __ipow__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to raise tensor group inplace

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to raise from current tensor group
        """
        ...

    @abstractmethod
    def __matmul__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to matrix multiply tensor group

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to matrix multiply from current tensor group
        """
        ...

    @abstractmethod
    def __imatmul__(self, other: 'AbstractTensorGroup') -> 'AbstractTensorGroup':
        """
        Abstract method to matrix multiply tensor group inplace

        Args:
            other (AbstractTensorGroup): Other AbstractTensorGroup instance to matrix multiply from current tensor group
        """
        ...

    @abstractmethod
    def map(self, func: Callable[[np.ndarray], np.ndarray]) -> 'AbstractTensorGroup':
        """
        Abstract method to map tensor group

        Args:
            func (Callable): Function to map tensor group
        """
        ...

    @abstractmethod
    def __neg__(self) -> 'AbstractTensorGroup':
        """
        Abstract method to negate tensor group
        """
        ...

    @abstractmethod
    def __abs__(self) -> 'AbstractTensorGroup':
        """
        Abstract method to take absolute value of tensor group
        """
        ...

    @property
    @abstractmethod
    def T(self) -> 'AbstractTensorGroup':
        """
        Abstract method to transpose tensor group
        """
        ...

    @abstractmethod
    def __contains__(self, item: Any) -> bool:
        """
        Abstract method to check if item is in tensor group

        Args:
            item (Any): Item to check if it is in tensor group
        """
        ...


class TensorGroup(AbstractTensorGroup):
    """
    A class that groups multiple NamedTensors and provides methods to add or remove tensors from the group.

    Attributes:
        data (Dict[str, np.ndarray]): A dictionary mapping tensor names to their corresponding numpy arrays.
    """
    def __init__(self, *tensors: NamedTensor, **kwargs):
        """
        Initializes a TensorGroup with a sequence of NamedTensors.

        Args:
            *tensors (NamedTensor): An unpacked sequence of NamedTensor objects to be included in the group.
                Can also accept a single dictionary to create TensorGroup using keys and values.
            **kwargs: TensorGroup can be created from raw dict.
        """
        self._data = dict()
        self.update(*tensors)

        if kwargs is not None:
            self._from_dict(kwargs)

    def _from_dict(self, data: dict[str, np.ndarray]):
        """
        Initializes a TensorGroup with a dictionary of tensors.

        Args:
            data (dict[str, np.ndarray]): A dictionary mapping tensor names to their corresponding numpy arrays.
        """
        for key, value in data.items():
            self.add_tensor(NamedTensor(key, value))

    def update(
        self,
        *tensors: NamedTensor,
    ):
        """
        Updates the TensorGroup with new NamedTensors.

        Args:
            *tensors (NamedTensor): An unpacked sequence of NamedTensor objects to be included in the group.
        """
        if len(tensors) == 1 and isinstance(tensors[0], dict):
            self._from_dict(tensors[0])
        else:
            for tensor in tensors:
                self.add_tensor(tensor)

    @property
    def data(self):
        """
        Returns the dictionary containing the tensor data.
        """
        return self._data

    @data.setter
    def data(self, value):
        """
        Sets the dictionary containing the tensor data.
        """
        self._data = value
        for key in value:
            self.__setattr__(key, value[key])

    def add_tensor(self, tensor: NamedTensor):
        """
        Adds a NamedTensor to the TensorGroup.

        Args:
            tensor (NamedTensor): The NamedTensor to be added to the group.
        """
        self.data[tensor.name] = tensor.weights
        self.__setattr__(tensor.name, self.data[tensor.name])

    def remove_tensor(self, name: str):
        """
        Removes a NamedTensor from the TensorGroup by its name.

        Args:
            name (str): The name of the tensor to be removed.

        Raises:
            KeyError: If the tensor with the given name does not exist in the group.
        """
        del self.data[name]
        del self.__dict__[name]

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Returns the numpy array corresponding to the given tensor name.

        Args:
            key (str): The name of the tensor.

        Returns:
            np.ndarray: The numerical data of the tensor.
        """
        return self.data[key]

    def __setitem__(self, key: str, value: np.ndarray):
        """
        Sets the numpy array corresponding to the given tensor name.

        Args:
            key (str): The name of the tensor.
            value (np.ndarray): The numerical data of the tensor.
        """
        self.add_tensor(NamedTensor(key, value))

    def __iter__(self) -> Generator[NamedTensor, None, None]:
        """
        A generator that yields NamedTensor objects from the TensorGroup.

        Yields:
            NamedTensor: The next NamedTensor in the group.
        """
        for tensor in self.data:
            yield NamedTensor(tensor, self.data[tensor])

    def __repr__(self) -> str:
        """
        Returns a string representation of the TensorGroup.

        Returns:
            str: A string representation of the TensorGroup.
        """
        keys_and_shapes = [
            f"{key}: ({value.shape})"
            for key, value in self.data.items()
        ]
        return f"TensorGroup({', '.join(keys_and_shapes)})"

    def keys(self) -> KeysView[str]:
        """
        Returns the keys of the TensorGroup.

        Returns:
            KeysView[str]: A view of the keys in the TensorGroup.
        """
        return self.data.keys()

    def values(self) -> ValuesView[np.ndarray]:
        """
        Returns the values of the TensorGroup.

        Returns:
            ValuesView[np.ndarray]: A view of the values in the TensorGroup.
        """
        return self.data.values()

    def items(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """
        Returns the items of the TensorGroup.

        Yields:
            Generator[tuple[str, np.ndarray], None, None]: A generator of key-value pairs in the TensorGroup.
        """
        for key, value in self.data.items():
            yield key, value

    def operation_with(
        self,
        other: 'TensorGroup',
        operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
        on_missing_weights: Literal['raise', 'ignore', 'concat'] | Callable[[np.ndarray], np.ndarray] = 'raise',
        inplace: bool = False
    ) -> 'TensorGroup':
        """
        Applies a binary operation to the tensors of two TensorGroups.

        Args:
            other (TensorGroup): The other TensorGroup to be used in the operation.
            operation (Callable[[np.ndarray, np.ndarray], np.ndarray]): A function that takes two numpy arrays and returns a third.
            on_missing_weights (Literal['raise', 'ignore', 'concat'] | Callable[[np.ndarray], np.ndarray], optional): A strategy to handle missing weights.
                If 'raise', a KeyError is raised. If 'ignore', the missing weights are skipped. If 'concat', the missing weights are concatenated. Defaults to 'raise'.
            inplace (bool): To perform operation inplace or not

        Returns:
            TensorGroup: A new TensorGroup with the results of the operation.
        """

        if not isinstance(other, TensorGroup):
            other = TensorGroup(
                *[
                    NamedTensor(key, other)
                    for key in self.data.keys()
                ]
            )

        new_data = deepcopy(self.data)
        for tensor, weights in other.data.items():
            if tensor in new_data:
                new_data[tensor] = operation(new_data[tensor], weights)
            else:
                if on_missing_weights == 'raise':
                    raise KeyError(f"Tensor '{tensor}' is not present in the group.")
                elif on_missing_weights == 'ignore':
                    continue
                elif on_missing_weights == 'concat':
                    new_data[tensor] = weights
                elif isinstance(on_missing_weights, Callable):
                    new_data[tensor] = on_missing_weights(weights)
                else:
                    raise ValueError(f"Invalid value for 'on_missing_weights': {on_missing_weights}")
        if not inplace:
            return TensorGroup(*[
                NamedTensor(name, weights)
                for name, weights in new_data.items()
            ])
        else:
            self.data = new_data
            return self

    def __add__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Adds two TensorGroups together.

        Args:
            other (TensorGroup): The other TensorGroup to be added.

        Returns:
            TensorGroup: A new TensorGroup with the sum of the two groups.
        """
        return self.operation_with(other, lambda a, b: a + b, on_missing_weights='concat')

    def __iadd__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Adds one TensorGroup to another inplace.

        Args:
            other (TensorGroup): The other TensorGroup to be added.

        Returns:
            TensorGroup: A new TensorGroup with the sum of the two groups.
        """
        return self.operation_with(other, lambda a, b: a + b, on_missing_weights='concat', inplace=True)


    def __sub__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Subtracts one TensorGroup from another.

        Args:
            other (TensorGroup): The other TensorGroup to be subtracted.

        Returns:
            TensorGroup: A new TensorGroup with the difference of the two groups.
        """
        return self.operation_with(other, lambda a, b: a - b, on_missing_weights='ignore')

    def __isub__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Subtracts one TensorGroup from another inplace.

        Args:
            other (TensorGroup): The other TensorGroup to be subtracted.

        Returns:
            TensorGroup: A new TensorGroup with the difference of the two groups.
        """
        return self.operation_with(other, lambda a, b: a - b, on_missing_weights='ignore', inplace=True)

    def __mul__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Multiplies two TensorGroups together.

        Args:
            other (TensorGroup): The other TensorGroup to be multiplied.

        Returns:
            TensorGroup: A new TensorGroup with the product of the two groups.
        """
        return self.operation_with(other, lambda a, b: a*b, on_missing_weights='ignore')

    def __imul__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Multiplies one TensorGroup by another inplace.

        Args:
            other (TensorGroup): The other TensorGroup to be multiplied.

        Returns:
            TensorGroup: A new TensorGroup with the product of the two groups.
        """
        return self.operation_with(other, lambda a, b: a*b, on_missing_weights='ignore', inplace=True)

    def __div__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Divides one TensorGroup by another.

        Args:
            other (TensorGroup): The other TensorGroup to be divided.

        Returns:
            TensorGroup: A new TensorGroup with the quotient of the two groups.
        """
        return self.operation_with(other, lambda a, b: a/b, on_missing_weights='ignore')

    def __idiv__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Divides one TensorGroup by another inplace.

        Args:
            other (TensorGroup): The other TensorGroup to be divided.

        Returns:
            TensorGroup: A new TensorGroup with the quotient of the two groups.
        """
        return self.operation_with(other, lambda a, b: a/b, on_missing_weights='ignore', inplace=True)

    def __floordiv__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Divides one TensorGroup by another using floor division.

        Args:
            other (TensorGroup): The other TensorGroup to be divided.

        Returns:
            TensorGroup: A new TensorGroup with the floor division of the two groups.
        """
        return self.operation_with(other, lambda a, b: a//b, on_missing_weights='ignore')

    def __ifloordiv__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Divides one TensorGroup by another using floor division inplace.

        Args:
            other (TensorGroup): The other TensorGroup to be divided.

        Returns:
            TensorGroup: A new TensorGroup with the floor division of the two groups.
        """
        return self.operation_with(other, lambda a, b: a//b, on_missing_weights='ignore', inplace=True)

    def __mod__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Computes the modulus of one TensorGroup by another.

        Args:
            other (TensorGroup): The other TensorGroup to be used in the operation.

        Returns:
            TensorGroup: A new TensorGroup with the modulus of the two groups.
        """
        return self.operation_with(other, lambda a, b: a%b, on_missing_weights='ignore')

    def __imod__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Computes the modulus of one TensorGroup by another inplace.

        Args:
            other (TensorGroup): The other TensorGroup to be used in the operation.

        Returns:
            TensorGroup: A new TensorGroup with the modulus of the two groups.
        """
        return self.operation_with(other, lambda a, b: a%b, on_missing_weights='ignore', inplace=True)

    def __pow__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Raises one TensorGroup to the power of another.

        Args:
            other (TensorGroup): The other TensorGroup to be used in the operation.

        Returns:
            TensorGroup: A new TensorGroup with the result of the operation.
        """
        return self.operation_with(other, lambda a, b: a**b, on_missing_weights='ignore')

    def __ipow__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Raises one TensorGroup to the power of another inplace.

        Args:
            other (TensorGroup): The other TensorGroup to be used in the operation.

        Returns:
            TensorGroup: A new TensorGroup with the result of the operation.
        """
        return self.operation_with(other, lambda a, b: a**b, on_missing_weights='ignore', inplace=True)

    def __matmul__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Performs a matrix multiplication of two TensorGroups.

        Args:
            other (TensorGroup): The other TensorGroup to be used in the operation.

        Returns:
            TensorGroup: A new TensorGroup with the result of the operation.
        """
        return self.operation_with(other, lambda a, b: a@b, on_missing_weights='ignore')

    def __imatmul__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Performs a matrix multiplication of two TensorGroups inplace.

        Args:
            other (TensorGroup): The other TensorGroup to be used in the operation.

        Returns:
            TensorGroup: A new TensorGroup with the result of the operation.
        """
        return self.operation_with(other, lambda a, b: a@b, on_missing_weights='ignore', inplace=True)


    def map(self, func: Callable[[np.ndarray], np.ndarray], inplace: bool = False) -> 'TensorGroup':
        """
        Applies a function to each tensor in the group.

        Args:
            func (Callable[[np.ndarray], np.ndarray]): A function that takes a numpy array and returns a numpy array.
            inplace (bool): To perform operation inplace or not

        Returns:
            TensorGroup: A new TensorGroup with the results of the function applied to each tensor.
        """

        if inplace:
            new_data = {
                name: func(weights)
                for name, weights in self.data.items()
            }
            self.data = new_data
            return self
        else:
            return TensorGroup(
                NamedTensor(name, func(weights))
                for name, weights in self.data.items()
            )

    def __neg__(self) -> 'TensorGroup':
        """
        Negates the TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup with the negated tensors.
        """
        return self.map(lambda a: -a)

    def __abs__(self) -> 'TensorGroup':
        """
        Computes the absolute value of the TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup with the absolute values of the tensors.
        """
        return self.map(abs)

    @property
    def T(self) -> 'TensorGroup':
        """
        Transposes the TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup with the transposed tensors.
        """
        return self.map(lambda a: a.T)

    def __contains__(self, item: str) -> bool:
        """
        Check if a neural layer is present in the DirectedTensorGroup.

        Args:
            item (str): The name of the neural layer to check for.

        Returns:
            bool: True if the layer is present, otherwise False.
        """
        return item in self.data


class TensorConnection(TensorGroup):
    """
    Represents a connection of neural weights between layers.

    This class allows you to store and access neural weight data in a dictionary-like manner.

    Attributes:
        data (dict[str, np.ndarray]): A dictionary containing weight data, where keys represent connections
            to layers, and values are NumPy arrays containing the weight values.

    Example:
    ```python
    # Creating a TensorConnection object
    weight_data = {
        'layer1': np.array([[0.5, 0.3], [0.1, 0.8]]),
        'layer2': np.array([[0.2, 0.6], [0.4, 0.7]])
    }
    connection = TensorConnection(data=weight_data)

    # Accessing weight data
    layer1_weights = connection['layer1']
    print(layer1_weights)
    # Output:
    # array([[0.5, 0.3],
    #        [0.1, 0.8]])
    ```

    You can access weight data using square brackets (`[]`) with the connection name as the key.
    """
    def __init__(self, group: 'DirectedTensorGroup', from_: str):
        """
        Initialize a TensorConnection object with weight data.

        Args:
            group_ (DirectedTensorGroup): The DirectedTensorGroup object containing the weight data.
            from_ (str): The name or identifier of the source layer.
        """
        ...
        self.group_ = group
        self.from_ = from_

    @property
    def data(self):
        """
        Returns the dictionary containing the tensor data.
        """
        return self.group_.data[self.from_]

    @data.setter
    def data(self, value):
        """
        Sets the dictionary containing the tensor data.
        """
        self.update(value)

    def __getitem__(self, key: str):
        """
        Retrieve weight data for a specific connection.

        Args:
            key (str): The name or identifier of the population.

        Returns:
            np.ndarray: A NumPy array containing the weight values for the specified connection.
        """
        return self.group_.data[self.from_][key]

    def __setitem__(self, key: str, value: np.ndarray):
        """
        Set weight data for a specific connection.

        Args:
            key (str): The name or identifier of the population.
            value (np.ndarray): A NumPy array containing the weight values for the specified connection.
        """
        self.add_tensor(NamedTensor(key, value))

    def __setattr__(self, key: str, value: np.ndarray):
        """
        Set weight data for a specific connection.

        Args:
            key (str): The name or identifier of the population.
            value (np.ndarray): A NumPy array containing the weight values for the specified connection.
        """
        if key in ('group_', 'from_', 'data'):
            super().__setattr__(key, value)
        else:
            self.add_tensor(NamedTensor(key, value))

    def __getattribute__(self, name: str) -> Any:
        """
        Retrieve an attribute from the TensorConnection object.

        Args:
            name (str): The name of the attribute to retrieve.
        """
        if name in ('group_', 'from_', 'data') or (name.startswith("__") and name.endswith("__")):
            return super().__getattribute__(name)
        elif name in self.group_.data[self.from_].keys():
            return self.group_.data[self.from_][name]
        else: # No, it is not a mistake.
            return super().__getattribute__(name)

    def __iter__(self) -> Generator[str, None, None]:
        """
        Iterate over the connections in the TensorConnection object.

        Yields:
            Generator[str, None, None]: A generator of connection names.
        """
        for key in self.group_.data[self.from_]:
            yield key

    def __repr__(self) -> str:
        """
        Return a string representation of the TensorConnection object.

        Returns:
            str: A string representation of the TensorConnection object.
        """
        keys_and_shapes = [
            f"{key}: ({value.shape})"
            for key, value in self.group_.data[self.from_].items()
        ]
        return f"TensorConnection(from {self.from_}| {', '.join(keys_and_shapes)})"

    def keys(self) -> KeysView[str]:
        """
        Return the keys of the TensorConnection object.

        Returns:
            KeysView[str]: A view of the keys in the TensorConnection object.
        """
        return self.group_.data[self.from_].keys()

    def values(self) -> ValuesView[np.ndarray]:
        """
        Return the values of the TensorConnection object.

        Returns:
            ValuesView[np.ndarray]: A view of the values in the TensorConnection object.
        """
        return self.group_.data[self.from_].values()

    def items(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """
        Return the items of the TensorConnection object.

        Yields:
            Generator[tuple[str, np.ndarray], None, None]: A generator of key-value pairs in the TensorConnection object.
        """
        for key, value in self.group_.data[self.from_].items():
            yield key, value

    def add_tensor(self, tensor: NamedTensor):
        """
        Adds a NamedTensor to the DirectedTensorGroup.

        Args:
            tensor (NamedTensor): The NamedTensor to be added to the group.
        """
        self.group_.add_tensor(DirectedTensor(self.from_, tensor.name, tensor.weights))

    def remove_tensor(self, name: str):
        """
        Removes a TensorConnection from the DirectedTensorGroup by its name.

        Args:
            name (str): The name of the tensor to be removed.

        Raises:
            KeyError: If the tensor with the given name does not exist in the group.
        """
        self.group_.remove_tensor(self.from_, name)


class ConnectionProxy:
    """
    Represents a proxy to access neural weight connections between layers.

    This class provides a convenient way to access neural weight connections through a property.

    Attributes:
        __to (TensorConnection): A private instance of the `TensorConnection` class that
            stores the weight data.

    Example:
    ```python
    # Creating a ConnectionProxy object
    weight_data = {
        'layer1': np.array([[0.5, 0.3], [0.1, 0.8]]),
        'layer2': np.array([[0.2, 0.6], [0.4, 0.7]])
    }
    proxy = ConnectionProxy(data=weight_data)

    # Accessing weight data using the 'to' property
    layer1_weights = proxy.to['layer1']
    print(layer1_weights)
    # Output:
    # array([[0.5, 0.3],
    #        [0.1, 0.8]])

    # You can also directly access the 'to' property to get the TensorConnection object
    connections = proxy.to
    ```
    """
    def __init__(self, group: 'DirectedTensorGroup', from_: str):
        """
        Initialize a ConnectionProxy object with weight data.

        Args:
            group (DirectedTensorGroup): The DirectedTensorGroup object containing the weight data.
            from_ (str): The name or identifier of the source layer.
        """
        self.__to = TensorConnection(group, from_)

    @property
    def to(self):
        """
        Get access to the neural weight connections.

        Returns:
            TensorConnection: An instance of the `TensorConnection` class that stores
            the weight data and allows access to the weight connections.
        """
        return self.__to

    def __getitem__(self, key: str):
        """
        Retrieve weight data for a specific connection.

        Args:
            key (str): The name or identifier of the population.

        Returns:
            np.ndarray: A NumPy array containing the weight values for the specified connection.
        """
        return self.__to[key]

    def __setitem__(self, key: str, value: np.ndarray):
        """
        Set weight data for a specific connection.

        Args:
            key (str): The name or identifier of the population.
            value (np.ndarray): A NumPy array containing the weight values for the specified connection.
        """
        self.__to[key] = value

    def __setattr__(self, key: str, value: Any) -> None:
        """"
        Set weight data for a specific connection.

        Args:
            key (str): The name or identifier of the population.
            value (np.ndarray): A NumPy array containing the weight values for the specified connection.
        """
        if key in ('__to', 'to') or '__to' in key:
            super().__setattr__(key, value)
        else:
            self.__to[key] = value

    def __getattribute__(self, name: str) -> Any:
        """
        Retrieve an attribute from the TensorConnection object.

        Args:
            name (str): The name of the attribute to retrieve.
        """
        if name in ('__to', 'to') or ('__to' in name) or (name.startswith("__") and name.endswith("__")):
            return super().__getattribute__(name)

        return self.__to.__getattribute__(name)

    def __iter__(self) -> Generator[str, None, None]:
        """
        Iterate over the connections in the TensorConnection object.

        Yields:
            Generator[str, None, None]: A generator of connection names.
        """
        return self.__to.__iter__()

    def __repr__(self) -> str:
        """
        Return a string representation of the TensorConnection object.

        Returns:
            str: A string representation of the TensorConnection object.
        """
        return self.__to.__repr__()

    def keys(self) -> KeysView[str]:
        """
        Return the keys of the TensorConnection object.

        Returns:
            KeysView[str]: A view of the keys in the TensorConnection object.
        """
        return self.__to.keys()

    def values(self) -> ValuesView[np.ndarray]:
        """
        Return the values of the TensorConnection object.

        Returns:
            ValuesView[np.ndarray]: A view of the values in the TensorConnection object.
        """
        return self.__to.values()

    def items(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """
        Return the items of the TensorConnection object.

        Yields:
            Generator[tuple[str, np.ndarray], None, None]: A generator of key-value pairs in the TensorConnection object.
        """
        return self.__to.items()

    def __add__(self, other: 'TensorGroup') -> 'TensorConnection':
        """
        Adds two TensorGroups together.

        Args:
            other (TensorGroup): The other TensorGroup to be added.

        Returns:
            TensorGroup: A new TensorGroup with the sum of the two groups.
        """
        return self.__to.__add__(other)

    def __iadd__(self, other: 'TensorGroup') -> 'TensorConnection':
        """
        Adds one TensorGroup to another inplace.

        Args:
            other (TensorGroup): The other TensorGroup to be added.

        Returns:
            TensorGroup: A new TensorGroup with the sum of the two groups.
        """
        return self.__to.__iadd__(other)

    def __sub__(self, other: 'TensorGroup') -> 'TensorConnection':
        """
        Subtracts one TensorGroup from another.

        Args:
            other (TensorGroup): The other TensorGroup to be subtracted.

        Returns:
            TensorGroup: A new TensorGroup with the difference of the two groups.
        """
        return self.__to.__sub__(other)

    def __isub__(self, other: 'TensorGroup') -> 'TensorConnection':
        """
        Subtracts one TensorGroup from another inplace.

        Args:
            other (TensorGroup): The other TensorGroup to be subtracted.

        Returns:
            TensorGroup: A new TensorGroup with the difference of the two groups.
        """
        return self.__to.__isub__(other)


def plot_weighted_graph(weights: 'DirectedTensorGroup', ax: plt.Axes = None, show: bool = True, fig_kwargs: dict = None, **kwargs) -> plt.Figure:
    """
    Plots a directed graph based on the provided connection weights.

    Args:
        weights (DirectedTensorGroup): A DirectedTensorGroup object containing connection weights.
        ax (plt.Axes, optional): The Matplotlib Axes on which the graph will be plotted.
            If not provided, a new subplot will be created. Default is None.
        show (bool, optional): Whether to display the plot. Default is True.
        fig_kwargs (dict, optional): Keyword arguments for creating the Matplotlib Figure.
        kwargs: Keyword arguments for netgraph.Graph object

    Returns:
        plt.Figure: The Matplotlib Figure object representing the generated graph.
    """

    def get_description(data: np.ndarray) -> str:
        if hasattr(data, 'shape'):
            return str(data.shape)
        else:
            return f'Param: {type(data)}'

    adj_matrix = weights.connection_map
    sources, targets = np.where(adj_matrix)
    edges = list(zip(sources, targets))

    if ax is None:
        fig_kwargs = fig_kwargs or dict()
        fig_kwargs.setdefault('figsize', (20, 20))
        _, ax = plt.subplots(**fig_kwargs)

    columns = weights.connection_map.columns.to_list()
    index = weights.connection_map.index.to_list()
    col_diff = [elem for elem in columns + index if elem in columns and elem not in index]
    ind_diff = [elem for elem in columns + index if elem not in columns and elem in index]
    common = [elem for elem in index if elem not in ind_diff]
    order = common + ind_diff + col_diff

    kwargs.setdefault('edge_layout', 'arc')
    kwargs.setdefault('node_layout', 'dot')
    kwargs.setdefault('edge_label_position', 0.66)
    kwargs.setdefault('arrows', True)
    kwargs.setdefault(
        'node_labels',
        dict(zip(
            range(len(order)),
            order
        ))
    )
    kwargs.setdefault(
        'edge_labels',
        {
            (i, j): get_description(weights[f'{ind}->{col}'])
            for i, ind in enumerate(index)
            for j, col in enumerate(columns)
            if weights.connection_map.iloc[i, j]
        }
    )

    Graph(
        edges,
        ax=ax,
        **kwargs
    )

    if show:
        plt.show()

    return ax.figure


class DirectedTensorGroup(AbstractTensorGroup):
    """
    DirectedTensorGroup represents a collection of neural layers and their connections within a neural model.

    Attributes:
        data (dict): A dictionary that stores neural layer connections and their corresponding weights.
        _connection_map (pd.DataFrame): A DataFrame representing the connection map between neural layers.

    Args:
        *layers (list[DirectedTensor]): Variable-length argument list of DirectedTensor instances to initialize the DirectedTensorGroup.

    Methods:
        add_tensor(layer: DirectedTensor):
            Add a neural layer to the DirectedTensorGroup, along with its connections and weights.

        __getitem__(key: Union[tuple[str, str], str]) -> Union[np.ndarray, ConnectionProxy]:
            Retrieve weights associated with a connection between layers using layer names in the format "from->to".
            If a single layer name is provided, returns a ConnectionProxy object representing the connections from that layer.

    Properties:
        connection_map (pd.DataFrame): Read-only property to access the connection map DataFrame.

    Example:
        >>> layer1 = DirectedTensor("input", "hidden", weights_array1)
        >>> layer2 = DirectedTensor("hidden", "output", weights_array2)
        >>> neural_mass = DirectedTensorGroup(layer1, layer2)
        >>> weights = neural_mass["input->hidden"]
        >>> weights = neural_mass["input", "hidden"]
        >>> weights = neural_mass["input"].to["hidden"]
        >>> weights = neural_mass.input.to.hidden
    """
    def __init__(self, *layers: DirectedTensor, **kwargs):
        """
        Initialize a DirectedTensorGroup with one or more neural layers.

        Args:
            *layers (DirectedTensor): Variable-length argument list of DirectedTensor instances.
            **kwargs: DirectedTensorGroup can be created from nested raw dict.
        """
        self._data = dict()
        self._connection_map = None
        self.update(*layers)

        if kwargs is not None:
            self._from_dict(kwargs)

    def _from_dict(self, data: dict[str, dict[str, np.ndarray]]):
        """
        Initializes a DirectedTensorGroup with a dictionary of tensors.

        Args:
            data (dict[str, dict[str, np.ndarray]]): A dictionary mapping neural layer names to their corresponding connections and weights.
        """
        for from_ in data:
            for to in data[from_]:
                self.add_tensor(DirectedTensor(from_, to, data[from_][to]))

    def update(
        self,
        *layers: DirectedTensor,
    ):
        """
        Updates the DirectedTensorGroup with new DirectedTensors.

        Args:
            *layers (DirectedTensor): An unpacked sequence of DirectedTensor objects to be included in the group.
        """
        if len(layers) == 1:
            if isinstance(layers[0], dict):
                self._from_dict(layers[0])
            elif isinstance(layers[0], DirectedTensorGroup):
                for layer in layers[0]:
                    for tensor in layers[0][layer]:
                        self.add_tensor(
                            DirectedTensor(
                                layer, tensor, layers[0][layer][tensor]
                            )
                        )
            else:
                raise ValueError(f"Invalid input. Must be a tuple of DirectedTensor, DirectedTensorGroup or a dictionary, got {type(layers[0])}")
        else:
            for layer in layers:
                self.add_tensor(layer)

    @property
    def data(self):
        """
        Returns the dictionary containing the tensor data.
        """
        return self._data

    @data.setter
    def data(self, value: dict[str, dict[str, np.ndarray]]):
        """
        Sets the dictionary containing the tensor data.

        Args:
            value (dict[str, dict[str, np.ndarray]]): A dictionary containing the tensor data.
        """
        self._data = value

    def __rearrange_connection_map(self):
        """
        Rearranges connection map to make elements with index == column to be diagonal
        """
        columns = set(self._connection_map.columns)
        index = set(self._connection_map.index)

        common = list(columns and index)
        ind_diff = list(index - columns)
        col_diff = list(columns - index)

        new_columns = common + col_diff
        new_index = common + ind_diff

        df = self._connection_map[new_columns]
        self._connection_map = df.reindex(new_index)

    def __update_connection_map(self, from_: str = None, to: str = None):
        """
        Update the connection map with a connection from 'from_' to 'to'.

        Args:
            from_ (str): The source neural layer name.
            to (str): The target neural layer name.
        """
        if self._connection_map is None:
            self._connection_map = pd.DataFrame()

        if from_ not in self._connection_map.columns:
            self._connection_map[from_] = 0

        self._connection_map.loc[from_, to] = 1
        self._connection_map.fillna(0, inplace=True)
        self._connection_map = self.remove_disconnected_nodes(self._connection_map)
        self._connection_map = self._connection_map.astype(int)
        self.__rearrange_connection_map()

    @staticmethod
    def remove_disconnected_nodes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove nodes from a DataFrame if they meet one of the specified conditions indicating disconnection.

        Args:
            df (pd.DataFrame): A DataFrame representing connections between nodes, with nodes as both columns and rows.

        Returns:
            pd.DataFrame: A DataFrame with disconnected nodes removed.
        """
        # Condition 1 & 2: Check for nodes not present in either rows or columns and have all values zero
        rows_to_drop = df.index[(df == 0).all(axis=1) & ~df.index.isin(df.columns)]
        cols_to_drop = df.columns[(df == 0).all(axis=0) & ~df.columns.isin(df.index)]

        # Drop rows and columns found in condition 1 & 2
        df = df.drop(index=rows_to_drop, columns=cols_to_drop, errors='ignore')

        # Condition 3: For nodes present in both, check if both row and column are all zeros
        # This needs to be checked after removing nodes from condition 1 & 2 to avoid missing some cases
        nodes_to_check = df.index.intersection(df.columns)
        for node in nodes_to_check:
            if (df.loc[node] == 0).all() and (df[node] == 0).all():
                df = df.drop(index=node, columns=node, errors='ignore')

        return df

    def add_tensor(self, tensor: DirectedTensor):
        """
        Add a tensor to the DirectedTensorGroup, including its connections and weights.

        Args:
            layer (DirectedTensor): The tensor to be added.
        """
        if tensor.from_ in self.data:
            self.data[tensor.from_][tensor.to] = tensor.weights
        else:
            self.data[tensor.from_] = {tensor.to: tensor.weights}

        self.__setattr__(tensor.from_, ConnectionProxy(self, tensor.from_))

        self.__update_connection_map(tensor.from_, tensor.to)

    def remove_tensor(self, from_: str, to: str = None):
        """
        Remove a neural layer from the DirectedTensorGroup, including its connections and weights.

        Args:
            from_ (str): The name of the neural layer to be removed.
            to (str, optional): The name of the target connection to be removed. If not provided, the entire layer will be removed.
        """
        if to is None:
            del self.data[from_]
            del self.__dict__[from_]
            self._connection_map.drop(from_, axis=0, inplace=True)
            self._connection_map.drop(from_, axis=1, inplace=True)
            self.__rearrange_connection_map()
        else:
            del self.data[from_][to]
            del self.__dict__[from_]
            self.__setattr__(from_, ConnectionProxy(self, from_))
            self._connection_map.loc[from_, to] = 0

    def __setattr__(self, key: str, value: ConnectionProxy):
        """
        Set a ConnectionProxy object for a specific neural layer.

        Args:
            key (str): The name of the neural layer.
            value (ConnectionProxy): The ConnectionProxy object representing the connections from the layer.
        """
        if key in ('_data', 'data', '_connection_map'):
            super().__setattr__(key, value)
        else:
            if isinstance(value, ConnectionProxy):
                super().__setattr__(key, value)
            elif isinstance(value, (TensorConnection, TensorGroup, dict)):
                value = value.copy()
                keys = list(self[key].keys())
                for key_ in keys:
                    self.remove_tensor(key, key_)

                for key_, weights in value.items():
                    self.add_tensor(
                        DirectedTensor(key, key_, weights)
                    )

    def __contains__(self, item: str) -> bool:
        """
        Check if a neural layer is present in the DirectedTensorGroup.

        Args:
            item (str): The name of the neural layer to check for.

        Returns:
            bool: True if the layer is present, otherwise False.
        """
        return item in self.data

    def __getitem__(self, key: tuple[str, str] | str) -> np.ndarray | ConnectionProxy:
        """
        Retrieve weights associated with a connection between layers or a ConnectionProxy object representing
        connections from a specific layer.

        Args:
            key (Union[tuple[str, str], str]): A tuple of two layer names in the format "from->to" or a single
            layer name.

        Returns:
            Union[np.ndarray, ConnectionProxy]: Weights for the specified connection or a ConnectionProxy object.

        Example:
            >>> weights = neural_mass["input->hidden"]
            >>> connection_proxy = neural_mass["input"]
            >>> weights = connection_proxy.to["hidden"]
        """
        if isinstance(key, tuple) and len(key) == 2:
            return self.data[key[0]][key[1]]
        else:
            key_ = [k for k in key.split('->') if k]
            if len(key_) == 2:
                return self.data[key_[0]][key_[1]]
            else:
                return ConnectionProxy(self, key_[0])

    def __setitem__(self, key: tuple[str, str] | str, value: np.ndarray):
        """
        Set weights for a connection between two neural layers.

        Args:
            key (Union[tuple[str, str], str]): A tuple of two layer names in the format "from->to" or a single
            layer name.
        """
        if isinstance(key, tuple) and len(key) == 2:
            self.add_tensor(DirectedTensor(key[0], key[1], value))
        else:
            key_ = [k for k in key.split('->') if k]
            if len(key_) == 2:
                self.add_tensor(DirectedTensor(key_[0], key_[1], value))
            elif isinstance(key, str):
                if isinstance(value, (TensorGroup, TensorConnection, dict)):
                    value = value.copy()
                    keys = list(self[key].keys())
                    for key_ in keys:
                        self.remove_tensor(key, key_)

                    for key_, weights in value.items():
                        self.add_tensor(
                            DirectedTensor(key, key_, weights)
                        )
            else:
                raise ValueError(f"Invalid key format: {key}. Use 'from->to'.")

    @property
    def connection_map(self):
        """
        Read-only property to access the connection map DataFrame.

        Returns:
            pd.DataFrame: The DataFrame representing the connection map between neural layers.
        """
        return self._connection_map

    def plot(self, ax: plt.Axes = None, show: bool = True, fig_kwargs: dict = None, **kwargs):
        """
        Plots a directed graph based on the current DirectedTensorGroup object.

        Args:
            ax (plt.Axes, optional): The Matplotlib Axes on which the graph will be plotted.
                If not provided, a new subplot will be created. Default is None.
            show (bool, optional): Whether to display the plot. Default is True.
            fig_kwargs (dict, optional): Keyword arguments for the Matplotlib Figure object.
            kwargs: Keyword arguments for netgraph.Graph object.

        Returns:
            plt.Figure: The Matplotlib Figure object representing the generated graph.
        """
        return plot_weighted_graph(self, ax, show, fig_kwargs, **kwargs)

    def __iter__(self) -> Generator[str, None, None]:
        """
        Iterate over the keys within the DirectedTensorGroup.

        Yields:
            str: A group name of DirectedTensorGroup.
        """
        return iter(self.data)

    def __repr__(self) -> str:
        """
        Returns a string representation of the DirectedTensorGroup.

        Returns:
            str: A string representation of the DirectedTensorGroup.
        """
        populations_info = list()
        for key1, value in self.data.items():
            for key2, data in value.items():
                populations_info.append(f"{key1}->{key2}: {data.shape}")

        return f"DirectedTensorGroup({', '.join(populations_info)})"

    def keys(self) -> KeysView[str]:
        """
        Returns the keys of the DirectedTensorGroup.

        Returns:
            KeysView[str]: A view of the keys in the DirectedTensorGroup.
        """
        return self.data.keys()

    def values(self) -> list[TensorConnection]:
        """
        Returns the values of the DirectedTensorGroup.

        Returns:
            list[TensorConnection]: A list of TensorConnection objects in the DirectedTensorGroup.
        """
        return [TensorConnection(self, key) for key in self.data.keys()]

    def items(self) -> Generator[tuple[str, TensorConnection], None, None]:
        """
        Returns the items of the DirectedTensorGroup.

        Yields:
            Generator[tuple[str, TensorConnection], None, None]: A generator of key-TensorConnection pairs in the DirectedTensorGroup.
        """
        for key in self.data:
            yield key, TensorConnection(self, key)

    def operation_with(
        self,
        other: 'DirectedTensorGroup',
        operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
        on_missing_weights: Literal['raise', 'ignore', 'concat'] | Callable[[np.ndarray], np.ndarray] = 'raise',
        on_missing_sources: Literal['raise', 'ignore', 'concat'] | Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] = 'raise',
        inplace: bool = False
    ) -> 'DirectedTensorGroup':
        """
        Apply an operation between two DirectedTensorGroup instances.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to apply the operation to.
            operation (callable[[np.ndarray, np.ndarray], np.ndarray]): A function that takes two NumPy arrays
                and returns a NumPy array.
            on_missing_weights (Literal['raise', 'ignore', 'concat'] | Callable[[np.ndarray], np.ndarray], optional): What to do when a connection is missing weights.
            on_missing_sources (Literal['raise', 'ignore', 'concat'] | Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]], optional): What to do when a source is missing connections.
            inplace (bool): To perform operation inplace or not
        """
        new_data = deepcopy(self.data)

        for source, targetdict in other.data.items():
            if source in new_data:
                for target, weights in targetdict.items():
                    if target in new_data[source]:
                        new_data[source][target] = operation(new_data[source][target], weights)
                    else:
                        if on_missing_weights == 'raise':
                            raise KeyError(f"Connection from {source} to {target} does not exist in the current DirectedTensorGroup.")
                        elif on_missing_weights == 'ignore':
                            continue
                        elif on_missing_weights == 'concat':
                            new_data[source][target] = weights
                        elif isinstance(on_missing_weights, Callable):
                            new_data[source][target] = on_missing_weights(weights)
                        else:
                            raise ValueError(f"Invalid value for 'on_missing_sources': {on_missing_weights}")
            else:
                if on_missing_sources == 'raise':
                    raise KeyError(f"Source {source} does not exist in the current DirectedTensorGroup.")
                elif on_missing_sources == 'ignore':
                    continue
                elif on_missing_sources == 'concat':
                    new_data[source] = targetdict
                elif isinstance(on_missing_sources, Callable):
                    new_data[source] = on_missing_sources(targetdict)
                else:
                    raise ValueError(f"Invalid value for 'on_missing_sources': {on_missing_sources}")

        if not inplace:
            return DirectedTensorGroup(*list(dict2directed_tensor(new_data)))
        else:
            self.data = new_data
            return self

    def __add__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Add the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be added to the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a + b, on_missing_weights='concat', on_missing_sources='concat')

    def __iadd__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Add the connection information from another DirectedTensorGroup to the current DirectedTensorGroup inplace.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be added to the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a + b, on_missing_weights='concat', on_missing_sources='concat', inplace=True)

    def __sub__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Subtract the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be subtracted from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a - b, on_missing_weights='ignore', on_missing_sources='ignore')

    def __isub__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Subtract the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be subtracted from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a - b, on_missing_weights='ignore', on_missing_sources='ignore', inplace=True)

    def __mul__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Multiply the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be multiplied to the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a*b, on_missing_weights='ignore', on_missing_sources='ignore')

    def __imul__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Multiply the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be multiplied to the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a*b, on_missing_weights='ignore', on_missing_sources='ignore', inplace=True)

    def __div__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Divide the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be divided from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a/b, on_missing_weights='ignore', on_missing_sources='ignore')

    def __idiv__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Divide the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be divided from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a/b, on_missing_weights='ignore', on_missing_sources='ignore', inplace=True)

    def __floordiv__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Floor divide the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be floor divided from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a//b, on_missing_weights='ignore', on_missing_sources='ignore')

    def __ifloordiv__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Floor divide the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be floor divided from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a//b, on_missing_weights='ignore', on_missing_sources='ignore', inplace=True)

    def __mod__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Mod the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be modded from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a%b, on_missing_weights='ignore', on_missing_sources='ignore')

    def __imod__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Mod the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be modded from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a%b, on_missing_weights='ignore', on_missing_sources='ignore', inplace=True)

    def __pow__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Raise the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be raised from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a**b, on_missing_weights='ignore', on_missing_sources='ignore')

    def __ipow__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Raise the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be raised from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a**b, on_missing_weights='ignore', on_missing_sources='ignore', inplace=True)

    def __matmul__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Matrix multiply the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be matrix multiplied from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a@b, on_missing_weights='ignore', on_missing_sources='ignore')

    def __imatmul__(self, other: 'DirectedTensorGroup') -> 'DirectedTensorGroup':
        """
        Matrix multiply the connection information from another DirectedTensorGroup to the current DirectedTensorGroup.

        Args:
            other (DirectedTensorGroup): Another DirectedTensorGroup instance to be matrix multiplied from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a@b, on_missing_weights='ignore', on_missing_sources='ignore', inplace=True)

    def map(self, func: Callable[[np.ndarray], np.ndarray], inplace: bool = False) -> 'DirectedTensorGroup':
        """
        Applies a function to the current DirectedTensorGroup weights.

        Args:
            func (callable[[np.ndarray], np.ndarray]): A function that takes a NumPy array and returns a NumPy array.
            inplace (bool): To perform operation inplace or not.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with negated connection information.
        """
        if inplace:
            new_data = {
                source: {
                    target: func(weights)
                } for source, targetdict in self.data.items() for target, weights in targetdict.items()
            }
            self.data = new_data
            return self
        else:
            return DirectedTensorGroup(*list(dict2directed_tensor({source: {target: func(weights) for target, weights in targetdict.items()} for source, targetdict in self.data.items()})))

    def __neg__(self) -> 'DirectedTensorGroup':
        """
        Negate the connection information from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with negated connection information.
        """
        return self.map(lambda x: -x)

    def __abs__(self) -> 'DirectedTensorGroup':
        """
        Take the absolute value of the connection information from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with absolute value of connection information.
        """
        return self.map(abs)

    @property
    def T(self) -> 'DirectedTensorGroup':
        """
        Transpose the connection information from the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with absolute value of connection information.
        """
        return self.map(lambda x: x.T)

    def copy(self) -> 'DirectedTensorGroup':
        """
        Create a deep copy of the current DirectedTensorGroup.

        Returns:
            DirectedTensorGroup: A new DirectedTensorGroup instance with the same connection information.
        """
        return DirectedTensorGroup(*list(dict2directed_tensor(deepcopy(self.data))))
