from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Generator, Literal
from matplotlib import pyplot as plt
from netgraph import Graph
import numpy as np
import pandas as pd

from bbtoolkit.utils.data import Copyable, WritablePickle



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


class TensorConnection:
    """
    Represents a connection of neural weights between layers.

    This class allows you to store and access neural weight data in a dictionary-like manner.

    Attributes:
        data (dict[str, np.ndarray]): A dictionary containing weight data, where keys represent connections
            to layers, and values are NumPy arrays containing the weight values.

    Example:
    ```python
    # Creating a DirectedTensorConnection object
    weight_data = {
        'layer1': np.array([[0.5, 0.3], [0.1, 0.8]]),
        'layer2': np.array([[0.2, 0.6], [0.4, 0.7]])
    }
    connection = DirectedTensorConnection(data=weight_data)

    # Accessing weight data
    layer1_weights = connection['layer1']
    print(layer1_weights)
    # Output:
    # array([[0.5, 0.3],
    #        [0.1, 0.8]])
    ```

    You can access weight data using square brackets (`[]`) with the connection name as the key.
    """
    def __init__(self, data: dict[str, np.ndarray]):
        """
        Initialize a DirectedTensorConnection object with weight data.

        Args:
            data (dict[str, np.ndarray]): A dictionary containing weight data representing connection from the current pipulation to others.
                Keys represent connected layers, and values are NumPy arrays
                containing the weight values.
        """
        self.data = data
        for key in data:
            self.__setattr__(key, data[key])
    def __getitem__(self, key: str):
        """
        Retrieve weight data for a specific connection.

        Args:
            key (str): The name or identifier of the population.

        Returns:
            np.ndarray: A NumPy array containing the weight values for the specified connection.
        """
        return self.data[key]


class ConnectionProxy:
    """
    Represents a proxy to access neural weight connections between layers.

    This class provides a convenient way to access neural weight connections through a property.

    Attributes:
        __to (DirectedTensorConnection): A private instance of the `DirectedTensorConnection` class that
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

    # You can also directly access the 'to' property to get the DirectedTensorConnection object
    connections = proxy.to
    ```
    """
    def __init__(self, data: dict[str, np.ndarray]):
        """
        Initialize a ConnectionProxy object with weight data.

        Args:
            data (dict[str, np.ndarray]): A dictionary containing weight data.
                Keys represent connected layers, and values are NumPy arrays
                containing the weight values.
        """
        self.__to = TensorConnection(data)

    @property
    def to(self):
        """
        Get access to the neural weight connections.

        Returns:
            DirectedTensorConnection: An instance of the `DirectedTensorConnection` class that stores
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


def plot_weighted_graph(weights: 'DirectedTensorGroup', ax: plt.Axes = None, show: bool = True, **kwargs) -> plt.Figure:
    """
    Plots a directed graph based on the provided connection weights.

    Args:
        weights (DirectedTensorGroup): A DirectedTensorGroup object containing connection weights.
        ax (plt.Axes, optional): The Matplotlib Axes on which the graph will be plotted.
            If not provided, a new subplot will be created. Default is None.
        show (bool, optional): Whether to display the plot. Default is True.
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
        fig, ax = plt.subplots(figsize=(20, 20))

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
    def __init__(self, *tensors: NamedTensor):
        """
        Initializes a TensorGroup with a sequence of NamedTensors.

        Args:
            *tensors (NamedTensor): An unpacked sequence of NamedTensor objects to be included in the group.
        """
        self._data = dict()
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

    def __iter__(self) -> Generator[NamedTensor, None, None]:
        """
        A generator that yields NamedTensor objects from the TensorGroup.

        Yields:
            NamedTensor: The next NamedTensor in the group.
        """
        for tensor in self.data:
            yield NamedTensor(tensor, self.data[tensor])

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
    def __init__(self, *layers: list[DirectedTensor]):
        """
        Initialize a DirectedTensorGroup with one or more neural layers.

        Args:
            *layers (list[DirectedTensor]): Variable-length argument list of DirectedTensor instances.
        """
        self._data = dict()
        self._connection_map = None
        if len(layers):
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
        self._connection_map = self._connection_map.astype(int)
        self.__rearrange_connection_map()

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

        self.__setattr__(tensor.from_, ConnectionProxy(self.data[tensor.from_]))

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
            self.__setattr__(from_, ConnectionProxy(self.data[from_]))
            self._connection_map.loc[from_, to] = 0

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
                return ConnectionProxy(self.data[key_[0]])
    @property
    def connection_map(self):
        """
        Read-only property to access the connection map DataFrame.

        Returns:
            pd.DataFrame: The DataFrame representing the connection map between neural layers.
        """
        return self._connection_map

    def plot(self, ax: plt.Axes = None, show: bool = True, **kwargs):
        """
        Plots a directed graph based on the current DirectedTensorGroup object.

        Args:
            ax (plt.Axes, optional): The Matplotlib Axes on which the graph will be plotted.
                If not provided, a new subplot will be created. Default is None.
            show (bool, optional): Whether to display the plot. Default is True.
            kwargs: Keyword arguments for netgraph.Graph object.

        Returns:
            plt.Figure: The Matplotlib Figure object representing the generated graph.
        """
        return plot_weighted_graph(self, ax, show, **kwargs)

    def __iter__(self) -> Generator[DirectedTensor, None, None]:
        """
        Iterate over the DirectedTensor instances within the DirectedTensorGroup.

        Yields:
            DirectedTensor: A DirectedTensor instance.
        """
        return dict2directed_tensor(self.data)

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
