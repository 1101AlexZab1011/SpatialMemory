from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Generator, Literal
from matplotlib import pyplot as plt
from netgraph import Graph
import numpy as np
import pandas as pd

from bbtoolkit.data import WritablePickle


@dataclass
class DirectedTensor:
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

    from_: str
    to: str
    weights: np.ndarray


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


def plot_weighted_graph(weights: 'TensorGroup', ax: plt.Axes = None, show: bool = True, **kwargs) -> plt.Figure:
    """
    Plots a directed graph based on the provided connection weights.

    Args:
        weights (TensorGroup): A TensorGroup object containing connection weights.
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

class TensorGroup(WritablePickle):
    """
    TensorGroup represents a collection of neural layers and their connections within a neural model.

    Attributes:
        data (dict): A dictionary that stores neural layer connections and their corresponding weights.
        _connection_map (pd.DataFrame): A DataFrame representing the connection map between neural layers.

    Args:
        *layers (list[DirectedTensor]): Variable-length argument list of DirectedTensor instances to initialize the TensorGroup.

    Methods:
        add_layer(layer: DirectedTensor):
            Add a neural layer to the TensorGroup, along with its connections and weights.

        __getitem__(key: Union[tuple[str, str], str]) -> Union[np.ndarray, ConnectionProxy]:
            Retrieve weights associated with a connection between layers using layer names in the format "from->to".
            If a single layer name is provided, returns a ConnectionProxy object representing the connections from that layer.

    Properties:
        connection_map (pd.DataFrame): Read-only property to access the connection map DataFrame.

    Example:
        >>> layer1 = DirectedTensor("input", "hidden", weights_array1)
        >>> layer2 = DirectedTensor("hidden", "output", weights_array2)
        >>> neural_mass = TensorGroup(layer1, layer2)
        >>> weights = neural_mass["input->hidden"]
        >>> weights = neural_mass["input", "hidden"]
        >>> weights = neural_mass["input"].to["hidden"]
        >>> weights = neural_mass.input.to.hidden
    """
    def __init__(self, *layers: list[DirectedTensor]):
        """
        Initialize a TensorGroup with one or more neural layers.

        Args:
            *layers (list[DirectedTensor]): Variable-length argument list of DirectedTensor instances.
        """
        self.data = dict()
        self._connection_map = None
        if len(layers):
            for layer in layers:
                self.add_layer(layer)

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

    def add_layer(self, layer: DirectedTensor):
        """
        Add a neural layer to the TensorGroup, including its connections and weights.

        Args:
            layer (DirectedTensor): The neural layer to be added.
        """
        if layer.from_ in self.data:
            self.data[layer.from_][layer.to] = layer.weights
        else:
            self.data[layer.from_] = {layer.to: layer.weights}

        self.__setattr__(layer.from_, ConnectionProxy(self.data[layer.from_]))

        self.__update_connection_map(layer.from_, layer.to)

    def remove_layer(self, from_: str, to: str = None):
        """
        Remove a neural layer from the TensorGroup, including its connections and weights.

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

    def __getitem__(self, key: tuple[str, str] | str):
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
        Plots a directed graph based on the current TensorGroup object.

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
        Iterate over the DirectedTensor instances within the TensorGroup.

        Yields:
            DirectedTensor: A DirectedTensor instance.
        """
        return dict2directed_tensor(self.data)

    def operation_with(
        self,
        other: 'TensorGroup',
        operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
        on_missing_weights: Literal['raise', 'ignore', 'concat'] | Callable[[np.ndarray], np.ndarray] = 'raise',
        on_missing_sources: Literal['raise', 'ignore', 'concat'] | Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] = 'raise'
    ) -> 'TensorGroup':
        """
        Apply an operation between two TensorGroup instances.

        Args:
            other (TensorGroup): Another TensorGroup instance to apply the operation to.
            operation (callable[[np.ndarray, np.ndarray], np.ndarray]): A function that takes two NumPy arrays
                and returns a NumPy array.
            on_missing_weights (Literal['raise', 'ignore', 'concat'] | Callable[[np.ndarray], np.ndarray], optional): What to do when a connection is missing weights.
            on_missing_sources (Literal['raise', 'ignore', 'concat'] | Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]], optional): What to do when a source is missing connections.
        """
        new_data = deepcopy(self.data)

        for source, targetdict in other.data.items():
            if source in new_data:
                for target, weights in targetdict.items():
                    if target in new_data[source]:
                        new_data[source][target] = operation(new_data[source][target], weights)
                    else:
                        if on_missing_weights == 'raise':
                            raise KeyError(f"Connection from {source} to {target} does not exist in the current TensorGroup.")
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
                    raise KeyError(f"Source {source} does not exist in the current TensorGroup.")
                elif on_missing_sources == 'ignore':
                    continue
                elif on_missing_sources == 'concat':
                    new_data[source] = targetdict
                elif isinstance(on_missing_sources, Callable):
                    new_data[source] = on_missing_sources(targetdict)
                else:
                    raise ValueError(f"Invalid value for 'on_missing_sources': {on_missing_sources}")

        return TensorGroup(*list(dict2directed_tensor(new_data)))

    def __add__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Add the connection information from another TensorGroup to the current TensorGroup.

        Args:
            other (TensorGroup): Another TensorGroup instance to be added to the current TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a + b, on_missing_weights='concat', on_missing_sources='concat')

    def __sub__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Subtract the connection information from another TensorGroup to the current TensorGroup.

        Args:
            other (TensorGroup): Another TensorGroup instance to be subtracted from the current TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a - b, on_missing_weights='ignore', on_missing_sources='ignore')

    def __mul__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Multiply the connection information from another TensorGroup to the current TensorGroup.

        Args:
            other (TensorGroup): Another TensorGroup instance to be multiplied to the current TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a*b, on_missing_weights='ignore', on_missing_sources='ignore')

    def __div__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Divide the connection information from another TensorGroup to the current TensorGroup.

        Args:
            other (TensorGroup): Another TensorGroup instance to be divided from the current TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a/b, on_missing_weights='ignore', on_missing_sources='ignore')

    def __floordiv__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Floor divide the connection information from another TensorGroup to the current TensorGroup.

        Args:
            other (TensorGroup): Another TensorGroup instance to be floor divided from the current TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a//b, on_missing_weights='ignore', on_missing_sources='ignore')

    def __mod__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Mod the connection information from another TensorGroup to the current TensorGroup.

        Args:
            other (TensorGroup): Another TensorGroup instance to be modded from the current TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a%b, on_missing_weights='ignore', on_missing_sources='ignore')

    def __pow__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Raise the connection information from another TensorGroup to the current TensorGroup.

        Args:
            other (TensorGroup): Another TensorGroup instance to be raised from the current TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a**b, on_missing_weights='ignore', on_missing_sources='ignore')

    def __matmul__(self, other: 'TensorGroup') -> 'TensorGroup':
        """
        Matrix multiply the connection information from another TensorGroup to the current TensorGroup.

        Args:
            other (TensorGroup): Another TensorGroup instance to be matrix multiplied from the current TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup instance with combined connection information.
        """
        return self.operation_with(other, lambda a, b: a@b, on_missing_weights='ignore', on_missing_sources='ignore')

    def map(self, func: Callable[[np.ndarray], np.ndarray]) -> 'TensorGroup':
        """
        Applies a function to the current TensorGroup weights.

        Args:
            func (callable[[np.ndarray], np.ndarray]): A function that takes a NumPy array and returns a NumPy array.

        Returns:
            TensorGroup: A new TensorGroup instance with negated connection information.
        """
        return TensorGroup(*list(dict2directed_tensor({source: {target: func(weights) for target, weights in targetdict.items()} for source, targetdict in self.data.items()})))

    def __neg__(self) -> 'TensorGroup':
        """
        Negate the connection information from the current TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup instance with negated connection information.
        """
        return self.map(lambda x: -x)

    def __abs__(self) -> 'TensorGroup':
        """
        Take the absolute value of the connection information from the current TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup instance with absolute value of connection information.
        """
        return self.map(abs)

    @property
    def T(self) -> 'TensorGroup':
        """
        Transpose the connection information from the current TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup instance with absolute value of connection information.
        """
        return self.map(lambda x: x.T)

    def copy(self) -> 'TensorGroup':
        """
        Create a deep copy of the current TensorGroup.

        Returns:
            TensorGroup: A new TensorGroup instance with the same connection information.
        """
        return TensorGroup(*list(dict2directed_tensor(deepcopy(self.data))))
