from dataclasses import dataclass
import numpy as np
import pandas as pd

from utils.data import WritablePickle, read_pkl, save_pkl


@dataclass
class NeuralWeights:
    """
    Represents the weights between two layers in a neural network.

    Attributes:
        from_ (str): The name or identifier of the source layer.
        to (str): The name or identifier of the target layer.
        weights (np.ndarray): A NumPy array containing the weights connecting the source and target layers.

    Example:
    ```python
    # Creating a NeuralWeights object
    weights = NeuralWeights(from_='hidden_layer', to='output_layer', weights=np.array([[0.5, 0.3], [0.1, 0.8]]))

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


class NeuralWeightsConnnection:
    """
    Represents a connection of neural weights between layers.

    This class allows you to store and access neural weight data in a dictionary-like manner.

    Attributes:
        data (dict[str, np.ndarray]): A dictionary containing weight data, where keys represent connections
            to layers, and values are NumPy arrays containing the weight values.

    Example:
    ```python
    # Creating a NeuralWeightsConnection object
    weight_data = {
        'layer1': np.array([[0.5, 0.3], [0.1, 0.8]]),
        'layer2': np.array([[0.2, 0.6], [0.4, 0.7]])
    }
    connection = NeuralWeightsConnection(data=weight_data)

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
        Initialize a NeuralWeightsConnection object with weight data.

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
        __to (NeuralWeightsConnection): A private instance of the `NeuralWeightsConnection` class that
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

    # You can also directly access the 'to' property to get the NeuralWeightsConnection object
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
        self.__to = NeuralWeightsConnnection(data)

    @property
    def to(self):
        """
        Get access to the neural weight connections.

        Returns:
            NeuralWeightsConnection: An instance of the `NeuralWeightsConnection` class that stores
            the weight data and allows access to the weight connections.
        """
        return self.__to


class NeuralMass(WritablePickle):
    """
    NeuralMass represents a collection of neural layers and their connections within a neural model.

    Attributes:
        data (dict): A dictionary that stores neural layer connections and their corresponding weights.
        _connection_map (pd.DataFrame): A DataFrame representing the connection map between neural layers.

    Args:
        *layers (list[NeuralWeights]): Variable-length argument list of NeuralWeights instances to initialize the NeuralMass.

    Methods:
        add_layer(layer: NeuralWeights):
            Add a neural layer to the NeuralMass, along with its connections and weights.

        __getitem__(key: Union[tuple[str, str], str]) -> Union[np.ndarray, ConnectionProxy]:
            Retrieve weights associated with a connection between layers using layer names in the format "from->to".
            If a single layer name is provided, returns a ConnectionProxy object representing the connections from that layer.

    Properties:
        connection_map (pd.DataFrame): Read-only property to access the connection map DataFrame.

    Example:
        >>> layer1 = NeuralWeights("input", "hidden", weights_array1)
        >>> layer2 = NeuralWeights("hidden", "output", weights_array2)
        >>> neural_mass = NeuralMass(layer1, layer2)
        >>> weights = neural_mass["input->hidden"]
        >>> weights = neural_mass["input", "hidden"]
        >>> weights = neural_mass["input"].to["hidden"]
        >>> weights = neural_mass.input.to.hidden
    """
    def __init__(self, *layers: list[NeuralWeights]):
        """
        Initialize a NeuralMass with one or more neural layers.

        Args:
            *layers (list[NeuralWeights]): Variable-length argument list of NeuralWeights instances.
        """
        self.data = dict()
        self._connection_map = None
        if len(layers):
            for layer in layers:
                self.add_layer(layer)

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

    def add_layer(self, layer: NeuralWeights):
        """
        Add a neural layer to the NeuralMass, including its connections and weights.

        Args:
            layer (NeuralWeights): The neural layer to be added.
        """
        if layer.from_ in self.data:
            self.data[layer.from_][layer.to] = layer.weights
        else:
            self.data[layer.from_] = {layer.to: layer.weights}

        self.__setattr__(layer.from_, ConnectionProxy(self.data[layer.from_]))

        self.__update_connection_map(layer.from_, layer.to)

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
            key_ = key.split('->')
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
