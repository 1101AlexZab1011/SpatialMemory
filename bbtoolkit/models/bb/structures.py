from dataclasses import dataclass, field
from typing import Literal

from bbtoolkit.environment import Environment
from bbtoolkit.models.bb.neural_generators import TCGenerator
from bbtoolkit.models.bb.utils import Grid2CartTransition
from bbtoolkit.structures import DotDict
from bbtoolkit.structures.tensorgroups import DirectedTensorGroup, TensorGroup


@dataclass
class DynamicParameters(DotDict):
    """
    A data class to hold dynamic parameters for the simulation, extending DotDict for dot-accessible dictionary attributes.

    Attributes:
        dt (float): The time step for the simulation.
        mode (Literal['bottom-up', 'top-down', 'recall']): The current mode of the simulation.
        step (int): The current step number in the simulation, defaulting to 0.
    """
    dt: float
    mode: Literal['bottom-up', 'top-down', 'recall']
    step: int = field(default_factory=lambda:0)


@dataclass
class EcodingParameters(DotDict):
    """
    A data class to hold encoding parameters for the simulation, including information about encoded objects and the object currently being recalled.

    Attributes:
        encoded_objects (DirectedTensorGroup): A group of tensors representing encoded objects, defaulting to None.
        object_to_recall (int): The identifier of the object to recall, defaulting to None.
    """
    encoded_objects: DirectedTensorGroup = field(default_factory=lambda:None)
    object_to_recall: int = field(default_factory=lambda:None)


@dataclass
class ClickParameters(DotDict):
    """
    A data class to hold parameters related to mouse click interactions within the simulation.

    Attributes:
        xy_data (tuple[float, float]): The x and y coordinates of the click, defaulting to None.
        inside_object (bool): A flag indicating whether the click was inside an object, defaulting to False.
        inside_wall (bool): A flag indicating whether the click was inside a wall, defaulting to False.
    """
    xy_data: tuple[float, float] = field(default_factory=lambda:None)
    inside_object: bool = field(default_factory=bool)
    inside_wall: bool = field(default_factory=bool)


@dataclass
class BBCache(DotDict):
    """
    A data class to hold the simulation cache for a brain-based controller (BBC), extending DotDict for dot-accessible dictionary attributes.

    Attributes:
        connectivity (DirectedTensorGroup): The connectivity information between different parts of the simulation.
        weights (DirectedTensorGroup): The synaptic weights between neurons or groups of neurons.
        k_ratio (TensorGroup): The ratio of different types of connections or activities.
        activity (TensorGroup): The neural activity within the simulation.
        rates (TensorGroup): The firing rates of neurons or groups of neurons.
        tc_gen (TCGenerator): The transformation circuit generator for the simulation.
        env (Environment): The environment in which the agent operates.
        grid2cart (Grid2CartTransition): The transition matrix or function from grid to Cartesian coordinates.
        dynamics_params (DynamicParameters): The dynamic parameters controlling the simulation's state and progression.
        encoding_params (EcodingParameters): The encoding parameters related to object recognition and memory, defaulting to an empty instance.
        click_params (ClickParameters): The parameters related to user interactions with the simulation, defaultally to an empty instance.
    """
    connectivity: DirectedTensorGroup
    weights: DirectedTensorGroup
    k_ratio: TensorGroup
    activity: TensorGroup
    rates: TensorGroup
    tc_gen: TCGenerator
    env: Environment
    grid2cart: Grid2CartTransition
    dynamics_params: DynamicParameters
    encoding_params: EcodingParameters = field(default_factory=lambda:EcodingParameters())
    click_params: ClickParameters = field(default_factory=lambda:ClickParameters())
