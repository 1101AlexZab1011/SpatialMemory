from dataclasses import dataclass
from typing import Literal
from bbtoolkit.data import Copyable


@dataclass
class EnvironmentMetaData(Copyable):
    """
    A data class representing metadata of an environment.

    Attributes:
        type (Literal['object', 'wall']): The type of the entity.
        vp_slice (slice): The slice of the visible plane.
        vec_slice (slice): The slice of the boundaries in the array of all boundary points.
    """
    type: Literal['object', 'wall']
    vp_slice: slice
    vec_slice: slice