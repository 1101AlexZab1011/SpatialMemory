from dataclasses import dataclass
from typing import Sequence

import numpy as np
from bbtoolkit.data import Copyable
from bbtoolkit.structures import Proxy
from shapely import Polygon


@dataclass
class Texture(Copyable):
    """
    A data class to represent a Texture.

    Attributes:
        id_ (int): The unique identifier for the texture. Default is -1.
        color (str): The color of the texture. Default is None.
        name (str): The name of the texture. Default is None.
    """
    id_: int = -1
    color: str = None
    name: str = None

    def __post_init__(self):
        """
        Post-initialization method to validate the id_ attribute.

        Raises:
            ValueError: if id_ is 0.
        """
        if self.id_ == 0:
            raise ValueError('Texture ID cannot be 0.')



class TexturedPolygon(Proxy):
    """
    A class representing a textured polygon, inheriting from the Proxy class.

    Attributes:
        shell (Sequence | Polygon): The outer boundary of the polygon or a Polygon instance to transform into TexturedPolygon.
        holes (Sequence[Sequence], optional): The holes within the polygon. Default is None.
        texture (Texture, optional): The texture of the polygon. Default is None.
    """
    def __init__(self, shell: Sequence | Polygon, holes: Sequence[Sequence] = None, *, texture: Texture = None):
        if not isinstance(shell, Polygon):
            polygon = Polygon(shell, holes)
        elif isinstance(shell, Polygon):
            if holes is not None:
                raise ValueError('Cannot specify holes for a Polygon object.')
            polygon = shell
        else:
            raise ValueError(f'Invalid shell type: {type(shell)}')

        texture = texture if texture is not None else Texture()
        super().__init__(polygon, texture=texture)

    def __copy__(self):
        return TexturedPolygon(
            Polygon(self.exterior, self.interiors),
            texture=self.texture
        )

    def __deepcopy__(self, memo: dict):
        result = TexturedPolygon(
            Polygon(self.exterior, self.interiors),
            texture=self.texture
        )
        memo[id(self)] = result
        return result


@dataclass
class Coordinates2D:
    """
    A data class for storing x and y coordinates as numpy arrays.

    Attributes:
        x (int | float | Sequence): Data representing x-coordinates.
        y (int | float | Sequence): Data representing y-coordinates.

    Raises:
        ValueError: If the shapes of x and y arrays do not match during object initialization.

    Example:
        >>> coordinates = Coordinates(
        >>>     x=np.array([1.0, 2.0, 3.0]),
        >>>     y=np.array([4.0, 5.0, 6.0])
        >>> )

        You can access the x and y coordinates using `coordinates.x` and `coordinates.y` respectively.
    """
    x: int | float | Sequence
    y: int | float | Sequence

    def __post_init__(self):
        """
        Ensure that x and y arrays have the same type (and shape) after object initialization.
        """
        if type(self.x) != type(self.y):
            raise ValueError(f'x and y must have the same type, got {type(self.x)} and {type(self.y)} instead')
        if hasattr(self.x, 'shape') and (self.x.shape != self.y.shape):
            raise ValueError(f'x and y must have the same shape, got {self.x.shape} and {self.y.shape} instead')
