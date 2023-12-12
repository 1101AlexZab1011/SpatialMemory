from dataclasses import dataclass
from typing import Sequence
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