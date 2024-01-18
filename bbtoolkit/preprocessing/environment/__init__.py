from dataclasses import dataclass
import numpy as np
from bbtoolkit.data import Copyable, WritablePickle
from bbtoolkit.structures.geometry import TexturedPolygon
from shapely import Polygon


class Area(Copyable):
    """
    A class representing an area defined by a polygon and a set of points.

    Attributes:
        polygon (Polygon): The polygon defining the area.
        points (np.ndarray): The set of points within the area.
        vectors (tuple[np.ndarray, ...]): The space vectors of the area (axes).
    """
    def __init__(
        self,
        polygon: Polygon,
        points: np.ndarray = None,
    ):
        """
        Initializes the Area with a polygon and an optional set of points.

        Args:
            polygon (Polygon): The polygon defining the area.
            points (np.ndarray, optional): The set of points within the area. Default is None.
        """
        self._polygon = polygon
        self._points = points

    @property
    def polygon(self) -> TexturedPolygon:
        """
        Returns the polygon defining the area.

        Returns:
            Polygon: The polygon defining the area.
        """
        return self._polygon

    @property
    def points(self) -> np.ndarray:
        """
        Returns the set of points within the area.

        Returns:
            np.ndarray: The set of points within the area.
        """
        return self._points


class Object(Area):
    """
    A class representing a boundary, which is a specific type of area with visible parts.

    Attributes:
        visible_parts (np.ndarray): The visible parts of the boundary.
    """
    def __init__(
        self,
        polygon: TexturedPolygon,
        points: np.ndarray,
        visible_parts: np.ndarray,
        starting_points: np.ndarray,
        directions: np.ndarray
    ):
        """
        Initializes the Object with a polygon, a set of points, and visible parts.

        Args:
            polygon (TexturedPolygon): The polygon defining the boundary.
            points (np.ndarray): The set of points within the boundary.
            visible_parts (np.ndarray): The visible parts of the boundary.
            starting_points (np.ndarray): The starting points of the boundary.
            directions (np.ndarray): The directions of the boundary (vertex-wise difference).
        """
        super().__init__(polygon, points)
        self._visible_parts = visible_parts
        self._starting_points = starting_points
        self._directions = directions

    @property
    def visible_parts(self) -> np.ndarray:
        """
        Returns the visible parts of the boundary.

        Returns:
            np.ndarray: The visible parts of the boundary.
        """
        return self._visible_parts

    @property
    def starting_points(self) -> np.ndarray:
        """
        Returns the starting points of the boundary.

        Returns:
            np.ndarray: The starting points of the boundary.
        """
        return self._starting_points

    @property
    def directions(self) -> np.ndarray:
        """
        Returns the directions of the boundary.

        Returns:
            np.ndarray: The directions of the boundary.
        """
        return self._directions


@dataclass
class SpatialParameters(Copyable):
    """
    A data class to store parameters of a space.

    Attributes:
        res (float): Resolution of space.
        vectors (tuple[np.ndarray, np.ndarray]): Spatial vectors representing X and Y axes.
        coords (np.ndarray): Coordinates of accessible points in space
    """
    res: float
    vectors: tuple[np.ndarray, ...]
    coords: np.ndarray


class EnvironmentProxy(WritablePickle):
    """
    A proxy class for the Environment class that provides a non-trivial way of saving the Object object.

    This class is necessary because the Object object has a wrap for the shapely polygon (TexturedPolygon), which adds the property
    texture to it. This makes the Object unable to be saved due to the overridden __new__ method in the shapely Polygon.

    Attributes:
        room: The room attribute from the Environment object.
        visible_area: The visible_area attribute from the Environment object.
        objects: The objects attribute from the Environment object, processed through the safe_objects method.
        walls: The walls attribute from the Environment object, processed through the safe_objects method.
        params: The params attribute from the Environment object.

    Methods:
        safe_objects(*objects: Object): Processes the given objects and returns a list of new Object instances with
                                        specific attributes.
        unsafe_objects(*objects: Object): Processes the given objects and returns a list of new Object instances with
                                          a TexturedPolygon instance as the polygon attribute.
    """
    def __init__(self, env: 'Environment'):
        """
        Initializes the EnvironmentProxy with the attributes of the given Environment object.

        Args:
            env (Environment): The Environment object to proxy.
        """
        self.room = env.room
        self.visible_area = env.visible_area
        self.objects = self.safe_objects(*env.objects)
        self.walls = self.safe_objects(*env.walls)
        self.params = env.params

    @staticmethod
    def safe_objects(*objects: Object):
        """
        Processes the given objects and returns a list of new Object instances with specific attributes. It turns TexturedPolygon to dict.

        This workaround is necessary because of overriding the __new__ method in the Polygon object, making its
        serialisation non-trivial.

        Args:
            *objects (Object): The objects to process.

        Returns:
            list: A list of new Object instances.
        """
        return [
            Object(
                {
                    'obj': obj.polygon.obj,
                    'texture': obj.polygon.texture
                },
                obj.points,
                obj.visible_parts,
                obj.starting_points,
                obj.directions
            ) for obj in objects
        ]

    @staticmethod
    def unsafe_objects(*objects: Object):
        """
        Processes the given objects and returns a list of new Object instances with a TexturedPolygon instance as
        the polygon attribute.

        Args:
            *objects (Object): The objects to process.

        Returns:
            list: A list of new Object instances.
        """
        return [
            Object(
                TexturedPolygon(obj.polygon['obj'], texture=obj.polygon['texture']), # This workaround is necessary because of overriding the __new__ method in the Polygon object, making its serialisation non-trivial
                obj.points,
                obj.visible_parts,
                obj.starting_points,
                obj.directions
            ) for obj in objects
        ]


@dataclass
class Environment(Copyable):
    """
    A data class representing an environment with a room, visible area, objects, and a visible plane.

    Attributes:
        room (Polygon): The room in the environment.
        visible_area (Polygon): The visible area in the environment.
        objects (list[Object]): The list of objects in the environment.
        params (SpatialParameters): Parameters of a space.
    """
    room: Polygon
    visible_area: Polygon
    objects: list[Object]
    walls: list[Object]
    params: SpatialParameters

    def save(self, path: str):
        """
        Save the generated environment to a specified .pkl file.

        Args:
            path (str): The file path to which the environment will be saved.
        """
        EnvironmentProxy(self).save(path)

    @staticmethod
    def load(path: str):
        """
        Load an environment from a specified .pkl file.

        Args:
            path (str): The file path from which the environment will be loaded.
        """
        proxy = EnvironmentProxy.load(path)
        return Environment(
            proxy.room,
            proxy.visible_area,
            proxy.unsafe_objects(*proxy.objects),
            proxy.unsafe_objects(*proxy.walls),
            proxy.params
        )
