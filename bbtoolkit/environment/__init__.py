from dataclasses import dataclass
import numpy as np
from bbtoolkit.utils.data import Copyable, WritablePickle
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


@dataclass
class Environment(Copyable, WritablePickle):
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
