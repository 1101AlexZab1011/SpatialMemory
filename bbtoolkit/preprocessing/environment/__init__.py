from abc import ABC, abstractmethod
from collections import OrderedDict
import configparser
from dataclasses import dataclass
from typing import Callable
import numbers
import shapely as spl
import shapely.prepared as splp
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from bbtoolkit.data import Cached, Copyable, WritablePickle, asynchronous
from bbtoolkit.data.configparser import EvalConfigParser
from bbtoolkit.math.geometry import compute_intersection3d, create_cartesian_space, create_shapely_points, find_closest_points, get_closest_points_indices, poly2vectors, regroup_min_max
from bbtoolkit.math.tensor_algebra import sub3d
from bbtoolkit.preprocessing.environment.viz import plot_polygon
from bbtoolkit.structures.geometry import Texture, TexturedPolygon
from shapely import Polygon, Point
from shapely.validation import explain_validity


class EnvironmentBuilder(Copyable):
    """
    A class for building environments, defining training areas, objects, and creating configurations.

    Attributes:
        xy_min (float): Minimum value for X and Y axes of the environment.
        xy_max (float): Maximum value for X and Y axes of the environment.
        xy_train_min (float | tuple[float, float]): Minimum training area coordinates for X and Y (default is None).
        xy_train_max (float | tuple[float, float]): Maximum training area coordinates for X and Y (default is None).
        res (float): The resolution used for processing geometry data (default is 0.3).

    Methods:
        to_config(self) -> configparser.ConfigParser: Convert the environment configuration to a ConfigParser object.
        save(self, path: str): Save the environment configuration to a file at the specified path.
        load(cls, path: str) -> 'EnvironmentBuilder': Load an environment configuration from a file.
        add_object(self, *args: Object2D) -> 'EnvironmentBuilder': Add objects to the environment.
        plot(self, show: bool = False) -> plt.Figure: Plot the environment.

    Example:
        >>> builder = EnvironmentBuilder(xy_min=0, xy_max=10, res=0.5)
        >>> builder.set_textures(5).set_polygons(8)
        >>> builder.add_object(Object2D(x=(0, 1, 1, 0), y=(0, 0, 1, 1)))
        >>> fig = builder.plot(show=True)
    """
    def __init__(
        self,
        xy_min: float,
        xy_max: float,
        xy_train_min: float | tuple[float, float] = None,
        xy_train_max: float | tuple[float, float] = None,
        res: float =  0.3,
    ) -> None:
        # Initialize the EnvironmentBuilder with specified configurations
        self.xy_min = xy_min
        self.xy_max = xy_max

        if xy_train_max is None:
            self.x_train_max, self.y_train_max = self.xy_max, self.xy_max
        elif isinstance(xy_train_max, float):
            self.x_train_max, self.y_train_max = xy_train_max, xy_train_max
        else:
            self.x_train_max, self.y_train_max = xy_train_max
        if xy_train_min is None:
            self.x_train_min, self.y_train_min = self.xy_min, self.xy_min
        elif isinstance(xy_train_min, float):
            self.x_train_min, self.y_train_min = xy_train_min, xy_train_min
        else:
            self.x_train_min, self.y_train_min = xy_train_min

        self.res = res
        self.objects = list()
        self.walls = list()

    @staticmethod
    def _obj2config(config: EvalConfigParser, name: str, obj: TexturedPolygon) -> None:
        """
        Add an object to the configuration.

        Args:
            config (EvalConfigParser): The configuration to which the object will be added.
            name (str): The name of the object.
            obj (TexturedPolygon): The object to be added to the configuration.
        """
        config.add_section(name)
        config.set(name, 'n_vertices', str(len(obj.exterior.xy[0])))
        config.set(name, 'exterior_x', str(obj.exterior.xy[0].tolist())[1:-1])
        config.set(name, 'exterior_y', str(obj.exterior.xy[1].tolist())[1:-1])
        config.set(name, 'interiors_x', str([interior.xy[0].tolist() for interior in obj.interiors]) if obj.interiors else '')
        config.set(name, 'interiors_y', str([interior.xy[1].tolist() for interior in obj.interiors]) if obj.interiors else '')
        config.set(name, 'texture_id', str(obj.texture.id_))
        config.set(name, 'texture_color', f'"{str(obj.texture.color)}"')
        config.set(name, 'texture_name', f'"{str(obj.texture.name)}"')

    def to_config(self) -> configparser.ConfigParser:
        """
        Generate a configuration parser instance containing environmental information.

        Returns:
            configparser.ConfigParser: Configuration parser instance representing the environmental boundaries,
            training area, building boundaries, and object vertices.

        The generated configuration contains sections representing different aspects of the environment:
        - 'ExternalSources': Empty sections for paths and variables.
        - 'GridBoundaries': Contains maximum and minimum XY coordinate and resolution details.
        - 'TrainingRectangle': Describes the training area coordinates.
        - 'BuildingBoundaries': Holds the maximum number of object points, number of objects, and
          counts of polygons and textures in the environment.

        The object-specific information is stored under individual sections 'Object{i}' for each object.
        Each object section contains 'n_vertices' and 'object_x'/'object_y' detailing the object's vertices.
        """
        parser = EvalConfigParser()
        parser.add_section('ExternalSources')
        parser.set('ExternalSources', 'paths', '')
        parser.set('ExternalSources', 'variables', '')

        parser.add_section('GridBoundaries')
        parser.set('GridBoundaries', 'max_xy', str(self.xy_max))
        parser.set('GridBoundaries', 'min_xy', str(self.xy_min))
        parser.set('GridBoundaries', 'res', str(self.res))

        parser.add_section('TrainingRectangle')
        parser.set('TrainingRectangle', 'min_train_x', str(self.x_train_min))
        parser.set('TrainingRectangle', 'min_train_y', str(self.y_train_min))
        parser.set('TrainingRectangle', 'max_train_x', str(self.x_train_max))
        parser.set('TrainingRectangle', 'max_train_y', str(self.y_train_max))

        parser.add_section('BuildingBoundaries')
        parser.set('BuildingBoundaries', 'max_n_obj_points', str(max([len(obj.exterior.xy[0]) for obj in self.objects])))
        parser.set('BuildingBoundaries', 'n_objects', str(len(self.objects)))

        for i, obj in enumerate(self.objects):
            self._obj2config(parser, f'Object{i+1}', obj)

        parser.set('BuildingBoundaries', 'n_walls', str(len(self.walls)))

        for i, obj in enumerate(self.walls):
            self._obj2config(parser, f'Wall{i+1}', obj)

        return parser

    def save(self, path: str):
        """
        Save the generated environment configuration to a specified .ini file.

        Args:
            path (str): The file path to which the configuration will be saved.

        This method uses the `to_config` method to generate the environment configuration and then writes
        it to a file specified by the 'path' argument.
        """
        config = self.to_config()

        with open(path, 'w') as f:
            config.write(f)

    @staticmethod
    def load(path: str) -> 'EnvironmentBuilder':
        """
        Load an environment configuration from a specified .ini file and create an `EnvironmentBuilder` instance.

        Args:
            path (str): The file path from which the environment configuration will be loaded.

        This method loads the configuration stored in the file specified by the 'path' argument. The loaded
        configuration includes details of the grid boundaries, training rectangle, objects, and building boundaries.
        It then uses this loaded information to create an `EnvironmentBuilder` instance.

        Returns:
            EnvironmentBuilder: An `EnvironmentBuilder` instance with the loaded environment configuration.

        Example:
            >>> builder = EnvironmentBuilder.load('environment_config.ini')
            >>> # The builder variable now contains an `EnvironmentBuilder` instance with the loaded configuration.
        """
        config = EvalConfigParser(interpolation=configparser.ExtendedInterpolation(), allow_no_value=True)
        config.read(path)
        return EnvironmentBuilder(
            config['GridBoundaries'].eval('min_xy'),
            config['GridBoundaries'].eval('max_xy'),
            (
                config['TrainingRectangle'].eval('min_train_x'),
                config['TrainingRectangle'].eval('min_train_y')
            ),
            (
                config['TrainingRectangle'].eval('max_train_x'),
                config['TrainingRectangle'].eval('max_train_y')
            ),
            config['GridBoundaries'].eval('res')
        ).add_wall(
            *[
                TexturedPolygon(
                    shell=[
                        Point(x, y)
                        for x, y in zip(
                            config[f'Wall{i}'].eval('exterior_x'),
                            config[f'Wall{i}'].eval('exterior_y')
                        )
                    ],
                    holes=[
                        [
                            Point(x, y)
                            for x, y in zip(
                                interiors_x,
                                interiors_y
                            )
                        ]
                        for interiors_x, interiors_y in zip(
                            config[f'Wall{i}'].eval('interiors_x'),
                            config[f'Wall{i}'].eval('interiors_y')
                        )
                    ] if config[f'Wall{i}'].eval('interiors_x') else None,
                    texture=Texture(
                        config[f'Wall{i}'].eval('texture_id'),
                        config[f'Wall{i}'].eval('texture_color'),
                        config[f'Wall{i}'].eval('texture_name')
                    )
                )
                for i in range(1, config['BuildingBoundaries'].eval('n_walls')+1)
            ]
        ).add_object(
            *[
                TexturedPolygon(
                    shell = (
                        Point(x, y)
                        for x, y in zip(
                            config[f'Object{i}'].eval('exterior_x'),
                            config[f'Object{i}'].eval('exterior_y')
                        )
                    ),
                    texture=Texture(
                        config[f'Object{i}'].eval('texture_id'),
                        config[f'Object{i}'].eval('texture_color'),
                        config[f'Object{i}'].eval('texture_name')
                    )
                )
                for i in range(1, config['BuildingBoundaries'].eval('n_objects')+1)
            ]
        )

    def __validate_objects(self, *objects: Polygon | TexturedPolygon) -> None:
        for object_ in objects:
            if not object_.is_valid:
                raise ValueError(f'Object {object_} is not valid: {explain_validity(object_)}')

    def __validate_textures(self, *objects: Polygon | TexturedPolygon) -> list[TexturedPolygon]:
        out = list()
        for obj in objects:
            if isinstance(obj, TexturedPolygon):
                out.append(obj)
            else:
                out.append(TexturedPolygon(obj))

        return out

    def add_object(self, *args: Polygon | TexturedPolygon) -> 'EnvironmentBuilder':
        """
        Add one or multiple objects to the environment being constructed.

        Args:
            *args (Polygon | TexturedPolygon): Variable number of objects to be added to the environment.

        This method appends one or more objects to the list of objects within the environment being built.
        Each object contain details such as texture and coordinates of the geometric objects present
        within the environment.

        Returns:
            EnvironmentBuilder: The updated instance of the EnvironmentBuilder with the added objects.
        """
        self.__validate_objects(*args)
        self.objects += list(self.__validate_textures(*args))
        return self

    def add_wall(self, *args: Polygon | TexturedPolygon) -> 'EnvironmentBuilder':
        """
        Add one or multiple objects to the environment being constructed.

        Args:
            *args (Polygon | TexturedPolygon): Variable number of objects to be added to the environment.

        This method appends one or more objects to the list of objects within the environment being built.
        Each object contain details such as texture and coordinates of the geometric objects present
        within the environment.

        Returns:
            EnvironmentBuilder: The updated instance of the EnvironmentBuilder with the added objects.
        """
        self.__validate_objects(*args)
        self.walls += list(self.__validate_textures(*args))
        return self

    def remove_object(self, i: int) -> 'EnvironmentBuilder':
        """
        Removes the object at the specified index from the list of objects in the environment.

        Args:
            i (int): The index of the object to be removed.

        Returns:
            EnvironmentBuilder: The modified EnvironmentBuilder object after removing the specified object.
        """
        self.objects.pop(i)
        return self

    def remove_wall(self, i: int) -> 'EnvironmentBuilder':
        """
        Removes the wall at the specified index from the list of objects in the environment.

        Args:
            i (int): The index of the object to be removed.

        Returns:
            EnvironmentBuilder: The modified EnvironmentBuilder object after removing the specified object.
        """
        self.walls.pop(i)
        return self

    def __add__(self, other: 'EnvironmentBuilder') -> 'EnvironmentBuilder':
        """
        Adds the objects and properties of two EnvironmentBuilder instances.

        Merges the objects from two separate EnvironmentBuilder instances into a new instance.
        The new instance retains the original attributes of the first instance (self), such as grid boundaries,
        training rectangle, resolution, and objects. It also appends the objects and updates the properties (textures
        and polygons) as specified.

        Args:
            other (EnvironmentBuilder): Another EnvironmentBuilder instance to be combined with the current one.

        Returns:
            EnvironmentBuilder: A new EnvironmentBuilder instance containing the combined objects and attributes from self and other.
        """
        return EnvironmentBuilder(
            self.xy_min,
            self.xy_max,
            self.x_train_min,
            self.y_train_min,
            self.x_train_max,
            self.y_train_max,
            self.res,
        ).add_object(
            *self.objects,
            *other.objects
        ).add_wall(
            *self.walls,
            *other.walls
        )

    def plot(self, ax: plt.Axes = None) -> plt.Figure:
        """
        Visualizes the environment layout by generating a plot using matplotlib.

        Args:
            ax (plt.Axes, optional): Matplotlib Axes to use for plotting. If None, a new subplot is created. Defaults to None.

        This method generates a plot that visualizes the layout of the environment using matplotlib. It plots the
        boundaries of the entire environment, the training area, and the objects within it.

        Returns:
            plt.Figure: A matplotlib Figure object representing the generated plot.

        Example:
            >>> builder = EnvironmentBuilder(xy_min=0, xy_max=10, xy_train_min=(2, 2), xy_train_max=(8, 8))
            >>> obj1 = Object2D(x=(0, 1, 1), y=(0, 1, 0))
            >>> obj2 = Object2D(x=(2, 3, 3, 2), y=(2, 2, 3, 3))

            >>> builder.add_object(obj1, obj2)
            >>> fig = builder.plot(show=True)
            >>> # The plot showing the environment layout will be displayed.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # plot whole area
        ax.plot(
            (self.xy_min, self.xy_min, self.xy_max, self.xy_max, self.xy_min),
            (self.xy_min, self.xy_max, self.xy_max, self.xy_min, self.xy_min),
            '-', color='#999', label='Whole Area'
        )
        # plot training area
        ax.plot(
            (self.x_train_min, self.x_train_min, self.x_train_max, self.x_train_max, self.x_train_min),
            (self.y_train_min, self.y_train_max, self.y_train_max, self.y_train_min, self.y_train_min),
            '--', color='tab:blue', label='Training Area'
        )

        # plot walls
        for wall in self.walls:
            if wall.texture.color is None:
                plot_polygon(wall, color='tab:orange', ax=ax)
            else:
                plot_polygon(wall, ax=ax)

        # plot objects
        for obj in self.objects:
            plot_polygon(obj, ax=ax)

        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
        ax.grid()

        return fig


class AbstractVisiblePlaneSubset(Copyable, ABC):
    """
    Represents an abstract class for subset of a visible plane.
    This class is used to define the interface for accesing visible points for particular object.
    """
    @abstractmethod
    def __getitem__(self, indices: int | tuple[int, int] | tuple[int, int, int]) -> np.ndarray: # position, points, axis
        """
        Allows the VisiblePlaneSubset object to be indexed. First index is the position, second is the points, third is the axis.
        """
        pass
    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of room points coordinates in the VisiblePlaneSubset.
        """
        pass
    @abstractmethod
    def __iter__(self) -> np.ndarray:
        """
        Allows iteration over points of the VisiblePlaneSubset object.
        """
        pass
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the VisiblePlaneSubset (number of positions in space, points of object boundaries, x or y axis).
        """
        pass


class AbstractVisiblePlane(Copyable, ABC):
    """
    Base class for visible plane in a 2D space.
    """
    @abstractmethod
    def __getitem__(self, i: int) -> np.ndarray | AbstractVisiblePlaneSubset:
        """
        Allows the VisiblePlane object to be indexed. Each index corresponds to a different object.
        """
        pass


class PrecomputedVisiblePlane(AbstractVisiblePlane):
    """
    A class representing a visible plane in a 3D space.

    Attributes:
        _data (np.ndarray): The visible coordinates of the plane.
        _slices (list[slice]): The slices for each object in the visible coordinates.
    """
    def __init__(
        self,
        visible_coordinates: np.ndarray, # shape (n_locations, n_boundary_points, 2)
        object_slices: list[slice], # list of slices for each object in visible_coordinates
    ):
        """
        Initializes the VisiblePlane with visible coordinates and object slices.

        Args:
            visible_coordinates (np.ndarray): The visible coordinates of the plane.
                Shape is (n_locations, n_boundary_points, 2).
            object_slices (list[slice]): The slices for each object in the visible coordinates.
        """
        self._data = visible_coordinates
        self._slices = object_slices

    def __getitem__(self, i: int) -> np.ndarray:
        """
        Returns the visible coordinates for the object at the given index.

        Args:
            i (int): The index of the object.

        Returns:
            np.ndarray: The visible coordinates for the object.
        """
        return self._data[:, self._slices[i], :]

    @property
    def data(self) -> np.ndarray:
        """
        Returns the visible coordinates of the plane.

        Returns:
            np.ndarray: The visible coordinates of the plane.
        """
        return self._data

    @property
    def slices(self) -> list[slice]:
        """
        Returns the slices for each object in the visible coordinates.

        Returns:
            list[slice]: The slices for each object.
        """
        return self._slices


class VisiblePlaneSubset(AbstractVisiblePlaneSubset):
    """
    A class that represents a subset of a visible plane in a 3D space.

    Attributes:
        visible_plane (VisiblePlane): The visible plane object that this subset belongs to.
        object_index (int): The index of the object in the visible plane.
    """
    def __init__(
        self,
        visible_plane: 'LazyVisiblePlane',
        object_index: int
    ):
        """
        Constructs all the necessary attributes for the VisiblePlaneSubset object.

        Args:
            visible_plane (VisiblePlane): The visible plane object that this subset belongs to.
            object_index (int): The index of the object in the visible plane.
        """
        self.visible_plane = visible_plane
        self.object_index = object_index

    def __getitem__(self, indices: int | tuple[int, int] | tuple[int, int, int]) -> np.ndarray: # position, points, axis
        """
        Allows the VisiblePlaneSubset object to be indexed.

        Args:
            indices (int | tuple[int, int] | tuple[int, int, int]): The indices to access.

        Returns:
            np.ndarray: The accessed elements.
        """
        if isinstance(indices, tuple):
            position_index = indices[0]
            rest_indices = indices[1:]
        else:
            position_index = indices
            rest_indices = ()

        coords_x = np.atleast_1d(self.visible_plane.room_points_coordinates[position_index, 0])
        coords_y = np.atleast_1d(self.visible_plane.room_points_coordinates[position_index, 1])

        return np.concatenate([self.visible_plane(x, y)[self.object_index] for x, y in zip(coords_x, coords_y)])[*rest_indices] # points, axis

    def __len__(self) -> int:
        """
        Returns the number of room points coordinates in the VisiblePlaneSubset.

        Returns:
            int: Number of room points coordinates.
        """
        return len(self.visible_plane.room_points_coordinates)

    def __iter__(self) -> np.ndarray:
        """
        Allows iteration over the VisiblePlaneSubset object.

        Yields:
            np.ndarray: The accessed elements.
        """
        for index in range(len(self)):
            yield self[index]

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the VisiblePlaneSubset.

        Returns:
            tuple[int, ...]: A tuple representing the shape of the VisiblePlaneSubset.
        """
        return self.visible_plane.room_points_coordinates.shape[0],\
            self.visible_plane.slices[self.object_index].stop - self.visible_plane.slices[self.object_index].start,\
            self.visible_plane.room_points_coordinates.shape[-1]


class LazyVisiblePlane(AbstractVisiblePlane):
    """
    A class that represents a visible plane in a 2D space.

    Attributes:
        room_points_coordinates (np.ndarray): Coordinates of the room points.
        slices (list[slice]): List of slices for boundary points.
        boundary_points (np.ndarray): Coordinates of the boundary points.
        starting_points (np.ndarray): Starting points for the plane.
        directions (np.ndarray): Directions for the plane.
        cache_manager (Cached): Cache manager for the plane.
    """
    def __init__(
        self,
        starting_points: np.ndarray | list[np.ndarray],
        directions: np.ndarray | list[np.ndarray],
        room_points_coordinates: np.ndarray,
        boundary_points_coordinates: list[np.ndarray],
        cache_manager: Cached = None
    ):
        """
        Constructs all the necessary attributes for the LazyVisiblePlane object.

        Args:
            starting_points (np.ndarray | list[np.ndarray]): Starting points for the plane.
            directions (np.ndarray | list[np.ndarray]): Directions for the plane.
            room_points_coordinates (np.ndarray): Coordinates of the room points.
            boundary_points_coordinates (list[np.ndarray]): Coordinates of the boundary points.
            cache_manager (Cached, optional): Cache manager for the plane. Defaults to None.
        """
        self.room_points_coordinates = room_points_coordinates

        cumulative_lengths = np.cumsum([len(boundary) for boundary in boundary_points_coordinates])
        self.slices = [slice(from_, to) for from_, to in zip([0] + list(cumulative_lengths[:-1]), cumulative_lengths)]
        boundary_points = np.concatenate(boundary_points_coordinates)
        self.boundary_points = np.concatenate( # add z coordinate with zeros to local boundary points
            [
                boundary_points,
                np.zeros((*boundary_points.shape[:-1], 1))
            ],
            axis=-1
        )

        starting_points = np.concatenate(starting_points) if isinstance(starting_points, list) else starting_points
        self.starting_points = np.concatenate( # add z coordinate with zeros to local starting points
            [
                starting_points,
                np.zeros((*starting_points.shape[:-1], 1))
            ],
            axis=-1
        )

        directions = np.concatenate(directions) if isinstance(directions, list) else directions
        self.directions = np.concatenate( # add z coordinate with zeros to directions
            [
                directions,
                np.zeros((*directions.shape[:-1], 1))
            ],
            axis=-1
        )
        self.cache_manager = cache_manager if cache_manager is not None else Cached()


    def _process_visible_coordinates(
        self,
        coords_x: float,
        coords_y: float
    ) -> list[np.ndarray]:
        """
        Compute all visible points from the given coordinate.

        Args:
            coords_x (float): X coordinate.
            coords_y (float): Y coordinate.

        Returns:
            list[np.ndarray]: List of visible xy coordinates.
        """
        local_starting_points = self.starting_points - np.array([[coords_x, coords_y, 0]])
        local_boundary_points = self.boundary_points - np.array([[coords_x, coords_y, 0]])
        alpha_pt, alpha_occ = compute_intersection3d(
            np.zeros_like(local_boundary_points),
            local_starting_points,
            local_boundary_points,
            self.directions
        )
        mask = ~np.any((alpha_pt < 1 - 1e-5) & (alpha_pt > 0) & (alpha_occ < 1) & (alpha_occ > 0), axis=0)
        visible_xy = np.full((len(local_boundary_points), 2), np.nan)

        visible_xy[mask] = self.boundary_points[mask, :2]

        return [
            visible_xy[slice_]
            for slice_ in self.slices
        ]


    def __call__(
        self,
        coords_x: float,
        coords_y: float
    ) -> list[np.ndarray]:
        """
        Makes the VisiblePlane object callable.

        Args:
            coords_x (float): X coordinate.
            coords_y (float): Y coordinate.

        Returns:
            list[np.ndarray]: List of visible xy coordinates for each object.
        """
        @self.cache_manager
        def nested_call(
            coords_x: float,
            coords_y: float
        ) -> list[np.ndarray]:
            return self._process_visible_coordinates(coords_x, coords_y)

        return nested_call(coords_x, coords_y)

    def __getitem__(self, index: int) -> VisiblePlaneSubset:
        """
        Allows the LazyVisiblePlane object to be indexed.

        Args:
            index (int): Index of the desired slice.

        Returns:
            VisiblePlaneSubset: A subset of the VisiblePlane.
        """
        return VisiblePlaneSubset(self, index)

    def __len__(self) -> int:
        """
        Returns the number of slices in the LazyVisiblePlane.

        Returns:
            int: Number of slices.
        """
        return len(self.slices)

    def __iter__(self) -> VisiblePlaneSubset:
        """
        Allows iteration over the VisiblePlane object.

        Yields:
            VisiblePlaneSubset: A subset of the LazyVisiblePlane.
        """
        for index in range(len(self)):
            yield VisiblePlaneSubset(self, index)


class AsyncVisiblePlaneSubset(VisiblePlaneSubset):
    """
    A class that represents an asynchronous subset of a visible plane in a 2D space.

    This class inherits from the VisiblePlaneSubset class and overrides some of its methods
    to provide asynchronous functionality.

    Attributes:
        visible_plane (VisiblePlane): The visible plane object that this subset belongs to.
        object_index (int): The index of the object in the visible plane.
    """
    def __getitem__(self, indices: int | tuple[int, int] | tuple[int, int, int]): # position, points, axis
        if isinstance(indices, tuple):
            position_index = indices[0]
            rest_indices = indices[1:]
        else:
            position_index = indices
            rest_indices = ()

        if isinstance(position_index, numbers.Integral):
            closest_points_indices = set(get_closest_points_indices(
                self.visible_plane.room_points_coordinates,
                position_index,
                tree=self.visible_plane.tree,
                n_points=self.visible_plane.n_neighbours + 1 # + 1 to include the point itself
            )) - {position_index}
            res = self.visible_plane( # validate presence in cache
                self.visible_plane.room_points_coordinates[position_index, 0],
                self.visible_plane.room_points_coordinates[position_index, 1]
            )

            for i in closest_points_indices:
                _ = self.visible_plane( # validate presence in cache
                    self.visible_plane.room_points_coordinates[i, 0],
                    self.visible_plane.room_points_coordinates[i, 1]
                )
            return res.result()[self.object_index][*rest_indices]

        coords_x = np.atleast_1d(self.visible_plane.room_points_coordinates[position_index, 0])
        coords_y = np.atleast_1d(self.visible_plane.room_points_coordinates[position_index, 1])

        return np.concatenate([self.visible_plane(x, y).result()[self.object_index] for x, y in zip(coords_x, coords_y)])[*rest_indices] # points, axis

    def __len__(self) -> int:
        """
        Returns the number of room points coordinates in the AsyncVisiblePlaneSubset.

        Returns:
            int: Number of room points coordinates.
        """
        return len(self.visible_plane.room_points_coordinates)

    def __iter__(self) -> np.ndarray:
        """
        Allows iteration over the AsyncVisiblePlaneSubset object.

        Yields:
            np.ndarray: The accessed elements.
        """
        for index in range(len(self)):
            yield self[index]

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the AsyncVisiblePlaneSubset.

        Returns:
            tuple[int, ...]: A tuple representing the shape of the AsyncVisiblePlaneSubset.
        """
        return self.visible_plane.room_points_coordinates.shape[0],\
            self.visible_plane.slices[self.object_index].stop - self.visible_plane.slices[self.object_index].start,\
            self.visible_plane.room_points_coordinates.shape[-1]


class AsyncVisiblePlane(LazyVisiblePlane):
    """
    A class that represents a visible plane in a 2D space.

    This class inherits from the LazyisiblePlane class and overrides some of its methods
    to provide lazy asynchronous functionality.

    Attributes:
        n_neighbours (int): The number of neighbours to consider.
        tree (KDTree): The KDTree for efficient nearest neighbour search.
    """
    def __init__(
        self,
        starting_points: np.ndarray | list[np.ndarray],
        directions: np.ndarray | list[np.ndarray],
        room_points_coordinates: np.ndarray,
        boundary_points_coordinates: list[np.ndarray], # list of boundary points coordinates for each object
        cache_manager: Cached = None,
        n_neighbours: int = 10,
    ):
        """
        Constructs all the necessary attributes for the AsyncVisiblePlane object.

        Args:
            starting_points (np.ndarray | list[np.ndarray]): Starting points for the plane.
            directions (np.ndarray | list[np.ndarray]): Directions for the plane.
            room_points_coordinates (np.ndarray): Coordinates of the room points.
            boundary_points_coordinates (list[np.ndarray]): Coordinates of the boundary points.
            cache_manager (Cached, optional): Cache manager for the plane. Defaults to None.
            n_neighbours (int, optional): The number of neighbours to consider. Defaults to 10.
        """
        if cache_manager is None:
            cache_manager = Cached(cache_storage=OrderedDict())

        super().__init__(
            starting_points,
            directions,
            room_points_coordinates,
            boundary_points_coordinates,
            cache_manager
        )
        self.n_neighbours = n_neighbours
        self.tree = KDTree(room_points_coordinates)

    def __call__(
        self,
        coords_x: float,
        coords_y: float
    ) -> list[np.ndarray]:
        """
        Makes the AsyncVisiblePlane object callable.

        Args:
            coords_x (float): X coordinate.
            coords_y (float): Y coordinate.

        Returns:
            list[np.ndarray]: List of visible xy coordinates for each object.
        """
        @self.cache_manager
        @asynchronous
        def nested_call(
            coords_x: float,
            coords_y: float
        ) -> list[np.ndarray]:
            return self._process_visible_coordinates(coords_x, coords_y)

        return nested_call(coords_x, coords_y)

    def __getitem__(self, index: int) -> AsyncVisiblePlaneSubset:
        """
        Allows the LasyAsyncVisiblePlane object to be indexed.

        Args:
            index (int): Index of the desired slice.

        Returns:
            AsyncVisiblePlaneSubset: An asynchronous subset of the LasyAsyncVisiblePlane.
        """
        return AsyncVisiblePlaneSubset(self, index)


class Area(Copyable):
    """
    A class representing an area defined by a polygon and a set of points.

    Attributes:
        polygon (Polygon): The polygon defining the area.
        points (np.ndarray): The set of points within the area.
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
class Environment(WritablePickle):
    """
    A data class representing an environment with a room, visible area, objects, and a visible plane.

    Attributes:
        room (Area): The room in the environment.
        visible_area (Area): The visible area in the environment.
        objects (list[Object]): The list of objects in the environment.
    """
    room: Area
    visible_area: Area
    objects: list[Object]
    walls: list[Object]


class EnvironmentCompiler:
    """
    A class that compiles an environment using an EnvironmentBuilder.

    Attributes:
        builder (EnvironmentBuilder): The builder used to compile the environment.
    """
    def __init__(
        self,
        builder: EnvironmentBuilder,
        visible_plane_compiler: Callable[
            [
                np.ndarray | list[np.ndarray],
                np.ndarray | list[np.ndarray],
                np.ndarray,
                list[np.ndarray]
            ],
            np.ndarray,
        ] = None,
    ):
        """
        Initializes the EnvironmentCompiler with an EnvironmentBuilder.

        Args:
            builder (EnvironmentBuilder): The builder used to compile the environment.
        """
        self._builder = builder
        self._visible_plane_compiler = visible_plane_compiler

    @property
    def builder(self) -> EnvironmentBuilder:
        """
        Returns the builder used to compile the environment.

        Returns:
            EnvironmentBuilder: The builder used to compile the environment.
        """
        return self._builder

    def compile_room_area(self) -> Polygon:
        """
        Compiles the room area boundaries into a Polygon.

        Returns:
            Polygon: The room area.
        """
        return Polygon([
            Point(self.builder.xy_min, self.builder.xy_min),
            Point(self.builder.xy_min, self.builder.xy_max),
            Point(self.builder.xy_max, self.builder.xy_max),
            Point(self.builder.xy_max, self.builder.xy_min)
        ])

    def compile_visible_area(self) -> Polygon:
        """
        Compiles the visible area boundaries into a Polygon.

        Returns:
            Polygon: The visible area.
        """
        return Polygon([
            Point(self.builder.x_train_min, self.builder.y_train_min),
            Point(self.builder.x_train_min, self.builder.y_train_max),
            Point(self.builder.x_train_max, self.builder.y_train_max),
            Point(self.builder.x_train_max, self.builder.y_train_min)
        ])

    def compile_space_points(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray
    ):
        """
        Compiles the space points from a range of coordinates.

        Args:
            from_ (tuple[int, int]): The starting coordinates.
            to (tuple[int, int]): The ending coordinates.

        Returns:
            list[Point]: The compiled space points.
        """
        return create_shapely_points(
            x_coords, y_coords,
            res=self.builder.res
        )

    @staticmethod
    def compile_room_points(
        space_points: list[Point],
        objects: list[Polygon]
    ) -> list[Point]:
        """
        Compiles the room points from a list of space points.

        Args:
            space_points (list[Point]): The space points.
            objects (list[Polygon]): The objects.

        Returns:
            list[Point]: The compiled room points.
        """
        prepared = splp.prep(spl.GeometryCollection([poly.obj for poly in objects]))
        return list(filter(prepared.disjoint, space_points))

    @staticmethod
    def compile_visible_area_points(
        room_points: list[Point],
        visible_area: Polygon
    ) -> list[Point]:
        """
        Compiles the visible area points from a list of room points and a visible area.

        Args:
            room_points (list[Point]): The room points.
            visible_area (Polygon): The visible area.

        Returns:
            list[Point]: The compiled visible area points.
        """
        prepared = splp.prep(visible_area)
        return list(filter(prepared.contains, room_points))

    @staticmethod
    def compile_boundary_points(
        space_points: list[Point],
        objects: list[Polygon]
    ) -> list[Point]:
        """
        Compiles the boundary points from a list of space points.

        Args:
            space_points (list[Point]): The space points.

        Returns:
            list[Point]: The compiled boundary points.
        """
        prepared_objects_boundaries = [
            splp.prep(obj.boundary)
            for obj in objects
        ]
        return [
            val
            for prepared in prepared_objects_boundaries
            if len(val := list(filter(prepared.crosses, space_points)))
        ]

    @staticmethod
    def align_objects(
        boundary_points: list[np.ndarray],
        objects: list[Polygon]
    ) -> list[TexturedPolygon]:
        """
        Aligns the object polygons with the resolution of the space grid.

        Args:
            boundary_points (np.ndarray): The boundary points.

        Returns:
            list[TexturedPolygon]: The aligned object boundaries.

        Notes:
            Alignment is done accordingly to the given boundary points. Shapes of resulting objects can be incorrect.
        """
        # points of space are centers of circles of r=0.5*res.
        # These centers sometimes are not consistent with boundaries of original objects.
        # So we need to make correction to be consistent with resolution
        object_matrices_exterior = [
            np.array(obj.exterior.coords.xy).T
            for obj in objects
        ]

        object_matrices_exterior_corrected = [
            find_closest_points(points, object_matrix)
            for object_matrix, points in zip(object_matrices_exterior, boundary_points)
        ]
        object_matrices_interior = [
            [np.array(interior.coords.xy).T for interior in obj.interiors]
            for obj in objects
        ]
        object_matrices_interior_corrected = [
            [
                find_closest_points(points, object_matrix)
                for object_matrix in interior
            ] if interior else None
            for interior, points in zip(object_matrices_interior, boundary_points)
        ]
        return [ # redefine objects according to space grid correction
            TexturedPolygon(
                zip(
                    obj_coords_exterior[:, 0],
                    obj_coords_exterior[:, 1]
                ),
                [
                    zip(hole[:, 0], hole[:, 1])
                    for hole in obj_holes
                ] if obj_holes else None,
                texture=obj.texture
            )
            for obj,
            obj_coords_exterior,
            obj_holes
            in zip(
                objects,
                object_matrices_exterior_corrected,
                object_matrices_interior_corrected
            )
        ]

    @staticmethod
    def compile_visible_plane(
        starting_points: np.ndarray | list[np.ndarray],
        directions: np.ndarray | list[np.ndarray],
        room_points_coordinates: np.ndarray,
        boundary_points_coordinates: list[np.ndarray] # list of boundary points coordinates for each object
    ) -> LazyVisiblePlane | PrecomputedVisiblePlane:
        """
        Compiles the visible plane from starting points, directions, room points coordinates, and boundary points coordinates.

        Args:
            starting_points (np.ndarray): The starting points.
            directions (np.ndarray): The directions.
            room_points_coordinates (np.ndarray): The room points coordinates.
            boundary_points_coordinates (list[np.ndarray]): The boundary points coordinates for each object.

        Returns:
            VisiblePlane | PrecomputedVisiblePlane: The compiled visible plane.
        """
        starting_points = np.concatenate(starting_points) if isinstance(starting_points, list) else starting_points
        directions = np.concatenate(directions) if isinstance(directions, list) else directions
        all_boundary_points_coordinates = np.concatenate(boundary_points_coordinates)

        n_boundary_points = len(all_boundary_points_coordinates)
        n_training_points = len(room_points_coordinates)

        directions = np.concatenate( # add z coordinate with zeros to directions
            [
                directions,
                np.zeros((*directions.shape[:-1], 1))
            ],
            axis=-1
        )
        local_starting_points = sub3d( # each starting point minus each point of room area
            starting_points,
            room_points_coordinates,
            return_2d=False
        )
        local_starting_points = np.concatenate( # add z coordinate with zeros to local starting points
            [
                local_starting_points,
                np.zeros((*local_starting_points.shape[:-1], 1))
            ],
            axis=-1
        )

        local_boundary_points = sub3d( # each boundary point minus each point of room area
            all_boundary_points_coordinates,
            room_points_coordinates,
            return_2d=False
        )
        local_boundary_points = np.concatenate( # add z coordinate with zeros to local boundary points
            [
                local_boundary_points,
                np.zeros((*local_boundary_points.shape[:-1], 1))
            ],
            axis=-1
        )

        alpha_pt, alpha_occ = compute_intersection3d( # compute intersection points between each line and each boundary using cross product
            np.zeros_like(local_boundary_points), # starting point of each line is [0, 0, 0] (egocentric location of agent)
            local_starting_points, # starting points of each line is each point of object relative to egocentric location of agent
            local_boundary_points, # direction of each line is each boundary point relative to egocentric location of agent
            np.repeat(directions[np.newaxis, :, :], n_training_points, axis=0) # direction of each line is distanec from one vertex of an object to another
        )

        mask = ~np.any((alpha_pt < 1 - 1e-5) & (alpha_pt > 0) & (alpha_occ < 1) & (alpha_occ > 0), axis=1)

        visible_xy = np.full((n_training_points, n_boundary_points, 2), np.nan)
        for location, location_mask in enumerate(mask):
            visible_xy[location, location_mask] = all_boundary_points_coordinates[location_mask]

        cumulative_lengths = np.cumsum([len(boundary) for boundary in boundary_points_coordinates])
        slices = [slice(from_, to) for from_, to in zip([0] + list(cumulative_lengths[:-1]), cumulative_lengths)]

        return PrecomputedVisiblePlane(visible_xy, slices)

    @staticmethod
    def compile_directions(objects: list[Polygon]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Compiles the directions from a list of objects.

        Args:
            objects (list[Polygon]): The objects.

        Returns:
            tuple[np.ndarray, np.ndarray]: The compiled directions.
        """
        starting_points, directions = list(), list()
        for obj in objects:
            starting_points_, directions_ = poly2vectors(obj)
            starting_points.append(starting_points_)
            directions.append(directions_)

        return starting_points, directions

    @staticmethod
    def compile_objects(
        space_points: list[Point],
        visible_coordinates: np.ndarray,
        objects: list[Polygon],
        visible_plane_compiler: Callable[
            [
                np.ndarray | list[np.ndarray],
                np.ndarray | list[np.ndarray],
                np.ndarray,
                list[np.ndarray]
            ],
                np.ndarray,
            ] = None
    ):
        space_points_coordinates = np.array([
            [point.centroid.xy[0][0], point.centroid.xy[1][0]]
            for point in space_points
        ])
        objects_corrected = EnvironmentCompiler.align_objects(
            [space_points_coordinates for _ in range(len(objects))],
            objects
        )
        boundary_points = EnvironmentCompiler.compile_boundary_points(space_points, objects_corrected)
        boundary_points_coordinates = [
            np.array([[point.centroid.xy[0][0], point.centroid.xy[1][0]] for point in boundary_point])
            for boundary_point in boundary_points
        ]

        starting_points, directions = EnvironmentCompiler.compile_directions(objects_corrected)
        visible_plane_compiler = EnvironmentCompiler.compile_visible_plane \
            if visible_plane_compiler is None else visible_plane_compiler
        visible_plane = visible_plane_compiler(
            starting_points,
            directions,
            visible_coordinates,
            boundary_points_coordinates
        )
        return [
            Object(
                obj,
                boundary_points_coordinates[i],
                visible_plane[i],
                starting_points[i],
                directions[i]
            )
            for i, obj in enumerate(objects_corrected)
        ]

    def compile(
        self
    ) -> Environment:
        """
        Compiles the environment.

        Returns:
            Environment: The compiled environment.
        """
        room_area = self.compile_room_area()
        visible_area = self.compile_visible_area()

        x_coords, y_coords = create_cartesian_space(
            *regroup_min_max(*visible_area.bounds),
            self.builder.res
        )

        space_points = self.compile_space_points(
            x_coords, y_coords
        )

        visible_space_points = self.compile_room_points(
            space_points,
            self.builder.walls + self.builder.objects
        )
        visible_space_points_coordinates = np.array([
            [point.centroid.xy[0][0], point.centroid.xy[1][0]]
            for point in visible_space_points
        ])

        visible_objects = self.compile_objects(
            space_points,
            visible_space_points_coordinates,
            self.builder.objects + self.builder.walls,
            self._visible_plane_compiler
        )
        return Environment(
            Area(room_area),
            Area(visible_area, visible_space_points_coordinates),
            visible_objects[:len(self.builder.objects)],
            visible_objects[len(self.builder.objects):]
        )