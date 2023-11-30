import configparser
from dataclasses import dataclass
from typing import Iterator, Sequence
import shapely as spl
import shapely.prepared as splp
from matplotlib import pyplot as plt
import numpy as np
from bbtoolkit.data import Copyable, WritablePickle
from bbtoolkit.data.configparser import EvalConfigParser
from bbtoolkit.math.geometry import compute_intersection3d, create_cartesian_space, create_shapely_points, find_closest_points, poly2vectors, regroup_min_max
from bbtoolkit.math.tensor_algebra import sub3d
from bbtoolkit.preprocessing.environment.viz import plot_polygon
from bbtoolkit.structures.geometry import Texture, TexuredPolygon
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
            parser.add_section(f'Object{i+1}')
            parser.set(f'Object{i+1}', 'n_vertices', str(len(obj.exterior.xy[0])))
            parser.set(f'Object{i+1}', 'object_x', str(obj.exterior.xy[0].tolist())[1:-1])
            parser.set(f'Object{i+1}', 'object_y', str(obj.exterior.xy[1].tolist())[1:-1])
            parser.set(f'Object{i+1}', 'texture_id', str(obj.texture.id_))
            parser.set(f'Object{i+1}', 'texture_color', str(obj.texture.color))
            parser.set(f'Object{i+1}', 'texture_name', str(obj.texture.name))

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
        ).add_object(
            *[
                TexuredPolygon(
                    (
                        Point(x, y)
                        for x, y in zip(
                            config[f'Object{i}'].eval('object_x'),
                            config[f'Object{i}'].eval('object_y')
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

    def __validate_objects(self, *objects: Polygon | TexuredPolygon) -> None:
        for object_ in objects:
            if not object_.is_valid:
                raise ValueError(f'Object {object_} is not valid: {explain_validity(object_)}')

    def __validate_textures(self, *objects: Polygon | TexuredPolygon) -> list[TexuredPolygon]:
        out = list()
        for obj in objects:
            if isinstance(obj, TexuredPolygon):
                out.append(obj)
            else:
                out.append(TexuredPolygon(obj))

        return out

    def add_object(self, *args: Polygon | TexuredPolygon) -> 'EnvironmentBuilder':
        """
        Add one or multiple Object2D instances to the environment being constructed.

        Args:
            *args (Object2D): Variable number of Object2D instances to be added to the environment.

        This method appends one or more Object2D instances to the list of objects within the environment being built.
        The Object2D instances contain details such as vertices and coordinates of the geometric objects present
        within the environment.

        Returns:
            EnvironmentBuilder: The updated instance of the EnvironmentBuilder with the added objects.

        Example:
            >>> builder = EnvironmentBuilder(xy_min=0, xy_max=10)  # Create an EnvironmentBuilder instance
            >>> obj1 = Object2D(x=(0, 1, 1), y=(0, 1, 0))  # Define an Object2D instance
            >>> obj2 = Object2D(x=(2, 3, 3, 2), y=(2, 2, 3, 3))  # Define another Object2D instance

            >>> builder.add_object(obj1, obj2)
            >>> # The builder instance now includes obj1 and obj2 within the list of objects.
        """
        self.__validate_objects(*args)
        self.objects += list(self.__validate_textures(*args))
        return self

    def remove_object(self, i: int) -> 'EnvironmentBuilder':
        """
        Removes the object at the specified index from the list of objects in the environment.

        Args:
            i (int): The index of the object to be removed.

        Returns:
            EnvironmentBuilder: The modified EnvironmentBuilder object after removing the specified object.

        Example:
            >>> builder = EnvironmentBuilder(xy_min=0, xy_max=10, xy_train_min=(2, 2), xy_train_max=(8, 8))
            >>> obj1 = Object2D(x=(0, 1, 1), y=(0, 1, 0))
            >>> obj2 = Object2D(x=(2, 3, 3, 2), y=(2, 2, 3, 3))

            >>> builder.add_object(obj1, obj2)
            >>> builder = builder.remove_object(0)
            >>> # Object at index 0 has been removed from the EnvironmentBuilder.
        """
        self.objects.pop(i)
        return self

    def __getitem__(self, i: int) -> Polygon:
        """
        Accesses the object at the specified index within the list of objects.

        Args:
            i (int): The index of the object to retrieve.

        Returns:
            Polygon: The object at the specified index in the list of objects.

        Example:
            >>> builder = EnvironmentBuilder(xy_min=0, xy_max=10, xy_train_min=(2, 2), xy_train_max=(8, 8))
            >>> obj1 = Object2D(x=(0, 1, 1), y=(0, 1, 0))
            >>> obj2 = Object2D(x=(2, 3, 3, 2), y=(2, 2, 3, 3))

            >>> builder.add_object(obj1, obj2)
            >>> retrieved_obj = builder[0]
            >>> # 'retrieved_obj' is now equal to the object at index 0 in the EnvironmentBuilder.
        """
        return self.objects[i]

    def __setitem__(self, i: int, obj: Polygon) -> None:
        """
        Sets the object at the specified index within the list of objects.

        Args:
            i (int): The index of the object to set.
            obj (Polygon): The object to be set at the specified index.

        Example:
            >>> builder = EnvironmentBuilder(xy_min=0, xy_max=10, xy_train_min=(2, 2), xy_train_max=(8, 8))
            >>> obj1 = Object2D(x=(0, 1, 1), y=(0, 1, 0))
            >>> obj2 = Object2D(x=(2, 3, 3, 2), y=(2, 2, 3, 3))

            >>> builder.add_object(obj1, obj2)
            >>> builder[0] = obj2
            >>> # Object at index 0 has been replaced with obj2.
        """
        self.__validate_objects(obj)
        self.objects[i] = self.__validate_textures(obj)[0]

    def __len__(self) -> int:
        """
        Returns the number of objects currently stored in the environment.

        Returns:
            int: The number of objects in the environment.

        Example:
            >>> builder = EnvironmentBuilder(xy_min=0, xy_max=10, xy_train_min=(2, 2), xy_train_max=(8, 8))
            >>> obj1 = Object2D(x=(0, 1, 1), y=(0, 1, 0))
            >>> obj2 = Object2D(x=(2, 3, 3, 2), y=(2, 2, 3, 3))

            >>> builder.add_object(obj1, obj2)
            >>> obj_count = len(builder)
            >>> # 'obj_count' is now equal to the number of objects stored in the EnvironmentBuilder.
        """
        return len(self.objects)

    def __iter__(self) -> Iterator[Polygon]:
        """
        Provides an iterator over the objects in the environment.

        Returns:
            Iterator[Object2D]: An iterator over the objects in the environment.

        Example:
            >>> builder = EnvironmentBuilder(xy_min=0, xy_max=10, xy_train_min=(2, 2), xy_train_max=(8, 8))
            >>> obj1 = Object2D(x=(0, 1, 1), y=(0, 1, 0))
            >>> obj2 = Object2D(x=(2, 3, 3, 2), y=(2, 2, 3, 3))

            >>> builder.add_object(obj1, obj2)
            >>> for obj in builder:
            >>>     print(obj)
            >>> # Iterates through each object in the EnvironmentBuilder and prints them.
        """
        return iter(self.objects)

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

        Example:
            >>> builder1 = EnvironmentBuilder(xy_min=0, xy_max=10, xy_train_min=(2, 2), xy_train_max=(8, 8))
            >>> builder2 = EnvironmentBuilder(xy_min=5, xy_max=15, xy_train_min=(7, 7), xy_train_max=(13, 13))

            >>> obj1 = Object2D(x=(0, 1, 1), y=(0, 1, 0))
            >>> obj2 = Object2D(x=(2, 3, 3, 2), y=(2, 2, 3, 3))

            >>> builder1.add_object(obj1).set_textures(2).set_polygons(3)
            >>> builder2.add_object(obj2)

            >>> merged_builder = builder1 + builder2
            >>> # 'merged_builder' contains combined objects and properties from 'builder1' and 'builder2'.
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

        # plot objects
        for obj in self.objects:
            plot_polygon(obj, ax=ax)

        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
        ax.grid()

        return fig


class VisiblePlane:
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
    def polygon(self) -> TexuredPolygon:
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


class Boundary(Area):
    """
    A class representing a boundary, which is a specific type of area with visible parts.

    Attributes:
        visible_parts (np.ndarray): The visible parts of the boundary.
    """
    def __init__(
        self,
        polygon: TexuredPolygon,
        points: np.ndarray,
        visible_parts: np.ndarray,
    ):
        """
        Initializes the Boundary with a polygon, a set of points, and visible parts.

        Args:
            polygon (TexuredPolygon): The polygon defining the boundary.
            points (np.ndarray): The set of points within the boundary.
            visible_parts (np.ndarray): The visible parts of the boundary.
        """
        super().__init__(polygon, points)
        self._visible_parts = visible_parts

    @property
    def visible_parts(self) -> np.ndarray:
        """
        Returns the visible parts of the boundary.

        Returns:
            np.ndarray: The visible parts of the boundary.
        """
        return self._visible_parts


@dataclass
class Environment(WritablePickle):
    """
    A data class representing an environment with a room, visible area, objects, and a visible plane.

    Attributes:
        room (Area): The room in the environment.
        visible_area (Area): The visible area in the environment.
        objects (list[Boundary]): The list of objects in the environment.
        visible_plane (VisiblePlane): The visible boundaries from every point of the environment.
    """
    room: Area
    visible_area: Area
    objects: list[Boundary]
    visible_plane: VisiblePlane


class EnvironmentCompiler:
    """
    A class that compiles an environment using an EnvironmentBuilder.

    Attributes:
        builder (EnvironmentBuilder): The builder used to compile the environment.
    """
    def __init__(self, builder: EnvironmentBuilder):
        """
        Initializes the EnvironmentCompiler with an EnvironmentBuilder.

        Args:
            builder (EnvironmentBuilder): The builder used to compile the environment.
        """
        self._builder = builder

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
        from_: tuple[int, int],
        to: tuple[int, int]
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
            *create_cartesian_space(
                from_, to, self.builder.res
            ),
            res=self.builder.res
        )

    def compile_room_points(
        self,
        space_points: list[Point]
    ) -> list[Point]:
        """
        Compiles the room points from a list of space points.

        Args:
            space_points (list[Point]): The space points.

        Returns:
            list[Point]: The compiled room points.
        """
        prepared = splp.prep(spl.GeometryCollection([poly.obj for poly in self.builder.objects]))
        return list(filter(prepared.disjoint, space_points))

    def compile_visible_area_points(
        self,
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

    def compile_boundary_points(
        self,
        space_points: list[list[Point]],
    ) -> list[Point]:
        """
        Compiles the boundary points from a list of space points.

        Args:
            space_points (list[list[Point]]): The space points.

        Returns:
            list[Point]: The compiled boundary points.
        """
        prepared_objects_boundaries = [
            splp.prep(obj.boundary)
            for obj in self.builder.objects
        ]
        return [
            list(filter(prepared.crosses, space_points))
            for prepared in prepared_objects_boundaries
        ]

    def align_object_boundaries(
        self,
        boundary_points: np.ndarray
    ) -> list[np.ndarray]:
        """
        Aligns the object boundaries with the resolution of the space grid.

        Args:
            boundary_points (np.ndarray): The boundary points.

        Returns:
            list[np.ndarray]: The aligned object boundaries.
        """
        # points of space are centers of circles of r=0.5*res.
        # These centers sometimes are not consistent with boundaries of original objects.
        # So we need to make correction to be consistent with resolution
        object_matrices = [
            np.array(obj.exterior.coords.xy).T
            for obj in self.builder.objects
        ]
        return [
            find_closest_points(boundary_points, object_matrix)
            for object_matrix in object_matrices
        ]

    @staticmethod
    def compile_visible_plane(
        starting_points: np.ndarray,
        directions: np.ndarray,
        room_points_coordinates: np.ndarray,
        boundary_points_coordinates: list[np.ndarray] # list of boundary points coordinates for each object
    ):
        """
        Compiles the visible plane from starting points, directions, room points coordinates, and boundary points coordinates.

        Args:
            starting_points (np.ndarray): The starting points.
            directions (np.ndarray): The directions.
            room_points_coordinates (np.ndarray): The room points coordinates.
            boundary_points_coordinates (list[np.ndarray]): The boundary points coordinates for each object.

        Returns:
            VisiblePlane: The compiled visible plane.
        """
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

        return VisiblePlane(visible_xy, slices)

    @staticmethod
    def compile_directions(objects: list[Polygon]) -> tuple[np.ndarray, np.ndarray]:
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

        return np.concatenate(starting_points), np.concatenate(directions)

    def compile(self) -> Environment:
        """
        Compiles the environment.

        Returns:
            Environment: The compiled environment.
        """
        room_area = self.compile_room_area()
        visible_area = self.compile_visible_area()

        space_points = self.compile_space_points(
            *regroup_min_max(*visible_area.bounds)
        )

        visible_space_points = self.compile_room_points(space_points)
        visible_space_points_coordinates = np.array([
            [point.centroid.xy[0][0], point.centroid.xy[1][0]]
            for point in visible_space_points
        ])

        boundary_points = self.compile_boundary_points(space_points)
        boundary_points_coordinates = [
            np.array([[point.centroid.xy[0][0], point.centroid.xy[1][0]] for point in boundary_point])
            for boundary_point in boundary_points
        ]

        objects_matrices_corrected = self.align_object_boundaries(
            np.concatenate(boundary_points_coordinates)
        )
        objects_corrected = [ # redefine objects according to space grid correction
            TexuredPolygon(
                zip(
                    obj_coords[:, 0],
                    obj_coords[:, 1]
                ),
                texture=obj.texture
            )
            for obj, obj_coords in zip(self.builder.objects, objects_matrices_corrected)
        ]

        starting_points, directions = self.compile_directions(objects_corrected)
        visible_plane = self.compile_visible_plane(
            starting_points,
            directions,
            visible_space_points_coordinates,
            boundary_points_coordinates
        )
        return Environment(
            Area(room_area),
            Area(visible_area, visible_space_points_coordinates),
            [
                Boundary(
                    obj,
                    boundary_points_coordinates[i],
                    visible_plane[i]
                )
                for i, obj in enumerate(objects_corrected)
            ],
            visible_plane
        )
