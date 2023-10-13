import configparser
from typing import Any, Callable, Optional
import numpy as np
from dataclasses import dataclass

from bbtoolkit.data import WritablePickle, read_pkl, save_pkl
from ..data.configparser import EvalConfigParser, validate_config_eval
from dataclasses import dataclass
import pandas as pd
import torch
from abc import ABC, abstractmethod
from ..math.geometry import compute_intersection, inpolygon
import matplotlib.pyplot as plt


@dataclass
class Coordinates2D:
    """
    A data class for storing x and y coordinates as numpy arrays.

    Attributes:
        x (np.ndarray): A numpy array containing x-coordinates.
        y (np.ndarray): A numpy array containing y-coordinates.

    Raises:
        ValueError: If the shapes of x and y arrays do not match during object initialization.

    Example:
        coordinates = Coordinates(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([4.0, 5.0, 6.0])
        )

        You can access the x and y coordinates using `coordinates.x` and `coordinates.y` respectively.
    """
    x: int | float | np.ndarray | pd.DataFrame | torch.Tensor
    y: int | float | np.ndarray | pd.DataFrame | torch.Tensor

    def __post_init__(self):
        """
        Ensure that x and y arrays have the same type (and shape) after object initialization.
        """
        if type(self.x) != type(self.y):
            raise ValueError(f'x and y must have the same type, got {type(self.x)} and {type(self.y)} instead')
        if hasattr(self.x, 'shape') and (self.x.shape != self.y.shape):
            raise ValueError(f'x and y must have the same shape, got {self.x.shape} and {self.y.shape} instead')

@dataclass
class GeometryParams:
    """
    A data class that stores various geometric parameters for a simulation or application.

    Attributes:
        max_xy (int): The maximum value for x and y coordinates in the geometric space.
        min_xy (int): The minimum value for x and y coordinates in the geometric space.
        min_train (Coordinates2D): The minimum coordinate values during training.
        max_train (Coordinates2D): The maximum coordinate values during training.
        max_n_obj_points (int): The maximum number of points an object can have.
        n_objects (int): The total number of objects in the simulation or application.
        n_polygons (int): The total number of polygons in the simulation or application.
        n_vertices (List[int]): A list of integers representing the number of vertices for each polygon.
        objects (Coordinates2D): Coordinates of objects' vertices.

    Note:
        - The attributes `objects.x` and `objects.y` store lists of lists to represent the x and y coordinates
          of vertices for multiple objects. Each inner list corresponds to the vertices of a single object.
        - The `n_vertices` list should have the same length as `n_polygons`, and each element represents
          the number of vertices for a corresponding polygon.

    Example:
        geometry_params = GeometryParams(
            max_xy=100,
            min_xy=0,
            min_train=Coordinates(x=np.array([10]), y=np.array([10])),
            max_train=Coordinates(x=np.array([90]), y=np.array([90])),
            max_n_obj_points=5,
            n_objects=3,
            n_polygons=2,
            n_vertices=[3, 4],
            objects=Coordinates2D(
                x=[
                    [20, 30, 40],
                    [60, 70, 80, 90],
                    [10, 20, 30]
                ],
                y=[
                    [15, 25, 35],
                    [65, 75, 85, 95],
                    [15, 25, 35]
                ]
            )
        )
    """
    max_xy: int
    min_xy: int
    min_train: Coordinates2D
    max_train: Coordinates2D
    max_n_obj_points: int
    n_objects: int
    n_polygons: int
    n_vertices: list[int]
    objects: Coordinates2D


def get_objects(
    config: configparser.ConfigParser,
    n_objects: int,
    max_n_obj_points: int,
    *args, **kwargs
) -> tuple[list[int], list[list[float]], list[list[float]]]:
    """
    Retrieve object information from a configuration file and return relevant data.

    Args:
        config (configparser.ConfigParser): The configuration parser containing object data.
        n_objects (int): The total number of objects to retrieve.
        max_n_obj_points (int): The maximum number of points an object can have.
        *args, **kwargs: Additional arguments and keyword arguments to be passed to `config.eval()`.

    Returns:
        Tuple[List[int], List[List[float]], List[List[float]]]: A tuple containing:
            - A list of integers representing the number of vertices for each object.
            - A list of lists representing x-coordinates of objects' vertices.
            - A list of lists representing y-coordinates of objects' vertices.

    Note:
        - The `config` parameter should be a `configparser.ConfigParser` object configured with the necessary
          sections and keys for object data.
        - Each object's information is retrieved from a separate section in the configuration file named 'Object1',
          'Object2', and so on, up to 'Object{n_objects}'.
        - The 'n_vertices' key in each section specifies the number of vertices for that object.
        - The 'object_x' and 'object_y' keys in each section should return lists of x and y coordinates of
          the object's vertices, respectively.
        - If an object has fewer vertices than `max_n_obj_points`, the remaining points in the respective
          `object_x` and `object_y` lists will be filled with zeros.

    Example:
        Assuming a configuration file contains the following sections and keys:
        [Object1]
        n_vertices = 3
        object_x = [10.0, 20.0, 30.0]
        object_y = [15.0, 25.0, 35.0]

        [Object2]
        n_vertices = 4
        object_x = [40.0, 50.0, 60.0, 70.0]
        object_y = [45.0, 55.0, 65.0, 75.0]

        The function call:
        n_vertices, object_x, object_y = get_objects(config, n_objects=2, max_n_obj_points=5)

        Would return:
        n_vertices = [3, 4]
        object_x = [[10.0, 20.0, 30.0, 0.0, 0.0], [40.0, 50.0, 60.0, 70.0, 0.0]]
        object_y = [[15.0, 25.0, 35.0, 0.0, 0.0], [45.0, 55.0, 65.0, 75.0, 0.0]]
    """
    object_x = [[0.0] * max_n_obj_points for _ in range(n_objects)]
    object_y = [[0.0] * max_n_obj_points for _ in range(n_objects)]
    n_vertices = [0] * n_objects

    for i in range(n_objects):
        section_name = f'Object{i + 1}'
        n_vertices_i = config.eval(section_name, 'n_vertices')
        n_vertices[i] = n_vertices_i

        object_x_i = config.eval(section_name, 'object_x', *args, **kwargs)
        object_y_i = config.eval(section_name, 'object_y', *args, **kwargs)

        object_x[i][:n_vertices_i] = object_x_i
        object_y[i][:n_vertices_i] = object_y_i

    return n_vertices, object_x, object_y

def get_coords(config: configparser.ConfigParser, *args, **kwargs) -> tuple[int, int, int, int, int, int]:
    """
    Retrieve coordinates and boundaries from a configuration file.

    Args:
        config (configparser.ConfigParser): The configuration parser containing coordinate and boundary data.
        *args, **kwargs: Additional arguments and keyword arguments to be passed to `config.eval()`.

    Returns:
        Tuple[int, int, int, int, int, int]: A tuple containing the following coordinates and boundaries:
            - max_xy (int): The maximum value for x and y coordinates in the geometric space.
            - min_xy (int): The minimum value for x and y coordinates in the geometric space.
            - min_train_x (int): The minimum value for x-coordinate during training.
            - min_train_y (int): The minimum value for y-coordinate during training.
            - max_train_x (int): The maximum value for x-coordinate during training.
            - max_train_y (int): The maximum value for y-coordinate during training.

    Note:
        - The `config` parameter should be a `configparser.ConfigParser` object configured with the necessary
          sections and keys for coordinate and boundary data.
        - Coordinates and boundaries are retrieved from specific sections and keys in the configuration file.
        - The 'GridBoundaries' section is used to obtain `max_xy` and `min_xy` values.
        - The 'TrainingRectangle' section is used to obtain `min_train_x`, `min_train_y`, `max_train_x`,
          and `max_train_y` values.

    Example:
        Assuming a configuration file contains the following sections and keys:
        [GridBoundaries]
        max_xy = 100
        min_xy = 0

        [TrainingRectangle]
        min_train_x = 10
        min_train_y = 20
        max_train_x = 90
        max_train_y = 80

        The function call:
        max_xy, min_xy, min_train_x, min_train_y, max_train_x, max_train_y = get_coords(config)

        Would return:
        max_xy = 100
        min_xy = 0
        min_train_x = 10
        min_train_y = 20
        max_train_x = 90
        max_train_y = 80
    """
    max_xy = config.eval('GridBoundaries', 'max_xy', *args, **kwargs)
    min_xy = config.eval('GridBoundaries', 'min_xy', *args, **kwargs)

    min_train_x = config.eval('TrainingRectangle', 'min_train_x', *args, **kwargs)
    min_train_y = config.eval('TrainingRectangle', 'min_train_y', *args, **kwargs)
    max_train_x = config.eval('TrainingRectangle', 'max_train_x', *args, **kwargs)
    max_train_y = config.eval('TrainingRectangle', 'max_train_y', *args, **kwargs)

    return max_xy, min_xy, min_train_x, min_train_y, max_train_x, max_train_y

def get_building(config: configparser.ConfigParser, *args, **kwargs) -> tuple[int, int]:
    """
    Retrieve objects-related parameters from a configuration file.

    Args:
        config (configparser.ConfigParser): The configuration parser containing building-related parameters.
        *args, **kwargs: Additional arguments and keyword arguments to be passed to `config.eval()`.

    Returns:
        Tuple[int, int]: A tuple containing the following building-related parameters:
            - n_objects (int): The total number of objects or structures in the building.
            - n_polygons (int): The total number of polygons or shapes used in the building.
            - max_n_obj_points (int): The maximum number of points a building object can have.

    Note:
        - The `config` parameter should be a `configparser.ConfigParser` object configured with the necessary
          sections and keys for building-related parameters.
        - The values are retrieved from the 'BuildingBoundaries' section in the configuration file.
        - 'n_objects' represents the total number of objects or structures in the building.
        - 'n_polygons' represents the total number of polygons or shapes used in the building.
        - 'max_n_obj_points' specifies the maximum number of points an individual building object can have.

    Example:
        Assuming a configuration file contains the following section and keys:
        [BuildingBoundaries]
        n_objects = 5
        n_polygons = 10
        max_n_obj_points = 6

        The function call:
        n_objects, n_polygons, max_n_obj_points = get_building(config)

        Would return:
        n_objects = 5
        n_polygons = 10
        max_n_obj_points = 6
    """
    n_objects = config.eval('BuildingBoundaries', 'n_objects', *args, **kwargs)
    n_polygons = config.eval('BuildingBoundaries', 'n_polygons', *args, **kwargs)
    max_n_obj_points = config.eval('BuildingBoundaries', 'max_n_obj_points', *args, **kwargs)

    return n_objects, n_polygons, max_n_obj_points

def get_geometry_params(config: str | configparser.ConfigParser, *args, **kwargs) -> GeometryParams:
    """
    Retrieve geometry parameters from a configuration file or parser and return them as a `GeometryParams` object.

    Args:
        config (Union[str, configparser.ConfigParser]): Either a path to a configuration file (str) or
            a pre-configured `configparser.ConfigParser` object containing geometry-related data.
        *args, **kwargs: Additional arguments and keyword arguments to be passed to `config.eval()`.

    Returns:
        GeometryParams: An instance of the `GeometryParams` data class containing the retrieved geometry parameters.

    Note:
        - This function orchestrates the retrieval of various geometry parameters using other functions
          (`get_coords`, `get_building`, and `get_objects`).
        - External variables, if required, can be provided via the `kwargs` dictionary under 'globals' or 'locals'.
        - The returned `GeometryParams` object contains all the geometry-related parameters required for simulation
          or application.
    """

    if isinstance(config, str):
        cfg = EvalConfigParser(interpolation=configparser.ExtendedInterpolation(), allow_no_value=True)
        cfg.read(config)
        config = cfg

    validate_config_eval(config, **kwargs)

    max_xy, min_xy, min_train_x, min_train_y, max_train_x, max_train_y = get_coords(config, *args, **kwargs)
    n_objects, n_polygons, max_n_obj_points = get_building(config, *args, **kwargs)

    n_vertices, object_x, object_y = get_objects(config, n_objects, max_n_obj_points, *args, **kwargs)

    return GeometryParams(
        Coordinates2D(max_xy, min_xy),
        Coordinates2D(min_train_x, min_train_y),
        Coordinates2D(max_train_x, max_train_y),
        max_n_obj_points,
        n_objects,
        n_polygons,
        np.array(n_vertices),
        Coordinates2D(
            np.array(object_x),
            np.array(object_y)
        )
    )


def get_complex_grid(geometry: GeometryParams, res: float) -> np.ndarray:
    """
    Generate a complex grid of coordinates within the specified geometric boundaries.

    Args:
        geometry (GeometryParams): A `GeometryParams` object representing the geometric parameters of the environment.
        res (float): The resolution or spacing between grid points along both x and y axes.

    Returns:
        np.ndarray: A 1D array of complex numbers representing grid points within the specified geometric boundaries.

    Example:
        Given a `GeometryParams` object representing the following geometric parameters:
        - min_xy = 0
        - max_xy = 100

        The function call:
        complex_grid = get_complex_grid(geometry, res=1.0)

        Would return a 1D array of complex numbers representing grid points spaced 1 unit apart within the [0, 100] range.
    """
    min_xy, max_xy = geometry.min_xy, geometry.max_xy
    grid_x = np.arange(min_xy, max_xy + res, res)  # Create a Cartesian grid of possible locations over the environment along the x-axis
    grid_y = np.arange(min_xy, max_xy + res, res)  # Create a Cartesian grid of possible locations over the environment along the y-axis

    # Create 2D grids of x and y values
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    # Convert Cartesian coordinates to complex numbers
    complex_grid = grid_x + 1j * grid_y

    # Reshape the complex grid into a 1D vector of grid points (x and y values as complex numbers)
    complex_grid = complex_grid.reshape(-1, 1)

    return complex_grid, grid_x, grid_y


class AbstractBuildingGeometryProcessor(ABC):
    """
    An abstract base class for processing building geometry within a space.

    Subclasses of this class are expected to implement the `get_line_identity` method.

    Attributes:
        None

    Methods:
        get_line_identity(poly: int, xf: float, xi: float) -> int:
            An abstract method to determine the identity of a line texture within a building polygon.

    Usage:
        - Subclass this abstract class to create custom building geometry processors that handle line identities
          and other processing specific to a simulation.

    Example:
        class MyBuildingGeometryProcessor(AbstractBuildingGeometryProcessor):
            def get_line_identity(self, poly: int, xf: float, xi: float) -> int:
                # Implement custom logic to determine the identity of a line texture within a building polygon.
                if poly >= 5 or (poly == 5 and (xf == 14 and xi == 8 or xf == 8 and xi == 14)):
                    return poly + 1

            # Add additional methods and custom logic specific to your building geometry processing needs.
    """

    @abstractmethod
    def get_line_identity(self, poly: int, xf: float, xi: float) -> int:
        """
        Determine the identity of a line texture within a building polygon.

        Args:
            poly (int): The index of the building polygon.
            xf (float): The x-coordinate of the ending point of the line segment.
            xi (float): The x-coordinate of the starting point of the line segment.

        Returns:
            int: The identity of the line texture.

        Note:
            This method should be implemented in subclasses to provide custom logic for line identity determination.
        """
        pass

    def __call__(self, geometry: GeometryParams, complex_grid: np.ndarray):
        """
        Process building geometry and return relevant data.

        Args:
            geometry (GeometryParams): A `GeometryParams` object representing the geometric parameters of the environment.
            complex_grid (np.ndarray): A complex grid of coordinates within the specified geometric boundaries.

        Returns:
            tuple: A tuple containing the following arrays:
                - foreground_pts (np.ndarray): Locations inside buildings, considered foreground.
                - line_tex (np.ndarray): Identity of each line texture within buildings.
                - dir_ (np.ndarray): Direction vectors of each line within buildings.
                - r0 (np.ndarray): Starting points of each line within buildings.

        Example:
            processor = MyBuildingGeometryProcessor()  # Create an instance of a custom processor.
            complex_grid = get_complex_grid(geometry, res=1.0)  # Generate a complex grid.
            foreground_pts, line_tex, dir_, r0 = processor(geometry, complex_grid)
        """
        foreground_pts = []  # Will be the possible locations from above that are inside buildings
        line_tex = []  # Identity of each line
        dir_ = []  # Direction of each line
        r0 = []  # Starting point of each line
        for poly in range(1, geometry.n_polygons + 1):
            # Create complex vertices for the current polygon
            vertices = np.array(geometry.objects.x[poly - 1, :] + 1j * geometry.objects.y[poly - 1, :])
            vertices = vertices[:geometry.n_vertices[poly - 1]]

            # Find locations inside this building
            in_poly_pts = np.where(inpolygon(complex_grid.real, complex_grid.imag, vertices.real, -vertices.imag))

            # Locations inside this building, foreground in the sense of looking at the map from above
            # The ground is background, buildings are foreground
            foreground_pts.extend(in_poly_pts[0])

            # Loop over "lines" of the building (same as # of vertices)
            for polyline in range(geometry.n_vertices[poly - 1] - 1):
                xi, xf = geometry.objects.x.T[polyline:polyline + 2, poly - 1]
                yi, yf = geometry.objects.y.T[polyline:polyline + 2, poly - 1]
                line_tex.append(self.get_line_identity(poly, xf, xi))

                dir_.append([xf - xi, yf - yi, 0])  # Line vectors, from one vertex of a building to the next
                r0.append([xi, yi, 0])  # Line start

        return np.array(foreground_pts), np.array(line_tex), np.array(dir_), np.array(r0)


@dataclass
class AbstractSpace(ABC):
    """
    An abstract base class representing a 2D space with coordinates.

    Attributes:
    - coords (Coordinates2D): An instance of the Coordinates2D class representing the coordinates of points in the space.

    This abstract base class defines the basic structure for any 2D space representation. It requires subclasses to provide
    the 'coords' attribute, which should be an instance of the Coordinates2D class containing coordinate information.
    """
    coords: Coordinates2D


@dataclass
class TrainingSpace(AbstractSpace):
    """
    A data class representing a training space for a simulation.

    Attributes:
        coords (Coordinates2D): A `Coordinates2D` object containing 2D coordinates within the training space.
        identities (np.ndarray): An array containing identities of line textures.
        directions (np.ndarray): An array containing direction of lines (from one vertex of an object to the next).
        starting_points (np.ndarray): An array containing starting points for lines.
        resolution (float): The resolution or spacing of the space.

    Raises:
        ValueError: If the shapes of `directions` and `starting_points` do not match, or if the lengths of
                    `identities` and `directions` do not match during object initialization.

    Methods:
        plot(ax: plt.Axes = None, *args, **kwargs) -> plt.Figure | None:
            Plot the training space on a given `plt.Axes` object or create a new figure and axes for plotting.

    Example:
        training_coords = Coordinates2D(x=np.array([1.0, 2.0, 3.0]), y=np.array([4.0, 5.0, 6.0]))
        training_space = TrainingSpace(
            coords=training_coords,
            identities=np.array([0, 1, 2]),
            directions=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            starting_points=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            resolution=0.1
        )

        # Plot the training space with default settings (gray squares)
        training_space.plot()
    """
    coords: Coordinates2D
    identities: np.ndarray
    directions: np.ndarray
    starting_points: np.ndarray
    resolution: float

    def __post_init__(self):
        """
        Ensure that directions and starting_points have the same shape, and identities and directions have the same length.
        """
        if self.directions.shape != self.starting_points.shape:
            raise ValueError(
                'directions and starting_points must have the same shape, '
                f'got {self.directions.shape} and {self.starting_points.shape} instead'
            )
        if len(self.identities) != len(self.directions):
            raise ValueError(
                'identities and directions must have the same length, '
                f'got {len(self.identities)} and {len(self.directions)} instead'
            )
    def plot(self, ax: plt.Axes = None, *args, **kwargs) -> plt.Figure | None:
        """
        Plot the training space on a given `plt.Axes` object or create a new figure and axes for plotting.

        Args:
            ax (plt.Axes, optional): An existing `plt.Axes` object to plot on. If not provided, a new figure
                                     and axes will be created.
            *args: Positional arguments for the matplotlib.pyplot.plot function.
            **kwargs: Keyword arguments for the matplotlib.pyplot.plot function.

        Returns:
            plt.Figure | None: If an `ax` argument is provided, returns None. Otherwise, returns the created figure.

        Example:
            training_space.plot()  # Plot the training space with default settings (gray squares).

            # Customize the plot with additional arguments and keyword arguments.
            training_space.plot(marker='o', color='blue')
        """
        if not len(args) and not len(kwargs):
            args = 's',
            kwargs = dict(color='tab:gray')
        if ax is not None:
            ax.plot(self.coords.x, self.coords.y, *args, **kwargs)
        else:
            fig, ax = plt.subplots()
            ax.plot(self.coords.x, self.coords.y, *args, **kwargs)
            return fig


def process_training_space(
    geometry: GeometryParams,
    res: float,
    building_geometry_processor: AbstractBuildingGeometryProcessor,
    *args, **kwargs
) -> TrainingSpace:
    """
    Process the training space within the specified geometric boundaries and building geometry.

    Args:
        geometry (GeometryParams): A `GeometryParams` object representing the geometric parameters of the environment.
        res (float): The resolution or spacing between grid points along both x and y axes.
        building_geometry_processor (AbstractBuildingGeometryProcessor): An instance of a building geometry processor
                                                                      that implements texture identity determination.
        *args: Additional positional arguments to pass to the building geometry processor.
        **kwargs: Additional keyword arguments to pass to the building geometry processor.

    Returns:
        TrainingSpace: A `TrainingSpace` object representing the processed training space.
    """
    # Get the complex grid
    complex_grid, grid_x, grid_y = get_complex_grid(geometry, res)

    # Process building geometry
    building_geometry_processor = building_geometry_processor(*args, **kwargs)
    foreground_pts, line_tex, dir_, r0 = building_geometry_processor(geometry, complex_grid)

    # Create background_x and background_y based on grid_x and grid_y
    background_x = grid_x.copy()
    shape = background_x.shape
    background_x = background_x.T.reshape(-1)
    background_y = grid_y.copy()
    background_y = background_y.T.reshape(-1)

    # Convert the elements which lie within the buildings into non-numbers (NaN)
    background_x[foreground_pts] = np.nan
    background_y[foreground_pts] = np.nan

    # Reshape background_x and background_y
    background_x = background_x.reshape(shape[::-1]).T
    background_y = background_y.reshape(shape[::-1]).T

    # Remove the non-numbers (NaN) from the arrays - forms a column vector
    background_x = background_x.T[np.isfinite(background_x).T]
    background_y = background_y.T[np.isfinite(background_y).T]

    # Find the indices of locations outside of objects, but inside training rect.
    train_ind = np.where(
        (background_x > geometry.min_train.x) &
        (background_x < geometry.max_train.x) &
        (background_y > geometry.min_train.y) &
        (background_y < geometry.max_train.y)
    )[0]


    # Extract the coordinates within the specified range
    train_x = background_x[train_ind]
    train_y = background_y[train_ind]

    return TrainingSpace(Coordinates2D(train_x, train_y), line_tex, dir_, r0, res)


@dataclass
class Boundary(AbstractSpace):
    """
    A data class representing a boundary with associated coordinates and textures.

    Attributes:
        coords (Coordinates2D): A `Coordinates2D` object containing 2D coordinates defining the boundary.
        textures (np.ndarray): An array containing textures or labels associated with the boundary segments.

    Raises:
        ValueError: If the shapes of `coords.x` and `coords.y` do not match, or if the lengths of `coords.x`
                    and `textures` do not match during object initialization.

    Example:
        boundary_coords = Coordinates2D(x=np.array([1.0, 2.0, 3.0]), y=np.array([4.0, 5.0, 6.0]))
        boundary = Boundary(
            coords=boundary_coords,
            textures=np.array([0, 1, 2])
        )

        # You can access the boundary coordinates using `boundary.coords.x` and `boundary.coords.y`.
    """
    coords: Coordinates2D
    textures: np.ndarray

    def __post_init__(self):
        """
        Ensure that coords.x and coords.y have the same shape, and coords.x and textures have the same length.
        """
        if self.coords.x.shape != self.coords.y.shape:
            raise ValueError(f'x and y must have the same shape, got {self.coords.x.shape} and {self.coords.y.shape} instead')
        if len(self.coords.x) != len(self.textures):
            raise ValueError(f'coords and textures must have the same length, got {len(self.coords.x)} and {len(self.textures)} instead')


def process_boundary(training_space: TrainingSpace) -> Boundary:
    """
    Process the boundary within the specified training space.

    Args:
        training_space (TrainingSpace): A `TrainingSpace` object representing the training space with identified lines.

    Returns:
        Boundary: A `Boundary` object representing the processed boundary with coordinates and textures.

    Example:
        # Process the boundary within a training space
        boundary = process_boundary(training_space)

        # You can access the boundary coordinates using `boundary.points.x` and `boundary.points.y`.
    """
    total_lines = len(training_space.identities)

    boundary_len = np.linalg.norm(training_space.directions, axis=1)
    Dir_unit = training_space.directions / boundary_len[:, np.newaxis]
    boundary_len[np.where(np.isclose(boundary_len % training_space.resolution, 0))[0]] += training_space.resolution

    boundary_points_x = []
    boundary_points_y = []
    boundary_textures = []

    for boundary in range(total_lines):
        x = training_space.starting_points[boundary, 0] + np.arange(0, boundary_len[boundary], training_space.resolution) * Dir_unit[boundary, 0]
        y = training_space.starting_points[boundary, 1] + np.arange(0, boundary_len[boundary], training_space.resolution) * Dir_unit[boundary, 1]

        boundary_points_x.extend(x.tolist())
        boundary_points_y.extend(y.tolist())
        boundary_textures.extend(np.full(len(x), training_space.identities[boundary]))

    boundary_points_x = np.array(boundary_points_x)
    boundary_points_y = np.array(boundary_points_y)
    boundary_textures = np.array(boundary_textures)

    return Boundary(Coordinates2D(boundary_points_x, boundary_points_y), boundary_textures)


@dataclass
class VisiblePlane(Boundary):
    """
    A data class representing a visible plane associated with a boundary and training locations.

    Attributes:
        coords (Coordinates2D): A `Coordinates2D` object containing 2D coordinates defining the visible plane.
        textures (np.ndarray): An array containing textures or labels associated with the visible plane segments.
        training_locations (np.ndarray): An array containing the 2D training locations associated with the visible plane.

    Raises:
        ValueError: If the shapes of `coords.x` and `coords.y` do not match, or if the lengths of `coords.x`
                    and `textures` do not match during object initialization, or if the lengths of `coords.x` and
                    `training_locations` do not match during object initialization.

    Example:
        visible_plane_coords = Coordinates2D(x=np.array([1.0, 2.0, 3.0]), y=np.array([4.0, 5.0, 6.0]))
        training_locations = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        visible_plane = VisiblePlane(
            coords=visible_plane_coords,
            textures=np.array([0, 1, 2]),
            training_locations=training_locations
        )

        # You can access the visible plane coordinates using `visible_plane.coords.x` and `visible_plane.coords.y`.
    """
    training_locations: np.ndarray

    def __post_init__(self):
        """
        Ensure that coords.x and coords.y have the same shape, and coords.x and textures have the same length,
        and that coords.x and training_locations have the same length.
        """
        super().__post_init__()

        if len(self.coords.x) != len(self.training_locations):
            raise ValueError(f'coords and training_locations must have the same length, got {len(self.coords.x)} and {len(self.training_locations)} instead')


def process_visible_plane(boundary: Boundary, training_space: TrainingSpace) -> VisiblePlane:
    """
    Calculate the visible plane and texture for a set of training points relative to a boundary.

    Args:
        boundary (Boundary): An instance of the boundary class containing boundary information.
        training_space (TrainingSpace): An instance of the training space class containing training points information.

    Returns:
        A `VisiblePlane` object representing the processed visible plane with coordinates, textures and locations.

    This function calculates the visible plane and texture for a set of training points relative to a boundary.
    It considers occluded boundary points and accumulates visible points based on occlusion criteria.
    The result is returned as a visible_plane and texture array.
    """
    n_boundary_points = boundary.coords.x.shape[0]
    n_training_points = training_space.coords.x.shape[0]

    visible_plane = np.full((2, n_boundary_points, n_training_points), np.nan)
    texture = np.full((n_training_points, n_boundary_points), np.nan)

    training_locations = np.zeros((n_training_points, 2))
    occluded_points = np.zeros(n_boundary_points, dtype=bool)

    for location in range(n_training_points):
        pos = Coordinates2D(training_space.coords.x[location], training_space.coords.y[location])
        training_locations[location] = [pos.x, pos.y]

        local_r0 = training_space.starting_points - np.array([pos.x, pos.y, 0])
        Loc_bndry_pts = np.column_stack((boundary.coords.x - pos.x, boundary.coords.y - pos.y, np.zeros(n_boundary_points)))

        occluded_points.fill(False)

        for occ_bndry in range(len(training_space.identities)):
            alpha_pt, alpha_occ = compute_intersection(
                np.zeros((n_boundary_points, 3)),
                np.expand_dims(local_r0[occ_bndry], 0),
                Loc_bndry_pts,
                np.expand_dims(training_space.directions[occ_bndry], 0)
            )

            occluded_points |= (alpha_pt < 1 - 1e-5) & (alpha_pt > 0) & (alpha_occ <= 1) & (alpha_occ >= 0)

        unocc_ind = np.where(~occluded_points)[0]
        num_vis_pts = unocc_ind.size

        visible_plane[:, :num_vis_pts, location] = Loc_bndry_pts[unocc_ind, :2].T + np.array([pos.x, pos.y])[:, np.newaxis]
        texture[location, :num_vis_pts] = boundary.textures[unocc_ind].T

    visible_plane = Coordinates2D(visible_plane[0].T, visible_plane[1].T)

    return VisiblePlane(visible_plane, texture, training_locations)


def shuffle_visible_plane(visible_plane: VisiblePlane) -> VisiblePlane:
    """
    Shuffle the order of elements in a VisiblePlane object.

    This function shuffles the order of elements in a VisiblePlane object while maintaining
    the correspondence between coordinates, textures, and training locations.

    Parameters:
        visible_plane (VisiblePlane): The VisiblePlane object to shuffle.

    Returns:
        VisiblePlane: A new VisiblePlane object with shuffled elements.

    Example:
        original_plane = VisiblePlane(coords, textures, training_locations)
        shuffled_plane = shuffle_visible_plane(original_plane)
    """
    permutation = np.random.permutation(len(visible_plane.textures))
    return VisiblePlane(
        Coordinates2D(visible_plane.coords.x[permutation],
        visible_plane.coords.y[permutation]),
        visible_plane.textures[permutation],
        visible_plane.training_locations[permutation]
    )


@dataclass
class Geometry(WritablePickle):
    """
    A data class representing a geometry configuration for a simulation or modeling task.

    Attributes:
        params (GeometryParams): An instance of the `GeometryParams` class containing geometric parameters.
        n_textures (int): The number of textures or labels associated with the geometry.
        training_space (TrainingSpace): An instance of the `TrainingSpace` class defining the training space.
        boundary (Boundary): An instance of the `Boundary` class defining the boundary of the geometry.
        visible_plane (VisiblePlane): An instance of the `VisiblePlane` class representing the visible plane.

    Methods:
        save(self, path: str):
            Save the current geometry configuration to a specified file path using pickle serialization.

        load(path: str) -> Geometry:
            Load a geometry configuration from a specified file path using pickle deserialization and return it as a `Geometry` object.

        shuffle_visible_plane(self) -> VisiblePlane:
            Shuffle the visible plane associated with the geometry and return a new `VisiblePlane` object with shuffled data.
    """
    params: GeometryParams
    n_textures: int
    training_space: TrainingSpace
    boundary: Boundary
    visible_plane: VisiblePlane

    def shuffle_visible_plane(self) -> VisiblePlane:
        """
        Shuffle the visible plane associated with the geometry and return a new `VisiblePlane` object with shuffled data.

        Returns:
            VisiblePlane: A new `VisiblePlane` object with shuffled data.

        """
        return shuffle_visible_plane(self.visible_plane)


class GeometryFactory:
    """
    A factory class for creating instances of the `Geometry` class based on configuration and processing functions.

    Attributes:
        cfg_path (str): The file path to the configuration used to create the geometry.
        geometry_getter (Callable): A callable function that retrieves geometric parameters and the number of textures.
        building_geometry_processor (Callable): A callable function that processes geometric parameters into a training space.
        res (float): The resolution used for processing geometry data (default is 0.3).

    Methods:
        __call__(self, getter_kwargs: dict[str, Any] = None, building_processor_kwargs: dict[str, Any] = None) -> Geometry:
            Create and return a `Geometry` instance based on the provided configuration and processing functions.

    Example:
        factory = GeometryFactory(
            cfg_path="geometry_config.json",
            geometry_getter=get_geometry_params,
            building_geometry_processor=process_building_geometry,
            res=0.1
        )
    """
    def __init__(
        self,
        cfg_path: str,
        geometry_getter: Callable[[tuple[Any, ...]], tuple[GeometryParams, int]],
        building_geometry_processor: Callable[
            [
                GeometryParams,
                float,
                AbstractBuildingGeometryProcessor,
                Optional[tuple[Any, ...]],
                Optional[dict[str, Any]]
            ], TrainingSpace
        ],
        res: float = .3,
    ):
        """
        Initialize the GeometryFactory.

        Args:
            cfg_path (str): The file path to the configuration used to create the geometry.
            geometry_getter (Callable): A callable function that retrieves geometric parameters and the number of textures.
            building_geometry_processor (Callable): A callable function that processes geometric parameters into a training space.
            res (float): The resolution used for processing geometry data (default is 0.3).
        """
        self.cfg_path = cfg_path
        self.geometry_getter = geometry_getter
        self.building_geometry_processor = building_geometry_processor
        self.res = res

    def __call__(self, getter_kwargs: dict[str, Any]= None, building_processor_kwargs: dict[str, Any] = None):
        """
        Create and return a `Geometry` instance based on the provided configuration and processing functions.

        Args:
            getter_kwargs (dict[str, Any]): Additional keyword arguments to pass to the geometry_getter function (default is None).
            building_processor_kwargs (dict[str, Any]): Additional keyword arguments to pass to the building_geometry_processor function (default is None).

        Returns:
            Geometry: A `Geometry` instance representing the processed geometry.

        Example:
            geometry_instance = factory(
                getter_kwargs={"param1": value1, "param2": value2},
                building_processor_kwargs={"param3": value3}
            )
        """
        if getter_kwargs is None:
            getter_kwargs = {}
        geometry, n_textures = self.geometry_getter(self.cfg_path, **getter_kwargs)
        training_space = process_training_space(geometry, self.res, self.building_geometry_processor, **building_processor_kwargs)
        boundary = process_boundary(training_space)
        visible_plane = process_visible_plane(boundary, training_space)
        return Geometry(geometry, n_textures, training_space, boundary, visible_plane)
