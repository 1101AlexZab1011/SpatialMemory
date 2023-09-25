import configparser
import numpy as np
from dataclasses import dataclass
from ..data.configparser import EvalConfigParser
from dataclasses import dataclass
import pandas as pd
import torch
from abc import ABC, abstractmethod
from ..math.geometry import inpolygon
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

    Raises:
        ValueError: If external variables are required but not provided, it raises an error with details of missing variables.

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

    if bool(config.get('ExternalSources', 'variables')) and not any(['globals' in kwargs, 'locals' in kwargs]):
        raise ValueError(
            f'Parser requires external sources that has not been provided: '
            f'{", ".join([variable  + " " + str(value) for variable, value in config.eval("ExternalSources", "variables").items()])}'
        )

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


def get_two_room(cfg_path: str = '../cfg/envs/two_room.ini') -> GeometryParams:
    """
    Create a `GeometryParams` object representing a two-room environment based on a configuration file.

    Args:
        cfg_path (str, optional): The path to the configuration file containing environment parameters.
            Defaults to '../cfg/envs/two_room.ini'.

    Returns:
        GeometryParams: An instance of the `GeometryParams` data class representing the two-room environment.

    Note:
        - After retrieving geometry parameters, this function scales the objects' coordinates based on a scaling factor
          specified in the configuration file.
    """
    config = EvalConfigParser(interpolation=configparser.ExtendedInterpolation(), allow_no_value=True)
    config.read(cfg_path)
    geometry = get_geometry_params(config)

    # Scale the objects to full width after coords were between -10 and 10
    scale = config.eval('BuildingBoundaries', 'scale')
    geometry.objects.x = [[scale * x for x in row] for row in geometry.objects.x]
    geometry.objects.y = [[scale * y for y in row] for row in geometry.objects.y]

    return geometry


def get_preplay_env(preplay_env_closed_cfg_path: str) -> GeometryParams:
    """
    Create a `GeometryParams` object representing a pre-play environment based on a configuration file.

    Args:
        preplay_env_closed_cfg_path (str): The path to the configuration file containing pre-play environment parameters.

    Returns:
        GeometryParams: An instance of the `GeometryParams` data class representing the pre-play environment.

    Note:
        - It utilizes a numpy array (red_grid) loaded from an external data source to calculate certain environment parameters.
        - The calculated parameters include the y_range, x_barrier_top_min, y_barrier_top, x_barrier_bot_min, and y_barrier_bot.
        - These calculated parameters are passed to `get_geometry_params` as local variables to construct the `GeometryParams` object.
    """
    config = EvalConfigParser(interpolation=configparser.ExtendedInterpolation(), allow_no_value=True)
    config.read(preplay_env_closed_cfg_path)
    datapath = config.eval('ExternalSources', 'paths')
    red_grid = np.load(datapath)[2:-2, 2:-2]
    y_range = config.eval('RoomDimensions', 'y_range', locals={'red_grid': red_grid})
    x_barrier_top_min = (red_grid[y_range // 2, :] == 1).argmax() + 1
    y_barrier_top = np.where(red_grid[:, -1] == 1)[0][-1] + 1
    x_barrier_bot_min = x_barrier_top_min
    y_barrier_bot = np.where(red_grid[:, -1] == 1)[0][0] + 1

    return get_geometry_params(
        config,
        locals={
            'red_grid': red_grid,
            'x_barrier_bot_min': x_barrier_bot_min,
            'x_barrier_top_min': x_barrier_top_min,
            'y_barrier_bot': y_barrier_bot,
            'y_barrier_top': y_barrier_top,
        }
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
class TrainingSpace:
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
class Boundary:
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
