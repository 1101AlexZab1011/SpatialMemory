import configparser
import numpy as np
from dataclasses import dataclass
from ..data.configparser import EvalConfigParser
from dataclasses import dataclass

@dataclass
class GeometryParams:
    """
    A data class that stores various geometric parameters for a simulation or application.

    Attributes:
        max_xy (int): The maximum value for x and y coordinates in the geometric space.
        min_xy (int): The minimum value for x and y coordinates in the geometric space.
        min_train_x (int): The minimum value for x-coordinate during training.
        min_train_y (int): The minimum value for y-coordinate during training.
        max_train_x (int): The maximum value for x-coordinate during training.
        max_train_y (int): The maximum value for y-coordinate during training.
        max_n_obj_points (int): The maximum number of points an object can have.
        n_objects (int): The total number of objects in the simulation or application.
        n_polygons (int): The total number of polygons in the simulation or application.
        n_vertices (List[int]): A list of integers representing the number of vertices for each polygon.
        objects_x (List[List[float]]): A list of lists representing x-coordinates of objects' vertices.
        objects_y (List[List[float]]): A list of lists representing y-coordinates of objects' vertices.

    Note:
        - The attributes `objects_x` and `objects_y` store lists of lists to represent the x and y coordinates
          of vertices for multiple objects. Each inner list corresponds to the vertices of a single object.
        - The `n_vertices` list should have the same length as `n_polygons`, and each element represents
          the number of vertices for a corresponding polygon.

    Example:
        geometry_params = GeometryParams(
            max_xy=100,
            min_xy=0,
            min_train_x=10,
            min_train_y=10,
            max_train_x=90,
            max_train_y=90,
            max_n_obj_points=5,
            n_objects=3,
            n_polygons=2,
            n_vertices=[3, 4],
            objects_x=[[20, 30, 40], [60, 70, 80, 90], [10, 20, 30]],
            objects_y=[[15, 25, 35], [65, 75, 85, 95], [15, 25, 35]]
        )
    """
    max_xy: int
    min_xy: int
    min_train_x: int
    min_train_y: int
    max_train_x: int
    max_train_y: int
    max_n_obj_points: int
    n_objects: int
    n_polygons: int
    n_vertices: list[int]
    objects_x: list[list[float]]
    objects_y: list[list[float]]


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
    if bool(config.get('ExternalSources', 'variables')) and not any(['globals' in kwargs, 'locals' in kwargs]):
        raise ValueError(
            f'Parser requires external sources that has not been provided: '
            f'{", ".join([variable  + " " + str(value) for variable, value in config.eval("ExternalSources", "variables").items()])}'
        )

    if isinstance(config, str):
        cfg = EvalConfigParser(interpolation=configparser.ExtendedInterpolation(), allow_no_value=True)
        cfg.read(config)
        config = cfg

    max_xy, min_xy, min_train_x, min_train_y, max_train_x, max_train_y = get_coords(config, *args, **kwargs)
    n_objects, n_polygons, max_n_obj_points = get_building(config, *args, **kwargs)

    n_vertices, object_x, object_y = get_objects(config, n_objects, max_n_obj_points, *args, **kwargs)

    return GeometryParams(
        max_xy,
        min_xy,
        min_train_x,
        min_train_y,
        max_train_x,
        max_train_y,
        max_n_obj_points,
        n_objects,
        n_polygons,
        n_vertices,
        object_x,
        object_y,
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
    geometry.objects_x = [[scale * x for x in row] for row in geometry.objects_x]
    geometry.objects_y = [[scale * y for y in row] for row in geometry.objects_y]

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

