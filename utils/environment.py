import configparser
from typing import Any, Callable, Optional

import numpy as np
from bbtoolkit.data.configparser import EvalConfigParser
from bbtoolkit.preprocessing.environment import AbstractBuildingGeometryProcessor, GeometryFactory, GeometryParams, TrainingSpace, get_geometry_params


class StandardBuildingGeometryProcessor(AbstractBuildingGeometryProcessor):
    """
    A concrete implementation of the Building Geometry Processor that defines line identities for standard geometries.

    This class provides a line identity determination specific to certain standard building geometries.
    It subclasses `AbstractBuildingGeometryProcessor` and customizes the `get_line_identity` method
    based on the geometry name.

    Attributes:
        geometry_name (str): The name of the standard building geometry.

    Args:
        geometry_name (str): The name of the standard building geometry. Must be one of:
            - 'two_room'
            - 'squared_room'
            - 'inserted_barrier'
            - 'preplay_env_open'
            - 'preplay_env_closed'

    Raises:
        NotImplementedError: If an invalid geometry name is provided.

    Methods:
        get_line_identity(poly: int, xf: float, xi: float) -> int:
            Determine the identity of a line texture within a building polygon based on the geometry.

    Example:
        >>> processor = StandartBuildingGeometryProcessor('two_room')
        >>> geometry = GeometryParams(...)  # Define geometry parameters
        >>> complex_grid = get_complex_grid(geometry, res=1.0)  # Generate a complex grid.
        >>> foreground_pts, line_tex, dir_, r0 = processor(geometry, complex_grid)
    """
    def __init__(self, geometry_name: str):
        """
        Initialize the StandartBuildingGeometryProcessor with a specific standard building geometry.

        Args:
            geometry_name (str): The name of the standard building geometry.
        """
        if geometry_name not in (
            'two_room',
            'squared_room',
            'inserted_barrier',
            'preplay_env_open',
            'preplay_env_closed'
        ):
            raise NotImplementedError(f'Invalid geometry name: {geometry_name}')

        self.geometry_name = geometry_name

    def get_line_identity(self, poly: int, xf: float, xi: float) -> int:
        """
        Determine the identity of a line texture within a building polygon based on the standard geometry.

        Args:
            poly (int): The index of the building polygon.
            xf (float): The x-coordinate of the ending point of the line segment.
            xi (float): The x-coordinate of the starting point of the line segment.

        Returns:
            int: The identity of the line texture based on the standard geometry.
        """
        match self.geometry_name:
            case 'two_room':
                if poly >= 3 or (poly == 2 and xf > 0 and xi > 0):
                    return poly + 1
                else:
                    return poly
            case 'squared_room' | 'preplay_env_open' | 'preplay_env_closed':
                return poly
            case 'inserted_barrier':
                if poly >= 5 or (poly == 5 and (xf == 14 and xi == 8 or xf == 8 and xi == 14)):
                    return poly + 1
                else:
                    return poly


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
    geometry.objects.x *= scale
    geometry.objects.y *= scale

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
    x_range = config.eval('GridBoundaries', 'max_x', locals={'red_grid': red_grid})
    x_barrier_top_min = (red_grid[x_range // 2, :] == 1).argmax() + 1
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


def get_geometry_by_name(cfg_path: str, geometry_name: str) -> tuple[GeometryParams, int]:
    """
    Get a geometry and the number of textures by name.

    This function returns a `GeometryParams` object representing a specific geometry configuration and the number of
    textures associated with that geometry. The `geometry_name` parameter specifies the desired geometry, and the
    function retrieves the corresponding geometry parameters and texture count accordingly.

    Args:
        cfg_path (str): The path to the configuration file.
        geometry_name (str): The name of the geometry to retrieve. Supported geometry names include:
            - 'two_room'
            - 'squared_room'
            - 'inserted_barrier'
            - 'preplay_env_open'
            - 'preplay_env_closed'

    Returns:
        tuple: A tuple containing the following items:
            - geometry (GeometryParams): A `GeometryParams` object representing the retrieved geometry.
            - n_textures (int): The number of textures associated with the geometry.

    Raises:
        ValueError: If an unsupported or invalid geometry name is provided.

    Example:
        >>> config_path = 'config.ini'
        >>> geometry_name = 'two_room'
        >>> geometry, n_textures = get_geometry_by_name(config_path, geometry_name)
        >>> print(f'Geometry: {geometry_name}')
        >>> print(f'Number of Textures: {n_textures}')
    """

    match geometry_name:
        case 'two_room':
            geometry = get_two_room(cfg_path)
        case 'squared_room':
            geometry = get_geometry_params(cfg_path)
        case 'inserted_barrier':
            geometry = get_geometry_params(cfg_path)
        case 'preplay_env_closed' | 'preplay_env_open':
            geometry = get_preplay_env(cfg_path)
        case _:
            raise ValueError(f"Unsupported geometry name: {geometry_name}. Available geometries are: 'two_room', 'squared_room', 'inserted_barrier', 'preplay_env_open', 'preplay_env_closed'")

    return geometry


class StandardGeometryFactory(GeometryFactory):
    """
    A factory class for creating instances of the `Geometry` class based on configuration and processing functions.
    This class is a wrapper for its parent with default arguments

    Attributes:
        cfg_path (str): The file path to the configuration used to create the geometry.
        geometry_getter (Callable): A callable function that retrieves geometric parameters and the number of textures.
        building_geometry_processor (Callable): A callable function that processes geometric parameters into a training space.

    Methods:
        __call__(self, getter_kwargs: dict[str, Any] = None, building_processor_kwargs: dict[str, Any] = None) -> Geometry:
            Create and return a `Geometry` instance based on the provided configuration and processing functions.

    Example:
        >>> factory = GeometryFactory(
        >>>     cfg_path="geometry_config.ini",
        >>>     geometry_getter=get_geometry_by_name,
        >>>     building_geometry_processor=StandartBuildingGeometryProcessor,
        >>> )
    """
    def __init__(
        self,
        cfg: str | configparser.ConfigParser,
        geometry_getter: Callable[[tuple[Any, ...]], tuple[GeometryParams, int]] = None,
        building_geometry_processor: Callable[
            [
                GeometryParams,
                float,
                AbstractBuildingGeometryProcessor,
                Optional[tuple[Any, ...]],
                Optional[dict[str, Any]]
            ], TrainingSpace
        ] = None,
    ):
        """
        Initialize the DefaultGeometryFactory.

        Args:
            cfg (str | configparser.ConfigParser): Config or the path to the configuration file used to create the geometry.
            geometry_getter (Callable, optional): A function that retrieves geometric parameters and the number of textures based on the configuration file (default is get_geometry_by_name).
            building_geometry_processor (Callable, optional): A function that processes geometric parameters into a training space (default is StandartBuildingGeometryProcessor).
        """
        if geometry_getter is None:
            geometry_getter = get_geometry_by_name
        if building_geometry_processor is None:
            building_geometry_processor = StandardBuildingGeometryProcessor

        super().__init__(cfg, geometry_getter, building_geometry_processor)

    def __call__(self, geometry_name: str):
        """
        Create and return a `Geometry` instance based on the provided configuration and processing functions.

        Args:
            geometry_name (str): The name of the standard building geometry. Must be one of:
            - 'two_room'
            - 'squared_room'
            - 'inserted_barrier'
            - 'preplay_env_open'
            - 'preplay_env_closed'
        Returns:
            Geometry: A `Geometry` instance representing the processed geometry.
        """
        return super().__call__(
            dict(geometry_name=geometry_name),
            dict(geometry_name=geometry_name)
        )
