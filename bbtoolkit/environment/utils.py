import numpy as np
from bbtoolkit.preprocessing.environment import Environment
from bbtoolkit.preprocessing.environment.builders import EnvironmentBuilder


def env2builder(environment: Environment) -> EnvironmentBuilder:
    """
    Converts an Environment instance into an EnvironmentBuilder instance.

    This function takes the boundary and visible area information from the Environment instance,
    along with the resolution parameter, to initialize an EnvironmentBuilder. It then transfers
    the wall and object polygon data to the builder.

    Args:
        environment (Environment): The environment instance to convert.

    Returns:
        EnvironmentBuilder: The builder instance created from the environment.
    """
    xy_min = np.min(environment.room.boundary.xy)
    xy_max = np.max(environment.room.boundary.xy)
    xy_train_min = tuple(np.min(environment.visible_area.boundary.xy, 1))
    xy_train_max = tuple(np.max(environment.visible_area.boundary.xy, 1))
    res = environment.params.res
    builder = EnvironmentBuilder(xy_min, xy_max, xy_train_min, xy_train_max, res)

    for wall in environment.walls:
        builder.add_wall(wall.polygon)
    for obj in environment.objects:
        builder.add_object(obj.polygon)

    return builder