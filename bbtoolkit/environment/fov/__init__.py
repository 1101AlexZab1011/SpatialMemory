from bbtoolkit.environment import Environment
from bbtoolkit.environment.fov.utils import get_fov, points_within_angles


class FOVManager:
    """
    Manages the field of view (FOV) within an environment.

    This class is responsible for calculating which objects and walls are visible from a given position
    and direction within the environment, considering the specified FOV.

    Attributes:
        environment (Environment): The environment in which the FOV calculations take place.
        fov (float): The field of view angle in radians.

    """
    def __init__(
        self,
        environment: Environment,
        fov: float
    ):
        """
        Initializes the FOVManager with the environment and the field of view angle.

        Args:
            environment (Environment): The environment instance to manage the FOV within.
            fov (float): The field of view angle in radians.
        """
        self.environment = environment
        self.fov = fov

    def __call__(
        self,
        position: tuple[float, float],
        direction: float,
    ):
        """
        Calculates and returns the visible walls and objects within the FOV from the specified position and direction.

        Args:
            position (tuple[float, float]): The (x, y) coordinates from which the FOV is calculated.
            direction (float): The direction in which the FOV is oriented, in radians.

        Returns:
            tuple[list, list]: A tuple containing two lists:
                - The first list contains the visible parts of walls within the FOV.
                - The second list contains the visible parts of objects within the FOV.
        """
        phi1, phi2 = get_fov(direction, self.fov)

        objects_fov = [
            obj.visible_parts(*position)[
                points_within_angles(
                    obj.visible_parts(*position) - position,
                    phi1, phi2
                )
            ]
            for obj in self.environment.objects
        ]
        walls_fov = [
                wall.visible_parts(*position)[
                    points_within_angles(
                        wall.visible_parts(*position) - position,
                        phi1, phi2
                    )
                ]
                for wall in self.environment.walls
            ]

        return walls_fov, objects_fov