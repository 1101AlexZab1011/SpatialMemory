import math

import numpy as np
from bbtoolkit.utils.math.geometry import a_star_search, interpolate_2d_points, remove_collinear_points
from shapely import Point
from bbtoolkit.utils.movement import MovementManager
from bbtoolkit.environment import Environment


class TrajectoryManager:
    """
    Manages the generation of trajectories between two points using interpolation methods.

    Attributes:
        n_points (int): The number of points to generate for the trajectory.
        method (str): The interpolation method to use. Supported methods include 'linear', 'quadratic', 'cubic', etc.
        dx (float, optional): The distance used to determine the control points for the interpolation. If not provided,
                              it is calculated based on the distance between the start and end positions.

    Args:
        n_points (int): The number of points to generate for the trajectory.
        method (str, optional): The interpolation method to use. Defaults to 'quadratic'.
        dx (float, optional): The distance used to determine the control points for the interpolation.

    Methods:
        __call__(position1: tuple[float, float], position2: tuple[float, float], angle: float) -> tuple[float, float]:
            Generates a trajectory between two points given an initial angle.

        create_point_on_angle(x: float, y: float, angle: float, distance: float) -> tuple[float, float]:
            Calculates a new point given an initial point, angle, and distance.

    Example:
        >>> tm = TrajectoryManager(n_points=100, method='quadratic')
        >>> trajectory = tm((0, 0), (10, 10), math.pi/4)
        >>> print(trajectory.shape)
        (134, 2)
    """
    def __init__(self, n_points: int, method: str = 'quadratic', dx: float = None):
        """
        Initializes the TrajectoryManager with the number of points, interpolation method, and optional distance for control points.
        """
        self.dx = dx
        self.n_points = n_points
        self.method = method

    def __call__(
        self,
        position1: tuple[float, float],
        position2: tuple[float, float],
        angle: float
    ) -> tuple[float, float]:
        """
        Generates a trajectory between two points given an initial angle.

        Args:
            position1 (tuple[float, float]): The starting position of the trajectory.
            position2 (tuple[float, float]): The ending position of the trajectory.
            angle (float): The initial angle in radians.

        Returns:
            np.ndarray: An array of points representing the generated trajectory.
        """
        angle %= 2*math.pi

        dx = self.dx if self.dx is not None else MovementManager.compute_distance(position1, position2)/4

        angle2 = MovementManager.get_angle_with_x_axis(
            [
                position2[0] - position1[0],
                position2[1] - position1[1]
            ]
        )

        point_1 = self.create_point_on_angle(*position1, angle + .25*(angle2 - angle), dx)
        point_2 = self.create_point_on_angle(*position1, angle + .5*(angle2 - angle), 2*dx)

        coords = np.array([position1, point_1, point_2, position2])
        coords = interpolate_2d_points(coords, int(self.n_points*4/3), method=self.method)

        return coords

    @staticmethod
    def create_point_on_angle(x: float, y: float, angle: float, distance: float) -> tuple[float, float]:
        """
        Calculates a new point given an initial point, angle, and distance.

        Args:
            x (float): The x-coordinate of the initial point.
            y (float): The y-coordinate of the initial point.
            angle (float): The angle in radians.
            distance (float): The distance from the initial point to the new point.

        Returns:
            tuple[float, float]: The coordinates of the new point.
        """
        new_x = x + distance * math.cos(angle)
        new_y = y + distance * math.sin(angle)
        return new_x, new_y


class AStarTrajectory(TrajectoryManager):
    """
    Extends TrajectoryManager to generate trajectories using A* search to navigate around obstacles in an environment.

    This class uses A* search to find a path between two points that avoids obstacles defined in the given environment.
    It then interpolates additional points along this path to create a smooth trajectory. The class allows for adjusting
    the granularity of the search grid and the amount by which obstacles are "inflated" to ensure clearance.

    Attributes:
        environment (Environment): The environment containing objects and walls.
        poly_increase_factor (float): The factor by which to increase the size of polygons (obstacles and walls) for
                                      collision avoidance. A larger value increases the clearance from obstacles.

    Args:
        environment (Environment): The environment in which the trajectory is to be generated.
        n_points (int): The number of points to generate for the trajectory.
        method (str, optional): The interpolation method to use. Defaults to 'quadratic'.
        dx (float, optional): The grid spacing for the A* search. Defaults to 1.
        poly_increase_factor (float, optional): The factor by which to increase the size of polygons for collision avoidance.
                                                Defaults to 0.

    Methods:
        __call__(position1: tuple[float, float], position2: tuple[float, float], angle: float) -> np.ndarray:
            Generates a trajectory between two points, avoiding obstacles in the environment.
    """
    def __init__(
        self,
        environment: Environment,
        n_points: int = 10,
        method: str = 'linear',
        dx: float = .5,
        poly_increase_factor: float = 1.5
    ):
        """
        Initializes the AStarTrajectory with the environment, number of points, interpolation method, grid spacing,
        and polygon increase factor for collision avoidance.
        """
        super().__init__(n_points, method, dx)
        self.environment = environment
        self.poly_increase_factor = poly_increase_factor

    def __call__(
        self,
        position1: tuple[float, float],
        position2: tuple[float, float],
        angle: float
    ) -> tuple[float, float]:
        """
        Generates a trajectory between two points, avoiding obstacles in the environment.

        The method first calculates an average angle between the initial angle and the angle between the start and end points.
        It then performs an A* search to find a path that avoids obstacles, and interpolates additional points along this path
        to create a smooth trajectory.

        Args:
            position1 (tuple[float, float]): The starting position of the trajectory.
            position2 (tuple[float, float]): The ending position of the trajectory.
            angle (float): The initial angle in radians.

        Returns:
            np.ndarray: An array of points representing the generated trajectory.
        """
        additional_points = remove_collinear_points(a_star_search(
            Point(position1),
            Point(position2),
            [obj.polygon.obj for obj in self.environment.objects] +
            [obj.polygon.obj for obj in self.environment.walls],
            self.dx, self.dx, self.poly_increase_factor
        ))

        for point in additional_points:
            for obj in self.environment.objects + self.environment.walls:
                if obj.polygon.contains(point):
                    raise

        if additional_points is None:
            additional_points = [position1, position2]
        else:
            additional_points = [(point.x, point.y) for point in additional_points]

        all_points = [position1, *additional_points[1:-1], position2]

        coords = np.array(all_points)
        coords = interpolate_2d_points(coords, int(self.n_points*4/3) + len(additional_points) - 2, method=self.method)

        return coords
