import math


class MovementManager:
    """
    Manages the movement of an entity, including its speed, rotation, and position.

    Attributes:
        speed (float): The speed of the entity in units per second.
        rotation_speed (float): The rotation speed of the entity in radians per second.
        position (tuple[float, float]): The current position of the entity as a tuple (x, y).
        direction (float): The current direction of the entity in radians, normalized to [0, 2π).

    Args:
        speed (float): The speed of the entity.
        rotation_speed (float): The rotation speed of the entity.
        position (tuple[float, float]): The initial position of the entity.
        direction (float): The initial direction of the entity in radians.

    Methods:
        time_per_distance(distance: float) -> int:
            Calculates the time required to cover a certain distance at the entity's speed.

        distance_per_time(time: float) -> float:
            Calculates the distance covered in a certain amount of time at the entity's speed.

        time_per_angle(angle: float) -> int:
            Calculates the time required to rotate through a certain angle at the entity's rotation speed.

        angle_per_time(time: float) -> float:
            Calculates the angle rotated through in a certain amount of time at the entity's rotation speed.

        compute_distance(position1: tuple[int, int], position2: tuple[int, int]) -> float:
            Calculates the Euclidean distance between two points.

        get_angle_with_x_axis(point: tuple[float, float]) -> float:
            Calculates the angle between the positive x-axis and a point, normalized to [0, 2π).

        smallest_angle_between(theta1: float, theta2: float) -> float:
            Calculates the smallest angle between two angles, considering the circular nature of angles.

        __call__(position: tuple[int, int]) -> tuple[float, float, float]:
            Calculates the distance, angle, and time required to move from the current position to a new position.
    """
    def __init__(self, speed: float, rotation_speed: float, position: tuple[float, float], direction: float):
        """
        Initializes the MovementManager with speed, rotation speed, initial position, and direction.
        """
        self.speed = speed
        self.rotation_speed = rotation_speed
        self.position = position
        self.direction = direction % (2 * math.pi)

    def time_per_distance(self, distance: float) -> int:
        """
        Calculates the time required to cover a certain distance at the entity's speed.

        Args:
            distance (float): The distance to be covered.

        Returns:
            int: The time required to cover the distance.
        """
        return distance / self.speed

    def distance_per_time(self, time: float) -> float:
        """
        Calculates the distance covered in a certain amount of time at the entity's speed.

        Args:
            time (float): The time during which the entity moves.

        Returns:
            float: The distance covered in the given time.
        """
        return time * self.speed

    def time_per_angle(self, angle: float) -> int:
        """
        Calculates the time required to rotate through a certain angle at the entity's rotation speed.

        Args:
            angle (float): The angle to be rotated through.

        Returns:
            int: The time required to rotate through the angle.
        """
        return angle / self.rotation_speed

    def angle_per_time(self, time: float) -> float:
        """
        Calculates the angle rotated through in a certain amount of time at the entity's rotation speed.

        Args:
            time (float): The time during which the entity rotates.

        Returns:
            float: The angle rotated through in the given time.
        """
        return time * self.rotation_speed

    @staticmethod
    def compute_distance(position1: tuple[int, int], position2: tuple[int, int]) -> float:
        """
        Calculates the Euclidean distance between two points.

        Args:
            position1 (tuple[int, int]): The first point.
            position2 (tuple[int, int]): The second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return math.sqrt(
            (position1[0] - position2[0])**2 +
            (position1[1] - position2[1])**2
        )

    @staticmethod
    def get_angle_with_x_axis(point: tuple[float, float]) -> float:
        """
        Calculates the angle between the positive x-axis and a point, normalized to [0, 2π).

        Args:
            point (tuple[float, float]): The point for which to calculate the angle.

        Returns:
            float: The angle between the positive x-axis and the point.
        """
        x, y = point
        angle = math.atan2(y, x)
        if angle < 0:
            angle += 2 * math.pi
        return angle

    @staticmethod
    def smallest_angle_between(theta1: float, theta2: float) -> float:
        """
        Calculates the smallest angle between two angles, considering the circular nature of angles.

        Args:
            theta1 (float): The first angle.
            theta2 (float): The second angle.

        Returns:
            float: The smallest angle between the two angles.
        """
        theta1 = theta1 % (2 * math.pi)
        theta2 = theta2 % (2 * math.pi)
        angle_diff = abs(theta1 - theta2)

        return min(angle_diff, 2 * math.pi - angle_diff)

    def __call__(self, position: tuple[int, int]) -> tuple[float, float, float]:
        """
        Calculates the distance, angle, and time required to move from the current position to a new position.

        Args:
            position (tuple[int, int]): The new position to move to.

        Returns:
            tuple[float, float, float]: A tuple containing the distance, angle, and time required for the movement.
        """
        d = self.compute_distance(self.position, position)
        phi = self.smallest_angle_between(self.direction, self.get_angle_with_x_axis(position))
        t = max(self.time_per_distance(d), self.time_per_angle(phi))
        return d, phi, t