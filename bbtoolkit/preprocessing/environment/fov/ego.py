import numpy as np

from bbtoolkit.preprocessing.environment.fov import FOVManager
from bbtoolkit.preprocessing.environment.fov.utils import rotate_coordinates


class EgoManager:
    """
    A class to manage the ego-centric transformation of coordinates within a field of view.

    Attributes:
        fov (FOVManager): An instance of FOVManager which provides the field of view data.

    Methods:
        rotate(coords, phi): Rotates a list of coordinates by a given angle.
        relative(coords, position): Translates a list of coordinates to a relative position.
        __call__(position, direction): Applies ego-centric transformations to the field of view data.
    """
    def __init__(self, fov: FOVManager):
        """
        Initializes the EgoManager with a FOVManager instance.

        Args:
            fov (FOVManager): An instance of FOVManager to provide field of view data.
        """
        self.fov = fov

    @staticmethod
    def rotate(coords: list[np.ndarray], phi: float) -> list[np.ndarray]:
        """
        Rotates a list of coordinate arrays by a given angle phi.

        Args:
            coords (list[np.ndarray]): A list of numpy arrays representing coordinates.
            phi (float): The rotation angle in radians.

        Returns:
            list[np.ndarray]: A list of rotated numpy arrays.
        """
        return [rotate_coordinates(coord, phi) for coord in coords]

    @staticmethod
    def relative(coords: list[np.ndarray], position: tuple[float, float]) -> list[np.ndarray]:
        """
        Translates a list of coordinate arrays to be relative to a given position.

        Args:
            coords (list[np.ndarray]): A list of numpy arrays representing coordinates.
            position (tuple[float, float]): A tuple representing the reference position.

        Returns:
            list[np.ndarray]: A list of translated numpy arrays.
        """
        return [coord - position for coord in coords]

    def __call__(self, position: tuple[float, float], direction: float):
        """
        Applies ego-centric transformations to the field of view data based on the agent's position and direction.

        Args:
            position (tuple[float, float]): The agent's position as a tuple of (x, y) coordinates.
            direction (float): The agent's direction in radians.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists of numpy arrays.
                The first list contains the ego-centric coordinates of walls,
                and the second list contains the ego-centric coordinates of objects.
        """
        direction %= 2*np.pi
        walls_fov, objects_fov = self.fov(position, direction)
        walls_ego = self.rotate(self.relative(walls_fov, position), direction - np.pi/2)
        objects_ego = self.rotate(self.relative(objects_fov, position), direction - np.pi/2)

        return walls_ego, objects_ego