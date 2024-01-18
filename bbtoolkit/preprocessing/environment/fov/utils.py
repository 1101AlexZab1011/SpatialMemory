import numpy as np


def get_fov(phi: float, theta:float) -> tuple[float, float]:
    """
    Calculates the lower and upper bounds of the field of view.

    Args:
        phi (float): The angle of the agent.
        theta (float): The field of view angle.

    Returns:
        tuple[float, float]: The lower and upper bounds of the field of view.
    """
    # Calculate the lower and upper bounds of the field of view
    lower_bound = (phi - theta/2) % (2*np.pi)
    upper_bound = (phi + theta/2) % (2*np.pi)

    return lower_bound, upper_bound


def points_within_angles(coordinates: np.ndarray, angle_start: float, angle_end: float) -> np.ndarray:
    """Determine which points from a set of coordinates are within a specified angular range.

    This function normalizes the start and end angles to be within the range [0, 2*pi). It calculates
    the angle of each point with respect to the positive x-axis and determines if each angle is within
    the specified range. The range is inclusive of the start angle and exclusive of the end angle if
    angle_start is less than angle_end. If angle_start is greater than angle_end, the function checks
    if the points are within the angular range that wraps around 2*pi.

    Args:
        coordinates (np.ndarray): An array of shape (N, 2) containing N points, where each point is
            represented by its (x, y) coordinates.
        angle_start (float): The starting angle of the angular range, in radians. Will be normalized
            to [0, 2*pi).
        angle_end (float): The ending angle of the angular range, in radians. Will be normalized to
            [0, 2*pi).

    Returns:
        np.ndarray: A boolean array of shape (N,) where each element is True if the corresponding
            point's angle is within the specified angular range, and False otherwise.

    Note:
        This function assumes that the input coordinates are in Cartesian format and that the angles
        are in radians.
    """
    # Normalize start and end angles to be within the range [0, 2*pi)
    angle_start = angle_start % (2 * np.pi)
    angle_end = angle_end % (2 * np.pi)

    angle = np.arctan2(coordinates[:, 1], coordinates[:, 0]) % (2 * np.pi)

    if angle_start <= angle_end:
        return np.logical_and(angle_start <= angle, angle <= angle_end)
    else:
        return np.logical_or(angle >= angle_start, angle <= angle_end)
