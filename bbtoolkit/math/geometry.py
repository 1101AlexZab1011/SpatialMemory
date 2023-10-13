from shapely.geometry import Point, Polygon
import numpy as np

def inpolygon(point_x: float | list[float], point_y: float | list[float], polygon_x: list[float], polygon_y: list[float]) -> list[bool]:
    """
    Check if points are inside or on the edge of a polygonal region.

    Args:
        point_x (float or list): X-coordinates of the points to check.
        point_y (float or list): Y-coordinates of the points to check.
        polygon_x (list): X-coordinates of the polygon vertices.
        polygon_y (list): Y-coordinates of the polygon vertices.

    Returns:
        list: A list of boolean values indicating whether each point is inside or on the edge of the polygon.
    """
    # Create a Polygon object from the polygon vertices
    polygon = Polygon(list(zip(polygon_x, polygon_y)))

    # If single point coordinates are provided, convert them into a list
    if isinstance(point_x, (int, float)):
        point_x = [point_x]
        point_y = [point_y]

    # Create Point objects for the input points
    points = [Point(x, y) for x, y in zip(point_x, point_y)]

    # Check if each point is inside or on the edge of the polygon
    # is_inside = [point.within(polygon) for point in points]

    is_inside = [polygon.covers(point) for point in points]

    return is_inside


def compute_intersection(point1: np.ndarray, point2: np.ndarray, direction1: np.ndarray, direction2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the intersection points between two lines defined by their starting points and directions.

    Args:
        point1 (numpy.ndarray): Starting point of the first line.
        point2 (numpy.ndarray): Starting point of the second line.
        direction1 (numpy.ndarray): Direction vector of the first line.
        direction2 (numpy.ndarray): Direction vector of the second line.

    Returns:
        Tuple of two numpy arrays, representing the intersection points:
          - alpha1 (numpy.ndarray): Parameter values indicating the intersection points along the first line.
          - alpha2 (numpy.ndarray): Parameter values indicating the intersection points along the second line.

    The function calculates the intersection points of two lines in three-dimensional space. It utilizes the cross product
    of the direction vectors of the lines to determine where they intersect. If the lines are parallel (cross product
    result is zero), the corresponding alpha value is set to NaN to indicate no intersection.

    Note: The function assumes that the input numpy arrays have the same shape and represent valid points and directions
    in 3D space.
    """
    # Calculate how far along each line two lines intersect.

    denominator_2 = np.cross(direction1, direction2).astype(float)  # Cross product of direction1 and direction2
    denominator_2[denominator_2 == 0] = np.nan

    denominator_1 = -denominator_2

    alpha2 = np.cross(point2 - point1, direction1) / denominator_2
    alpha1 = np.cross(point1 - point2, direction2) / denominator_1

    return alpha1[:, 2], alpha2[:, 2]


def calculate_polar_distance(max_radius: int) -> np.ndarray:
    """
    Calculate polar distances for a set of concentric circles with incremental radii.

    Args:
        max_radius (int): The maximum radius for the concentric circles.

    Returns:
        np.ndarray: An array of polar distances corresponding to each concentric circle.

    Example:
        # Calculate polar distances for concentric circles with a maximum radius of 5.
        polar_distances = calculate_polar_distance(5)

        # The result will be an array of polar distances for the circles.
    """
    # Define incremental radii for concentric circles
    radial_increment = 0.1 * np.arange(1, max_radius + 1)
    radial_increment[0] = 1

    # Calculate cumulative polar distances
    polar_dist = np.cumsum(radial_increment)
    max_polar_dist = polar_dist[-1]

    # Normalize polar distances to fit within the specified maximum radius
    polar_dist = (polar_dist / max_polar_dist) * (max_radius - 0.5)

    return polar_dist
