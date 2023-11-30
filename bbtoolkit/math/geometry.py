from shapely.geometry import Point, Polygon
import numpy as np
from itertools import product
from scipy.spatial import distance

from bbtoolkit.math.tensor_algebra import cross3d, sub3d


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

    The function calculates the intersection points of two lines in two-dimensional space. It utilizes the cross product
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


def create_cartesian_space(from_: int | tuple[int, ...], to: int | tuple[int, ...], res: float, return_grid: bool = False) -> list[np.ndarray]:
    """
    Function to create an n-dimensional coordinates system in specified ranges.

    Parameters:
    from_ (tuple): Tuple of n dimensions representing boundaries from which grid should start.
    to (tuple): Tuple of n dimensions representing end range of grid in each dimension.
    res (tuple): Tuple representing resolution at each dimension.
    return_grid (bool): If True, returns a list of n-dimensional meshgrids. Othervise returns a list of 1-dimensional arrays representing ordinate vectors. Default is False.

    Returns:
    list: List of n-dimensional meshgrids.
    """
    if isinstance(from_, (int, float)):
        from_ = (from_,)
    if isinstance(to, (int, float)):
        to = (to,)
    if isinstance(res, float):
        res = (res,)*len(from_)
    # Check if the lengths of the input tuples are equal
    if len(from_) != len(to) or len(from_) != len(res):
        raise ValueError("All input parameters must be of the same length.")

    # Create ranges for each dimension
    # ranges = [np.arange(from_[i], to[i], res[i]) for i in range(len(from_))]
    ranges = [np.arange(from_[i], to[i] + res[i], res[i]) for i in range(len(from_))]


    if return_grid:
        # Create meshgrid
        mesh = np.meshgrid(*ranges, indexing='ij')

        return mesh

    return ranges


def regroup_min_max(*args: tuple[int | float, ...]) -> tuple[tuple[int | float, ...], tuple[int | float, ...]]:
    """
    Function to regroup min and max values for each dimension.

    Args:
        *args: Variable number of tuples of min and max values for each dimension.

    Returns:
        tuple: Tuple of tuples of min values and tuple of tuples of max values for each dimension.
    """
    half = len(args) // 2
    min_values = args[:half]
    max_values = args[half:]
    return min_values, max_values


def create_shapely_points(*vectors: np.ndarray, res: float) -> list[Point]:
    """
    Function to create Shapely points filling a space defined by the given vectors.

    Args:
        *vectors: Variable number of vectors defining the space.
        res (float): Resolution of the space.
    Returns:
        list: List of Shapely points.
    """
    # Generate combinations of coordinates
    coordinates = product(*vectors)

    # Create Shapely points from coordinates
    points = [Point(coord).buffer(res/2) for coord in coordinates] # half resolution to make points to not overlap

    return points


def points2indices(points: np.ndarray, vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to create a list of indices of points in the vectors.

    Args:
        points (list): List of Shapely points.
        vectors (list): List of vectors defining the space.

    Returns:
        tuple: Tuple of arrays of indices of points in the vectors.
    """
    # Create a list of coordinates of points
    # Create a list of indices of points in the vectors
    x_indices = [np.argmin(np.abs(vectors[0] - center[0])) for center in points]
    y_indices = [np.argmin(np.abs(vectors[1] - center[1])) for center in points]

    return np.array(x_indices), np.array(y_indices)


def points2mask(points: np.ndarray, vectors: list[np.ndarray]) -> np.ndarray:
    """
    Function to create a mask from a list of Shapely points.

    Args:
        points (list): List of Shapely points.
        vectors (list): List of vectors defining the space.

    Returns:
        np.ndarray: Mask representing the points.
    """
    x_indices, y_indices = points2indices(points, vectors)

    # Create a mask
    mask = np.zeros((len(vectors[0]), len(vectors[1]))).astype(bool)
    mask[x_indices, y_indices] = True

    return mask


def poly2vectors(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms polygon to points and direction vectors.

    Args:
        poly (Polygon): Polygon to be transformed.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing points and direction vectors.
    """
    starting_points = np.array(poly.exterior.coords[1:])
    directions = np.array(poly.exterior.coords[:-1]) - starting_points

    return starting_points, directions


def matrix2vectors(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms matrix of polygon coordinates (n_polygons, n_points, 2) to points and direction vectors.

    Args:
        matrix (np.ndarray): Matrix to be transformed.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing points and direction vectors.
    """
    starting_points = matrix[:, 1:, :]
    directions = matrix[:, :-1, :] - starting_points

    return starting_points.reshape(-1, 2), directions.reshape(-1, 2)


def find_closest_points(space1: np.ndarray, space2: np.ndarray) -> np.ndarray:
    """
    Finds the points in space1 that are closest to each point in space2.

    This function computes the Euclidean distance between each point in space1 and each point in space2,
    then finds the index of the minimum distance for each point in space2. It then returns the points in space1
    that correspond to these indices.

    Args:
        space1 (np.ndarray): A 2D numpy array of shape (m, 2) representing m points in a 2D space.
        space2 (np.ndarray): A 2D numpy array of shape (n, 2) representing n points in a 2D space.

    Returns:
        np.ndarray: A 2D numpy array of shape (n, 2) representing the n points in space1 that are closest to each point in space2.

    """
    # Compute the pairwise distances between M and N
    distances = distance.cdist(space1, space2, 'euclidean')

    # Get the indices of the minimum distances in M for each point in N
    min_indices = np.argmin(distances, axis=0)

    # Return the closest points in M for each point in N
    return space1[min_indices]


def compute_intersection3d(
    point1: np.ndarray,
    point2: np.ndarray,
    direction1: np.ndarray,
    direction2: np.ndarray,
    return_flat: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the intersection points between two lines defined by their starting points and directions.
    Vectorized version of `compute_intersection`. Can compute intersection between n vs m lines. Output is of shape (n, m, 3).

    Args:
        point1 (numpy.ndarray): Starting point of the first line.
        point2 (numpy.ndarray): Starting point of the second line.
        direction1 (numpy.ndarray): Direction vector of the first line.
        direction2 (numpy.ndarray): Direction vector of the second line.
        return_flat (bool): If True, returns flat tensor. Default is False.

    Returns:
        Tuple of two numpy arrays, representing the intersection points:
          - alpha1 (numpy.ndarray): Parameter values indicating the intersection points along the first line.
          - alpha2 (numpy.ndarray): Parameter values indicating the intersection points along the second line.

    """
    batch_first = True if point1.ndim == 3 else False

    denominator_2 = cross3d(direction1, direction2, return_2d=False, batch_first=batch_first)
    denominator_2[denominator_2 == 0] = np.nan
    denominator_1 = -denominator_2

    if batch_first:
        transpose_dims = (0, 2, 1, 3)
    else:
        transpose_dims = (1, 0, 2)

    alpha2 = np.cross(
        sub3d(
            point2,
            point1,
            return_2d=False,
            batch_first=batch_first
        ),
        direction1[:, :, None, :] if batch_first else direction1[:, None, :],
        ).transpose(*transpose_dims)/denominator_2
    alpha1 = np.cross(
        sub3d(
            point1,
            point2,
            return_2d=False,
            batch_first=batch_first
        ),
        direction2[:, :, None, :] if batch_first else direction2[:, None, :],
    )/denominator_1

    alpha1 = alpha1[:, :, :, 2] if batch_first else alpha1[:, :, 2]
    alpha2 = alpha2[:, :, :, 2] if batch_first else alpha2[:, :, 2]

    if return_flat:
        return alpha1.reshape(-1), alpha2.reshape(-1)

    return alpha1, alpha2
