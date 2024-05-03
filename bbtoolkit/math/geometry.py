import scipy as sp
from shapely.geometry import Point, Polygon
import numpy as np
from itertools import product
from scipy.spatial import distance
from bbtoolkit.math.tensor_algebra import cross3d, sub3d
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from shapely.affinity import scale
from queue import PriorityQueue


def get_closest_points_indices(
    coords: np.ndarray,
    index: int,
    tree: KDTree = None,
    n_points: int = 10
) -> list[int]:
    """
    Gets the indices of the closest points to a given point in a set of coordinates.

    This function uses a KDTree to efficiently find the closest points.

    Args:
        coords (np.ndarray): The coordinates to search within (array of shape n_points x 2).
        index (int): The index of the point to find the closest points to.
        tree (KDTree, optional): The KDTree to use for the search. If None, a new KDTree is created. Defaults to None.
        n_points (int, optional): The number of closest points to find. Defaults to 10.

    Returns:
        list[int]: The indices of the closest points.
    """
    if tree is None:
        tree = KDTree(coords)
    _, indices = tree.query(np.atleast_2d(coords[index]), k=n_points)  # Get n closest points (including the point itself)

    return indices.reshape(-1)


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
    exterior_starting_points = np.array(poly.exterior.coords[1:])
    interior_starting_points = [np.array(interior.coords[1:]) for interior in poly.interiors]
    starting_points = np.concatenate([exterior_starting_points, *interior_starting_points])
    directions = np.concatenate([
        poly.exterior.coords[:-1],
        *[interior.coords[:-1] for interior in poly.interiors]
    ]) - starting_points

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


def interpolate_2d_points(points: list[tuple[float, float]] | np.ndarray, n_points: int, method='linear') -> np.ndarray:
    """
    Interpolates a given set of 2D points to generate a specified number of points along the curve defined by the original points.

    This function supports various interpolation methods, such as 'linear', 'nearest', 'zero', 'slinear', 'quadratic', and 'cubic'.

    Args:
        points (list[tuple[float, float]] | np.ndarray): The original set of 2D points to interpolate. Can be a list of tuples or a numpy array.
        n_points (int): The number of interpolated points to generate.
        method (str, optional): The method of interpolation to use. Defaults to 'linear'.

    Returns:
        np.ndarray: A numpy array of shape (n_points, 2), containing the interpolated 2D points.

    Example:
        >>> original_points = [(0, 0), (1, 1), (2, 0)]
        >>> interpolated_points = interpolate_2d_points(original_points, 5)
        >>> print(interpolated_points)
        [[0.   0.  ]
         [0.5  0.5 ]
         [1.   1.  ]
         [1.5  0.5 ]
         [2.   0.  ]]
    """
    # Ensure points is a numpy array
    points = np.array(points)

    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Create a parameter t along the curve (assuming points are ordered)
    t = np.linspace(0, 1, len(points))

    # Create an interpolation function for x and y separately
    fx = interp1d(t, x, kind=method)
    fy = interp1d(t, y, kind=method)

    # Create a new parameter space for the interpolated points
    t_new = np.linspace(0, 1, n_points)

    # Interpolate x and y
    x_new = fx(t_new)
    y_new = fy(t_new)

    # Combine x and y to get the interpolated points
    interpolated_points = np.vstack((x_new, y_new)).T

    return interpolated_points


def get_farthest_point_index(points: np.ndarray) -> int:
    """
    Finds the index of the point farthest from the centroid of a set of points.

    Args:
        points (np.ndarray): An array of points of shape (n_points, dimensions).

    Returns:
        int: The index of the point farthest from the centroid.

    Example:
        >>> points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        >>> index = get_farthest_point_index(points)
        >>> print(index)
        3
    """
    centroid = np.mean(points, axis=0)
    distances_from_centroid = np.linalg.norm(points - centroid, axis=1)
    return np.argmax(distances_from_centroid)


def sort_points_by_proximity(points: np.ndarray) -> np.ndarray:
    """
    Sorts a set of points starting from the point farthest from the centroid, 
    then by proximity to each subsequent point.

    Args:
        points (np.ndarray): An array of points of shape (n_points, dimensions).

    Returns:
        np.ndarray: An array of points sorted by proximity, starting with the point farthest from the centroid.

    Example:
        >>> points = np.array([[0, 0], [2, 2], [1, 1], [3, 3]])
        >>> sorted_points = sort_points_by_proximity(points)
        >>> print(sorted_points)
        [[3 3]
         [2 2]
         [1 1]
         [0 0]]
    """
    # Find the most distant point from the centroid
    farthest_index = get_farthest_point_index(points)
    starting_point = points[farthest_index]

    # Initialize the sorted points array
    sorted_points = np.zeros_like(points)
    sorted_points[0] = starting_point
    remaining_points = np.delete(points, farthest_index, axis=0).tolist()

    # Sort the remaining points by proximity
    for i in range(1, len(points)):
        current_point = sorted_points[i-1]
        distances = np.linalg.norm(np.array(remaining_points) - current_point, axis=1)
        closest_index = np.argmin(distances)
        sorted_points[i] = remaining_points.pop(closest_index)

    return sorted_points


def mask_to_slices(mask: np.ndarray) -> list[slice]:
    """
    Converts a boolean mask to a list of slice objects representing the True segments of the mask.

    Args:
        mask (np.ndarray): A 1D boolean array.

    Returns:
        list[slice]: A list of slice objects corresponding to the True segments of the mask.

    Example:
        >>> mask = np.array([True, True, False, True, True, True, False])
        >>> slices = mask_to_slices(mask)
        >>> print(slices)
        [slice(0, 2, None), slice(3, 6, None)]
    """
    ranges = np.concatenate([[0], np.where(~mask)[0] + 1, [len(mask)]])
    return [slice(start, end) for start, end in zip(ranges[:-1], ranges[1:])]


def split_points(points: np.ndarray) -> list[np.ndarray]:
    """
    Splits a set of points into segments based on the mode of the distances between consecutive points.

    Args:
        points (np.ndarray): An array of points of shape (n_points, dimensions).

    Returns:
        list[np.ndarray]: A list of arrays, each representing a segment of points.

    Example:
        >>> points = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11]])
        >>> segments = split_points(points)
        >>> for segment in segments:
        ...     print(segment)
        [[0 0]
         [1 1]
         [2 2]]
        [[10 10]
         [11 11]]
    """
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    res = sp.stats.mode(distances)
    return [points[slice_, :] for slice_ in mask_to_slices(np.isclose(distances, res.mode))]


def points2segments(coords: np.ndarray) -> np.ndarray:
    """
    Converts a set of coordinates into line segments by first sorting the points by proximity, 
    then splitting them into connected segments, and finally pairing adjacent points into segments.

    Args:
        coords (np.ndarray): An array of coordinates of shape (n_points, dimensions).

    Returns:
        np.ndarray: An array of line segments of shape (n_segments, 4), where each segment is represented by 
        the starting and ending coordinates (x1, y1, x2, y2).

    Example:
        >>> coords = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11]])
        >>> segments = points2segments(coords)
        >>> print(segments)
        [[ 0.  0.  1.  1.]
         [ 1.  1.  2.  2.]
         [10. 10. 11. 11.]]
    """
    # Ensure coords is a sequence
    coords = sort_points_by_proximity(coords)
    # Split interrupted segments
    connected_segments = split_points(coords)

    all_segments = []
    for points in connected_segments:

        # Split the coordinates into x and y components
        x_coords, y_coords = points[:, 0], points[:, 1]

        # Create segments by "zipping" adjacent points
        all_segments.append(np.column_stack((x_coords[:-1], y_coords[:-1], x_coords[1:], y_coords[1:])))

    return np.concatenate(all_segments)


def resize_polygon(polygon: Polygon, increase_factor: float) -> Polygon:
    """
    Resizes a polygon by scaling it up or down around its centroid based on a given increase factor.

    This function calculates the centroid of the given polygon and scales the polygon around this point.
    The scaling is uniform in both the x and y directions.

    Args:
        polygon (Polygon): The polygon to be resized. Must be an instance of a Polygon class.
        increase_factor (float): The factor by which the polygon is to be scaled. Values greater than 1 will
                                 enlarge the polygon, while values between 0 and 1 will shrink it.

    Returns:
        Polygon: A new Polygon instance representing the resized polygon.

    Example:
        >>> from shapely.geometry import Polygon
        >>> original_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> resized_polygon = resize_polygon(original_polygon, 2)
        >>> print(resized_polygon)
        POLYGON ((-0.5 -0.5, 1.5 -0.5, 1.5 1.5, -0.5 1.5, -0.5 -0.5))
    """
    # Calculate the centroid of the polygon
    centroid = polygon.centroid
    # Scale the polygon around its centroid
    scaled_polygon = scale(polygon, xfact=increase_factor, yfact=increase_factor, origin=centroid)
    return scaled_polygon


def is_point_outside_polygons(point: Point, polygons: list[Polygon]):
        """
        Checks if a given point is outside all polygons in a list.

        Args:
            point (Point): The point to check.
            polygons (list[Polygon]): A list of Polygon objects.

        Returns:
            bool: True if the point is outside all polygons, False otherwise.
        """
        for polygon in polygons:
            if polygon.contains(point):
                return False
        return True


def closest_grid_point(
    x: float,
    y: float,
    dx: float,
    dy: float,
    polygons: list[Polygon] = None,
    n_steps_max: int = 1000
) -> tuple[float, float]:
    """
    Finds the closest grid point to a given location (x, y) that is not contained within any of the specified polygons.
    The grid is defined by the spacing dx and dy in the x and y directions, respectively.

    This function performs a spiral search outward from the initial closest grid point to find a point that is not
    inside any of the given polygons. If the initial closest grid point is not inside any polygon, it is returned immediately.

    Args:
        x (float): The x-coordinate of the location.
        y (float): The y-coordinate of the location.
        dx (float): The grid spacing in the x direction.
        dy (float): The grid spacing in the y direction.
        polygons (list[Polygon], optional): A list of Polygon objects to avoid. Defaults to None, which is treated as an empty list.
        n_steps_max (int, optional): The maximum number of spiral steps to take. Defaults to 1000.

    Returns:
        tuple[float, float]: The coordinates of the closest grid point not inside any of the polygons.

    Raises:
        ValueError: If the function exceeds the maximum number of steps without finding a suitable point.

    Example:
        >>> from shapely.geometry import Polygon, Point
        >>> polygon = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
        >>> closest_point = closest_grid_point(1.5, 1.5, 0.1, 0.1, [polygon])
        >>> print(closest_point)
        (2.0, 1.9)
    """

    polygons = list() if polygons is None else polygons
    # Start with the closest grid point
    closest_x = round(x / dx) * dx
    closest_y = round(y / dy) * dy
    point = Point(closest_x, closest_y)

    if is_point_outside_polygons(point, polygons):
        return point.x, point.y

    # Spiral search for the closest point outside polygons
    step = 1
    while True:
        if step > n_steps_max:
            raise ValueError("Maximum number of steps reached")

        for dx_step in range(-step, step + 1):
            for dy_step in range(-step, step + 1):
                test_x = closest_x + dx * dx_step
                test_y = closest_y + dy * dy_step
                test_point = Point(test_x, test_y)
                if is_point_outside_polygons(test_point, polygons):
                    return test_x, test_y
        step += 1


def a_star_search(
    start: Point,
    goal: Point,
    polygons: list[Polygon],
    dx: float = 1,
    dy: float = 1,
    d: float = 0,
    n_steps_max: int = 1000
) -> list[Point]:
    """
    Performs an A* search to find a path from a start point to a goal point, avoiding specified polygons.

    The search area is discretized into a grid defined by dx and dy. Optionally, polygons can be resized to add a buffer
    around obstacles by specifying a non-zero value for d. The search includes diagonal movements and uses a heuristic
    based on the Euclidean distance.

    Args:
        start (Point): The starting point of the path.
        goal (Point): The goal point of the path.
        polygons (list[Polygon]): A list of Polygon objects representing obstacles to avoid.
        dx (float, optional): The grid spacing in the x direction. Defaults to 1.
        dy (float, optional): The grid spacing in the y direction. Defaults to 1.
        d (float, optional): The distance by which to resize polygons (increase or decrease). Defaults to 0.
        n_steps_max (int, optional): The maximum number of steps to take before giving up. Defaults to 1000.

    Returns:
        list[Point]: A list of Point objects representing the path from start to goal, or None if no path is found.

    Raises:
        ValueError: If the search exceeds the maximum number of steps without finding a path.

    Example:
        >>> from shapely.geometry import Point, Polygon
        >>> start = Point(0, 0)
        >>> goal = Point(10, 10)
        >>> polygons = [Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])]
        >>> path = a_star_search(start, goal, polygons, dx=1, dy=1, d=0.5)
        >>> print(path)
        [Point(0, 0), Point(1, 1), ..., Point(10, 10)]
    """
    def heuristic(a: tuple[float, float], b: tuple[float, float]) -> float:
        """
        Calculates the Euclidean distance between two points.

        Args:
            a (tuple[float, float]): The first point.
            b (tuple[float, float]): The second point.

        Returns:
            float: The Euclidean distance between the points.
        """
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    if dx % .5 != 0 or dy % .5 != 0:
        raise ValueError(f'In the current implementation grid spacing must be a multiple of 0.5, got dx={dx} and dy={dy}')

    if d != 0:
        polygons = [resize_polygon(polygon, d) for polygon in polygons]

    actual_start = (start.x, start.y)
    actual_goal = (goal.x, goal.y)
    start = closest_grid_point(*actual_start, dx, dy, polygons)
    goal = closest_grid_point(*actual_goal, dx, dy, polygons)

    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    step = 0
    while not open_set.empty():
        if step > n_steps_max:
            raise ValueError("Maximum number of steps reached")

        current = open_set.get()[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(Point(current[0], current[1]))
                current = came_from[current]
            path.append(Point(start[0], start[1]))
            out = path[::-1]
            out[0] = Point(actual_start)
            out[-1] = Point(actual_goal)
            return out

        # Include diagonal directions
        directions = [(dx, 0), (-dx, 0), (0, dy), (0, -dy), (dx, dy), (-dx, -dy), (dx, -dy), (-dx, dy)]
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            neighbor_point = Point(neighbor[0], neighbor[1])

            if any(polygon.contains(neighbor_point) for polygon in polygons):
                continue

            # Distance to neighbor is sqrt(2) for diagonals, else 1
            tentative_g_score = g_score[current] + np.sqrt(direction[0]**2 + direction[1]**2)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if not any(neighbor == item[1] for item in open_set.queue):
                    open_set.put((f_score[neighbor], neighbor))
        step += 1

    return None


def remove_collinear_points(points: list[Point]) -> list[Point]:
    """
    Removes collinear points from a list of points to simplify a polyline or polygon.

    This function iterates through a list of points and removes any point that forms a straight line with its
    immediate neighbors. At least three points are required to check for collinearity; if fewer are provided,
    the original list is returned unchanged.

    Args:
        points (list[Point]): A list of Point objects representing the vertices of a polyline or polygon.

    Returns:
        list[Point]: A list of Point objects with collinear points removed.

    Example:
        >>> from shapely.geometry import Point
        >>> points = [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4), Point(5, 5)]
        >>> simplified_points = remove_collinear_points(points)
        >>> print([(p.x, p.y) for p in simplified_points])
        [(0, 0), (5, 5)]
    """
    if len(points) < 3:
        return points  # Not enough points to form a line

    # Function to calculate the cross product of vectors AB and AC
    def cross_product(A: Point, B: Point, C: Point) -> float:
        """
        Calculates the cross product of vectors AB and AC.

        The cross product is used to determine the orientation of three points and to check if they are collinear.
        A result of 0 indicates that the points are collinear.

        Args:
            A (Point): The starting point of vectors AB and AC.
            B (Point): The ending point of vector AB.
            C (Point): The ending point of vector AC.

        Returns:
            float: The cross product of vectors AB and AC.
        """
        return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)

    # Initialize the result list with the first two points
    result = [points[0], points[1]]

    for i in range(2, len(points)):
        while len(result) >= 2 and cross_product(result[-2], result[-1], points[i]) == 0:
            # If the last three points are collinear, remove the middle point
            result.pop()
        result.append(points[i])

    return result
