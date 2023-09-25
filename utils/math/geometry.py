from shapely.geometry import Point, Polygon

def inpolygon(point_x, point_y, polygon_x, polygon_y):
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
