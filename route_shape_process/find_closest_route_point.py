import numpy as np

def closest_point_on_line(point, two_points_on_line):
    """
    Returns the projection of a point onto a line, i.e. the closest point on the line to the point.

    The formula for projecting the point P onto the line spanned by A and B is:

    C = A + (P-A).(B-A) * (B-A) / |B-A|^2,

    where . is the dot product, * is scalar multiplication,
    and |x|^2 is the squared Euclidean norm of the vector x.

    The calculation occurs in "affine coordinates" since the line is not assumed
    to go through the origin.

    Parameters
    ----------
    point : array_like (1D)
        The point to projection onto the line, represented in orthonormal coordinates.
        The shape should be (1,d), where d is the dimension of the space the point lives in.

    two_points_on_line : array_like (2D)
        The coordinates of two points on the line to project onto,
        in the same orthonormal basis as for `point`.
        The shape should be (2,d), where d is the same dimension as for `point`.
        Each of the two rows gives the coordinates of one point on the line.

    Returns
    -------
    closest_point : numpy.array
        The point on the line closest to the
    """
    segment_start = two_points_on_line[0]
    segment_end = two_points_on_line[1]
    direction = segment_end - segment_start
    projection = (point - segment_start).dot(direction) * direction / np.sum(direction**2)
    return segment_start[0] + projection
