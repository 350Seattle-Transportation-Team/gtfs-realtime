import numpy as np

#Hmm, actually, we want this function to also return the scalar ratio t
#such that C = A + t(B-A), i.e. t=(P-A).(B-A) / |B-A|^2,
#because we need t to calculate the shape distance to the point C
#using the shape distances to A and B.
def project_onto_line(point, two_points_on_line):
    """
    Returns the projection of a point onto a line, i.e. the closest point on the line to the point.

    The formula for projecting the point P onto the line spanned by A and B is:

    C = A + (P-A).(B-A) * (B-A) / |B-A|^2,

    where C is the projection (closest point to P), . is the dot product,
    * is scalar multiplication, and |x|^2 is the squared Euclidean norm of the vector x.

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
    closest_point : numpy.array (1D)
        The point on the line closest to `point`.
        The shape will be (1,d), the same as that of `point`.
    """
    segment_start, segment_end = two_points_on_line
    direction = segment_end - segment_start
    projection = (point - segment_start).dot(direction) * direction / np.sum(direction**2)
    return segment_start + projection

#Still working on this...
def find_adjacent_shape_point_data(shape_point_lat, shape_point_lat_lon, gtfs_shapes_df, shape_id):
    """
    Finds the 2 adjacent points to the specified shape point.
    """
    # mask = ((gtfs_shapes_df.shape_pt_lat==shape_point_lat)
    #         & (gtfs_shapes_df.shape_pt_lon==shape_point_lat_lon)
    #         & (gtfs_shapes_df.shape_id==shape_id))
    #gtfs_shapes_df[gtfs_shapes_df]
    point_seq_number = gtfs_shapes_df.loc[
        (gtfs_shapes_df.shape_pt_lat==shape_point_lat)
        & (gtfs_shapes_df.shape_pt_lon==shape_point_lat_lon)
        & (gtfs_shapes_df.shape_id==shape_id)
        ].shape_pt_sequence.values
    # print(point_seq_number)
    df = gtfs_shapes_df[
                (gtfs_shapes_df.shape_id==shape_id)
                & (np.abs(gtfs_shapes_df.shape_pt_sequence-point_seq_number)==1)
                ]
    # print(np.abs(gtfs_shapes_df.shape_pt_sequence-point_seq_number))
    return df
