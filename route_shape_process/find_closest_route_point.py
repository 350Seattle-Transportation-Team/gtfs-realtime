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

def get_shape_point_data(gtfs_shapes_df, shape_id, shape_point_lat, shape_point_lat_lon):
    """
    Find the row in the GTFS shapes dataframe for the (unique) point with the given
    lattitude, longitude, and shape_id.
    """
    return gtfs_shapes_df.loc[
        (gtfs_shapes_df.shape_id==shape_id)
        & (gtfs_shapes_df.shape_pt_lat==shape_point_lat)
        & (gtfs_shapes_df.shape_pt_lon==shape_point_lat_lon)
        ]

def get_adjacent_shape_point_data(gtfs_shapes_df, shape_point_index, use_index=True, use_shape_pt_sequence=False):
    """
    Gets the 2 adjacent points to the shape point with the specified index
    in the GTFS shapes dataframe.
    """
    point_data = gtfs_shapes_df.loc[shape_point_index]
    # print(point_data)

    if use_index:
        adjacent_indices = []
        if shape_point_index > 0:
            adjacent_indices.append(shape_point_index-1)
        if shape_point_index < len(gtfs_shapes_df)-1:
            adjacent_indices.append(shape_point_index+1)

        df_i = gtfs_shapes_df.loc[adjacent_indices]
        df_i = df_i[df_i.shape_id == point_data.shape_id]
        adjacent_point_data = df_i

    if use_shape_pt_sequence:
        # print(point_data.index)
        # print(shape_point_index)
        # point_seq_number = point_data.at[point_data.index,'shape_pt_sequence']
        # print(type(point_data.shape_pt_sequence))
        # point_seq_number = point_data.shape_pt_sequence
        df_s = gtfs_shapes_df[
                    (gtfs_shapes_df.shape_id==point_data.shape_id)
                    & (np.abs(gtfs_shapes_df.shape_pt_sequence-point_data.shape_pt_sequence)==1)
                    ]
        adjacent_point_data = df_s

    if use_index and use_shape_pt_sequence:
        assert all(df_i == df_s), "Different sets of adjacent shape points found"

    return adjacent_point_data

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
