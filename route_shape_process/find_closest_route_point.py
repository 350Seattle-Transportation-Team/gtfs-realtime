import numpy as np

#Hmm, actually, we want this function to also return the scalar ratio t
#such that C = A + t(B-A), i.e. t=(P-A).(B-A) / |B-A|^2,
#because we need t to calculate the shape distance to the point C
#using the shape distances to A and B.
def get_projection_and_dist_ratio(point, segment_start, segment_end):
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
    # segment_start, segment_end = two_points_on_line
    direction = segment_end - segment_start
    dist_ratio = (point - segment_start).dot(direction) / np.sum(direction**2)
    # projection = dist_ratio * direction
    return segment_start + dist_ratio*direction, dist_ratio

def get_shape_point_data(gtfs_shapes_df, shape_id, shape_pt_sequence=None, shape_point_lat=None, shape_point_lat_lon=None):
    """
    Find the row in the GTFS shapes dataframe for the (unique) point with the given
    shape_id and either (1) shape_pt_sequence or (2) lattitude, longitude.
    """
    if shape_pt_sequence is not None:
        shape_point_row = gtfs_shapes_df.loc[
            (gtfs_shapes_df.shape_id==shape_id)
            & (gtfs_shapes_df.shape_pt_sequence==shape_pt_sequence)
            ]
    elif (shape_point_lat is not None) and (shape_point_lat_lon is not None):
        shape_point_row = gtfs_shapes_df.loc[
            (gtfs_shapes_df.shape_id==shape_id)
            & (gtfs_shapes_df.shape_pt_lat==shape_point_lat)
            & (gtfs_shapes_df.shape_pt_lon==shape_point_lat_lon)
            ]
    else:
        raise ValueError("Must pass either `shape_pt_sequence` or `shape_point_lat` and `shape_point_lon`.")

    return shape_point_row

def get_adjacent_shape_point_data(gtfs_shapes_df, shape_point_index, use_index=True, use_shape_pt_sequence=False):
    """
    Gets the rows of the 2 adjacent points to the shape point with the specified index
    in the GTFS shapes dataframe.

    The function can determine adjacency of points in the shape using (1) the points' indices (default),
    in the shapes dataframe, (2) the points' 'shape_pt_sequence' field, or (3) both,
    in which case it will check to make sure the two methods returned the same result.
    """
    point_data = gtfs_shapes_df.loc[shape_point_index]

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
        df_s = gtfs_shapes_df[
                    (gtfs_shapes_df.shape_id==point_data.shape_id)
                    & (np.abs(gtfs_shapes_df.shape_pt_sequence-point_data.shape_pt_sequence)==1)
                    ]
        adjacent_point_data = df_s

    if use_index and use_shape_pt_sequence:
        assert all(df_i == df_s), f"Different sets of adjacent shape points found:\n{df_i}\n{df_s}"

    if not any([use_index, use_shape_pt_sequence]):
        raise ValueError("At least one of the methods for finding adjacent points must be True.")

    return adjacent_point_data

# #Deprecated
# def find_adjacent_shape_point_data(shape_point_lat, shape_point_lat_lon, gtfs_shapes_df, shape_id):
#     """
#     Finds the 2 adjacent points to the specified shape point.
#     """
#     # mask = ((gtfs_shapes_df.shape_pt_lat==shape_point_lat)
#     #         & (gtfs_shapes_df.shape_pt_lon==shape_point_lat_lon)
#     #         & (gtfs_shapes_df.shape_id==shape_id))
#     #gtfs_shapes_df[gtfs_shapes_df]
#     point_seq_number = gtfs_shapes_df.loc[
#         (gtfs_shapes_df.shape_pt_lat==shape_point_lat)
#         & (gtfs_shapes_df.shape_pt_lon==shape_point_lat_lon)
#         & (gtfs_shapes_df.shape_id==shape_id)
#         ].shape_pt_sequence.values
#     # print(point_seq_number)
#     df = gtfs_shapes_df[
#                 (gtfs_shapes_df.shape_id==shape_id)
#                 & (np.abs(gtfs_shapes_df.shape_pt_sequence-point_seq_number)==1)
#                 ]
#     # print(np.abs(gtfs_shapes_df.shape_pt_sequence-point_seq_number))
#     return df

#Not done yet...
def find_closest_point_on_route(shapes_df, shape_id, veh_lat, veh_lon, closest_shape_pt_sequence):
    """
    Find the closest point on the route to the vehicle's location.
    """
    shape_pt_data = get_shape_point_data(shapes_df, shape_id, closest_shape_pt_sequence)
    adjacent_shape_pt_data = get_adjacent_shape_point_data(shapes_df, shape_pt_data.index[0])

    # Put longitude before lattitude to have the coordinates ordered (x,y)
    vehicle_pt = np.array([veh_lon, veh_lat])
    closest_shape_pt = shape_pt_data[['shape_pt_lon', 'shape_pt_lat']].values
    # closest_shape_pt = np.array([shape_point_data.shape_pt_lon, shape_point_data.shape_pt_lat])
    adjacent_pts = [coordinates for coordinates in
                    adjacent_shape_pt_data[['shape_pt_lon', 'shape_pt_lat']].values]

    # Find the closest point and distance ratio for each of the segments (1 or 2)
    closest = [get_projection_and_dist_ratio(vehicle_pt, closest_shape_pt, adjacent_pt)
                for adjacent_pt in adjacent_pts]

    if len(closest) == 1:
        closest_pt, dist_ratio = *closest
    else:
        # Choose the closer of the two points
        closest_pt, dist_ratio = closest[0]
        pass

    # Determine if closest_shape_pt is ahead of or behind the vehicle on the route.
    # If it is ahead, set  dist_ratio = -dist_ratio

    # Get the shape distance traveled for the two endpoints of the line segment

    # Using the shape distances for the endpoints, compute the shape distance for the closest point
    shape_dist_traveled = shape_pt_data.shape_dist_traveled + dist_ratio # times segment length

    return closest_pt, shape_dist_traveled




#
