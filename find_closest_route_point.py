import numpy as np
import multiprocessing
import pandas as pd
from functools import partial

def get_projection_and_dist_ratio(point, segment_start, segment_end):
    """
    Returns the projection C of a point P onto a line containing points A and B,
    (i.e. C is the closest point on the line AB to the point P), and also the signed
    ratio t of the displacement AC to the (parallel) displacement AB.

    The formula for projecting the point P onto the line spanned by A and B is:

    C = A + (P-A).(B-A) * (B-A) / |B-A|^2,

    where C is the projection (closest point to P), . is the dot product,
    * is scalar multiplication, and |x|^2 is the squared Euclidean norm of the vector x.

    The signed distance ratio t = + or - |AC|/|AB| is the unique scalar t such that

    C = A + t(B-A),

    and is computed by t=(P-A).(B-A) / |B-A|^2.

    The calculatios occur in "affine coordinates" since the line is not assumed
    to go through the origin.

    Parameters
    ----------
    point : array_like (1D)
        The point P to projection onto the line, represented in orthonormal coordinates.
        The shape should be (1,d), where d is the dimension of the space the point lives in.

    segment_start : array_like
        The coordinates of the first point A on the line,
        in the same orthonormal basis as for `point` P.
        The shape should be (n,d), where n is the number of line segments to project onto
        and d is the same dimension as for `point` P.

    segment_end : array_like
        The coordinates of the second point B on the line,
        in the same orthonormal basis as for `point` P.
        The shape should be (n,d), where n is the number of line segments to project onto
        and d is the same dimension as for `point` P.


    Returns
    -------
    closest_point : numpy.array
        The point C on the line AB closest to `point` P.
        The shape will be (n,d), the same as that of `segment_start` and `segment_start`.

    dist_ratio : float or ndarray
        The signed distance ratio t satisfying C = A + t(B-A).
        If n=1, a float is returned. Otherwise the shape will be (1,n).
    """
    # segment_start, segment_end = two_points_on_line
    direction = segment_end - segment_start
    #calc the squared Euclidean norm of the direction
    euc_norm = np.sum(direction**2, axis=1)
    #replace 0 values with very small number
    euc_norm_fixed = np.where(euc_norm==0, 0.00001, euc_norm)
    # print(point, segment_start, direction)
    dist_ratio = ((point - segment_start).dot(direction.T) 
                    / 
                    euc_norm_fixed
                    
             ).reshape(-1,1)
                    
                    # we reshape so that adjacent point direction
    # projection = dist_ratio * direction
    return (segment_start + dist_ratio*direction, dist_ratio)

def get_closeset_point_parallel(df, full_shapes_gtfs, shape_id):
    '''
    '''
    df.loc[:,'closest_pt_on_route_tuple'] = (df.
                                        loc[:,['vehicle_lat',
                                        'vehicle_long',
                                        'shape_pt_sequence']]
                                        .apply(lambda x: 
                                        find_closest_point_on_route(
                                                                full_shapes_gtfs, 
                                                                shape_id, 
                                                                np.float(x.vehicle_lat), 
                                                                np.float(x.vehicle_long), 
                                                                x.shape_pt_sequence),
                                        axis=1
                                       ))
    return df

def get_closeset_point_process(df, full_shapes_gtfs, shape_id):
    #n_pools = multiprocessing.cpu_count() - 1
    n_pools = 2
    pool = multiprocessing.Pool(n_pools)
    num_splits = n_pools
    df_list = np.array_split(df, num_splits)
    df_w_closest_pt = pd.concat(pool.map(partial(get_closeset_point_parallel, 
                                                          full_shapes_gtfs=full_shapes_gtfs,
                                                          shape_id=shape_id
                                                          ),
                                                  df_list))
    pool.close()
    pool.join()
    return df_w_closest_pt

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

def find_closest_point_on_route(shapes_df, shape_id, veh_lat, veh_lon, closest_shape_pt_sequence):
    """
    Find the closest point on the route to the vehicle's location,
    and the shape distance traveled to that point.
    """
    # Get the shapes_df data for the closest shape point and its adjacent point(s) on the shape.
    shape_pt_data = get_shape_point_data(shapes_df, shape_id, closest_shape_pt_sequence)
    adjacent_shape_pt_data = get_adjacent_shape_point_data(shapes_df, shape_pt_data.index[0])

    # Put longitude before lattitude to have the coordinates ordered (x,y)
    vehicle_pt = np.array([veh_lon, veh_lat])
    closest_shape_pt = shape_pt_data[['shape_pt_lon', 'shape_pt_lat']].values
    if len(closest_shape_pt) > 1: #in case you have multiple gtfs rows
        closest_shape_pt = closest_shape_pt[0].reshape(1,2)
    else:
        pass

    adjacent_pts = adjacent_shape_pt_data[['shape_pt_lon', 'shape_pt_lat']].values

    # Find the closest point and distance ratio for each of the route segments.
    # This will return either 1 or 2 points and ratios, depending on the number of adjacent points.
    closest_pt, dist_ratio = get_projection_and_dist_ratio(vehicle_pt, closest_shape_pt, adjacent_pts)

    # Find the squared distance from the vehicle to each of the projections
    # so we can choose the closest point
    dist_squared = np.sum((vehicle_pt - closest_pt)**2, axis=1)

    # Find the point and distance ratio corresponding to the closest distance to the route
    # by first getting the index of the smallest distance.
    min_index = np.argmin(dist_squared)
    # Reset closest_pt, dist_ratio to be a single point and a single distance,
    # rather than arrays of possibly two points and distances.
    closest_pt = closest_pt[min_index]
    dist_ratio = dist_ratio.reshape(-1)[min_index]

    # Get the shape distance traveled for the two endpoints of the line segment,
    # i.e. the shape distance traveled to the original closest shape point
    # and to the other end of the segment that the closest route point lies on,
    # using min_index to find the correct adjacent shape point for the segment.
    closest_shape_dist = shape_pt_data.iloc[0].shape_dist_traveled
    next_shape_dist = adjacent_shape_pt_data.iloc[min_index].shape_dist_traveled

    # Using the shape distances for the endpoints, compute the shape distance for the closest point
    # Note that if next_shape_dist < closest_shape_dist, this will do the correct thing and
    # subtract the appropriate distance fromm closest_shape_dist.
    shape_dist_traveled = closest_shape_dist + dist_ratio * (next_shape_dist - closest_shape_dist)

    return (closest_pt, shape_dist_traveled)




#
