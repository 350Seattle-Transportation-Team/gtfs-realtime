import os
import pandas as pd
import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points
import multiprocessing
from datetime import datetime
import multiprocessing
import sys
import boto3
import logging
import time
import io
from functools import partial
import math
from os import listdir
from os.path import isfile, join

#create a log file - to store details about each step
named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%Y_%m_%d_%H%M", named_tuple)
log_filename = "LOG_{}_route_shape_process.log".format(time_string)

logging.basicConfig(filename=log_filename,

                    level=logging.INFO,

                    format='%(asctime)s.%(msecs)03d %(name)s - %(levelname)s - %(message)s',

                    datefmt="%Y-%m-%d %H:%M:%S")

#you'll need to save the s3 bucket name in your environmental variables (or switch this reference)
bucket_name = "bus350-data" #os.environ["BUS_BUCKET_NAME"]

##################################################################################
##########################    UTILITY FUNCTIONS   ################################
##################################################################################

def send_log_to_s3(log_filename):
    '''
    '''
    
    s3 = boto3.client('s3')
    put_filename = "AutomationLogs/"
    
    key_filename = put_filename+log_filename
    with open(log_filename) as f:
        result_body = f.read()
    s3.put_object(Bucket=bucket_name, Body=result_body, Key=key_filename)
    os.system('rm -r {}'.format(log_filename))

def get_csv_s3_make_df(file_key):
    '''
    '''
    s3 = boto3.client('s3')
    s3_object = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(io.BytesIO(s3_object['Body'].read()))
    return df

def send_output_df_to_s3(df, s3_prefix, csv_name):
    '''
    '''
    s3 = boto3.client('s3')
    key_filename = s3_prefix+csv_name

    result_body = df.to_csv(index=False)

    s3.put_object(Bucket=bucket_name, Body=result_body, Key=key_filename)

def get_send_route_progress_file(route_name, status):
    '''
    '''
    file_key = "progress/route_progress_status.csv"
    route_status = get_csv_s3_make_df(file_key)
    route_status.set_index('route_name', inplace=True)
    route_status.loc[route_name,'status'] = status
    s3_prefix = "progress/"
    csv_name = "route_progress_status.csv"
    route_status.reset_index(inplace=True)
    send_output_df_to_s3(route_status, s3_prefix, csv_name)

def get_positions_months(month_list,position_folder_location="input_position_files/"):
    '''
    month_list = ['201809', '201810', '201811'] need to point to
    the right folder holding your h5 files
    '''

    for i, position_date in enumerate(month_list):
        if i == 0:
            positions = pd.read_hdf(f"{position_folder_location}positions_{position_date}.h5",low_memory=False)
            logging.info("{}, {}, {}".format(position_date,len(positions), positions.columns))
            full_route_positions = positions.copy()
        else:
            positions = pd.read_hdf(f"{position_folder_location}positions_{position_date}.h5",low_memory=False)
            logging.info("{}, {}, {}".format(position_date,len(positions), positions.columns))

            full_route_positions = full_route_positions.append(positions)
    full_route_positions.sort_index(inplace=True)
    return full_route_positions

##################################################################################
#    convert index from UTC to pct
##################################################################################

def convert_index_to_pct(positions_df):
    '''
    '''
    positions_UTC = positions_df.tz_localize('UTC')
    positions_pacific = positions_UTC.tz_convert('US/Pacific')
    return positions_pacific

##################################################################################
#    GET MOST USED SHAPE ID PER ROUTE ID AND DIRECTION ID
##################################################################################

def get_most_used_shape_id_per_direction(full_trip_stop_schedule, route_id, direction_id):
    '''
    '''
    full_df = pd.DataFrame()
    for name, group in full_trip_stop_schedule[full_trip_stop_schedule['route_id'] == route_id].groupby(['start_gtfs_date','end_gtfs_date']):
        #logging.info(name)
        temp_df = group.groupby(['shape_id', 'direction_id','trip_headsign']).agg({'shape_id':'count'})\
                                .rename(columns={'shape_id':'shape_id_count'})\
                                .reset_index()
        #logging.info(temp_df)
        if full_df.empty:
            full_df = temp_df
        else:
            full_df = full_df.append(temp_df)
    #############################################################################
    '''full_df has all shape_id, direction_id combinations over many gtfs
    full_df_sorted groups by 'shape_id','direction_id','trip_headsign'
    and sorts by the highest sum of shape_id_count (i.e. the most used shape)
    TODO - there are routes with many "popular" shapes - this function will only pick 1
    ideally we could find the part of the route that is consistent (usually the middle)
    and combine all observations using this common shape'''
    ###############################################################################
    full_df_sorted = full_df.groupby(['shape_id','direction_id','trip_headsign'])\
                .agg({'shape_id_count':'sum'})\
                .reset_index()\
                .sort_values('shape_id_count', ascending=False)
    
    shape_id = full_df_sorted.loc[full_df_sorted['direction_id']==direction_id].iloc[0]['shape_id']
    trip_headsign = full_df_sorted.loc[full_df_sorted['direction_id']==direction_id].iloc[0]['trip_headsign']

    return (shape_id, trip_headsign)

##################################################################################
#    add time index columns to help make a unique trip column later
#   also for filtering instead of using time index <-- if you're into that :)
##################################################################################

def add_time_index_columns(positions_w_trips):
    '''
    '''
    positions_w_trips.loc[:,'day'] = positions_w_trips.index.day
    positions_w_trips.loc[:,'year'] = positions_w_trips.index.year #need year since we're combining years
    positions_w_trips.loc[:,'month'] = positions_w_trips.index.month
    positions_w_trips.loc[:,'hour'] = positions_w_trips.index.hour
    positions_w_trips.loc[:,'dow'] = positions_w_trips.index.dayofweek
    
    return positions_w_trips

def datetime_transform_df(df):
    '''
    '''
    df['time_pct'] = df['time_pct'].apply(lambda x: pd.to_datetime(x, utc=True))
    #df['time_pct'] = df['time_pct'].dt.tz_localize('UTC')
    df['time_pct']= df['time_pct'].dt.tz_convert('US/Pacific')
    
    return df

##################################################################################
#    turn gtfs shapes file into geopandas dataframe
##################################################################################

def make_geopandas_shape_df(gtfs_shapes, shape_id):
    '''
    INPUT
    -------
    gtfs_shapes = gtfs shapes.txt file dataframe
    shape_id = one shape_id to filter on
    OUTPUT
    -------
    route_vertex_geo <-- geopandas version of a particular schedule shapes.txt geometry = route vertex points
    '''
    one_shape_df = gtfs_shapes[gtfs_shapes['shape_id'] == shape_id].copy()
    if 'start_gtfs_date' in one_shape_df.columns.tolist():
        one_shape_df.drop('start_gtfs_date',axis=1, inplace=True)

    if 'end_gtfs_date' in one_shape_df.columns.tolist():
        one_shape_df.drop('end_gtfs_date', axis=1,inplace=True)
    one_shape_df.drop_duplicates(['shape_pt_sequence'], keep='first', inplace=True)
    crs = {'init':'epsg:4326'}
    shape_geometry = [Point(xy) for xy in zip(one_shape_df.shape_pt_lon, one_shape_df.shape_pt_lat)]
    route_vertex_geo = GeoDataFrame(one_shape_df, crs=crs, geometry=shape_geometry)
    return route_vertex_geo

##################################################################################
#    function to add unique_trip_id to each row
##################################################################################

def get_unique_trip_id(row):
    unique_trip = str(row['year'])+"_"+str(row['month'])+"_"+str(row['day'])+"_"+str(row['trip_id'])+"_"+str(row['vehicle_id'])
    return unique_trip

##################################################################################
#    JOIN VEHICLE POSITION OBSERVATIONS WITH GTFS
##################################################################################

def join_positions_with_gtfs_trips(positions, gtfs_trips, start_gtfs_date, end_gtfs_date):
    '''
    start_gtfs_date = YYYY-MM-DD e.g. '2018-01-01'
    end_gtfs_date = YYYY-MM-DD e.g. '2018-01-17'
    '''
    if positions.loc[start_gtfs_date:end_gtfs_date].empty:
        positions_w_trips = positions.loc[start_gtfs_date:end_gtfs_date]
    else:
        positions_w_trips = positions.loc[start_gtfs_date:end_gtfs_date].merge(gtfs_trips[['route_id',
                                                                                        'trip_id',
                                                                                        'direction_id',
                                                                                        'shape_id']],
                                                                    how='left',
                                                                    on=['route_id','trip_id'])
        positions_w_trips.loc[:,'month_day_trip_veh'] = positions_w_trips.apply(get_unique_trip_id, axis=1)

    return positions_w_trips

##################################################################################
#    find closest node to an observed raw bus lat/lon
#   alternatively could use shapely.ops nearest_points
##################################################################################

def get_nearest_route_node_sindex(row, route_vertex_geo):
    '''
    INPUT
    -----------
    row = a df row, each row is 1 vehicle observation
    route_vertex_geo = geodataframe from route shape
    OUTPUT
    -------
    shape_pt_sequence = the pt sequence along the route for the nearest node
    to the vehicle location
    ---------
    given a raw GPS coordinate from one bus away, get the closest
    bus route shape vertex node
    since we're using a spatial index, make sure the multiprocess chunks are large
    there's a cost to creating the spatial index but there's also a big benefit
    over a lot of observations vs. shapely nearest_point
    '''
    s_index = route_vertex_geo.sindex
    veh_coords = (row['vehicle_long'],row['vehicle_lat'])
    nearest_idx = list(s_index.nearest(veh_coords))
    if len(nearest_idx) == 0:
        shape_pt_sequence = 9999
    else:
        shape_pt_sequence = route_vertex_geo.iloc[nearest_idx]['shape_pt_sequence'].values[0]
    return shape_pt_sequence

def get_close_node_parallel(df, route_vertex_geo):
    '''
    '''
    df['shape_pt_sequence'] = df.apply(lambda x: get_nearest_route_node_sindex(x, 
                                                                        route_vertex_geo), 
                                        axis=1).copy()
    return df

def get_close_node_process(df, route_vertex_geo):
    #n_pools = multiprocessing.cpu_count() - 1
    n_pools = 2
    pool = multiprocessing.Pool(n_pools)
    num_splits = n_pools
    df_list = np.array_split(df, num_splits)
    positions_w_near_node_df = pd.concat(pool.map(partial(get_close_node_parallel, 
                                                          route_vertex_geo=route_vertex_geo
                                                          ),
                                                  df_list))
    pool.close()
    pool.join()
    positions_w_near_node_df = positions_w_near_node_df.merge(
                            route_vertex_geo[['shape_pt_sequence','shape_dist_traveled','shape_pt_lat','shape_pt_lon']],
                        how='left', on='shape_pt_sequence')
    return positions_w_near_node_df

def join_tripstart(distance_time_list_df, full_trip_stop_schedule, trip_id_with_starttime):
    '''
    '''
    normal_hours = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    sched_col_to_keep = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence',
                        'stop_name','stop_lat', 'stop_lon', 
        'route_id', 'trip_headsign', 'shape_id','route_short_name', 'route_desc']
    #drop hours near '0' because the hour comparison gets messed up - gtfs has hours > 23 :(
    full_trip_stop_schedule = full_trip_stop_schedule[sched_col_to_keep]
    distance_time_list_df.reset_index(inplace=True)
    positions_w_near_node_df = distance_time_list_df[distance_time_list_df.hour.isin(normal_hours)]
    position_w_node_schedule = positions_w_near_node_df.merge(full_trip_stop_schedule,how='left',
                                                        left_on=['trip_id','route_id','shape_id','shape_pt_sequence'], 
                                                        right_on=['trip_id','route_id','shape_id','stop_sequence'])
    position_w_node_schedule.drop_duplicates(['month_day_trip_veh','shape_pt_sequence'], keep='last', inplace=True)
    position_w_node_schedule['time_pct'] = position_w_node_schedule['time_pct'].dt.tz_convert('US/Pacific')

    position_w_node_schedule = position_w_node_schedule.merge(trip_id_with_starttime, how='left', on='trip_id')

    position_w_node_schedule['trip_start_time'] = position_w_node_schedule['trip_start_time'].apply(pd.to_datetime)
        
    #take the time at every node and subtract the scheduled start time
    position_w_node_schedule['actual_time_from_scheduled_start'] = (((position_w_node_schedule['time_pct'].dt.hour)*60+
                                                    position_w_node_schedule['time_pct'].dt.minute+
                                                    (position_w_node_schedule['time_pct'].dt.second)/60) - 
                                                    ((position_w_node_schedule.loc[:,'trip_start_time'].dt.hour)*60+
                                                    position_w_node_schedule.loc[:,'trip_start_time'].dt.minute+
                                                    (position_w_node_schedule.loc[:,'trip_start_time'].dt.second)/60))
    
    position_w_node_schedule['arrival_time'] = position_w_node_schedule['arrival_time'].apply(pd.to_datetime)
        
    #take the scheduled time at every stop and subtract the scheduled start time
    position_w_node_schedule['scheduled_time_from_scheduled_start'] = (((position_w_node_schedule['arrival_time'].dt.hour)*60+
                                                    position_w_node_schedule['arrival_time'].dt.minute+
                                                    (position_w_node_schedule['arrival_time'].dt.second)/60) - 
                                                    ((position_w_node_schedule.loc[:,'trip_start_time'].dt.hour)*60+
                                                    position_w_node_schedule.loc[:,'trip_start_time'].dt.minute+
                                                    (position_w_node_schedule.loc[:,'trip_start_time'].dt.second)/60))

    return position_w_node_schedule


def calc_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    return meters
    Note this is an approximation based on great circle distance:
    https://en.wikipedia.org/wiki/Great-circle_distance
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    km = (6371 * c)
    return km*1000


if __name__ == "__main__":
    '''
    first trial - needs to be in the same folder as the following files:
    - see notebook 01 for transforming gtfs
    "gtfs_routes_2018-08-15_2018-12-12.csv"
    "gtfs_shapes_2018-08-15_2018-12-12.csv"
    "gtfs_trips_2018-08-15_2018-12-12.csv"
    "gtfs_2018-08-15_2018-12-12.csv"
    usage route_shape_process_scripts.py <route_short_name>
    <route_short_name> can be a list of single route [15] or [1,18,E Line]
    -- if you put "all" instead of <route_short_name> the process will run
    for all routes 
    NOTE - you may have to change the progress csv file location
    '''

    gtfs_merge_file_path = "../data/gtfs_merge/"
    agg_filename = [f for f in listdir(gtfs_merge_file_path) if isfile(join(gtfs_merge_file_path, f)) 
                    and 'agg' in f][0]
    routes_filename = [f for f in listdir(gtfs_merge_file_path) if isfile(join(gtfs_merge_file_path, f)) 
                    and 'routes' in f][0]
    shapes_filename = [f for f in listdir(gtfs_merge_file_path) if isfile(join(gtfs_merge_file_path, f)) 
                    and 'shapes' in f][0]
    trips_filename = [f for f in listdir(gtfs_merge_file_path) if isfile(join(gtfs_merge_file_path, f)) 
                    and 'trips' in f][0]

    logging.info("grabbing csv")
    full_routes_gtfs = pd.read_csv(f"{gtfs_merge_file_path}{routes_filename}", 
                                    low_memory=False)
    full_shapes_gtfs = pd.read_csv(f"{gtfs_merge_file_path}{shapes_filename}", 
                                    low_memory=False)
    full_trips_gtfs = pd.read_csv(f"{gtfs_merge_file_path}{trips_filename}", 
                                    low_memory=False)
    full_trip_stop_schedule = pd.read_csv(f"{gtfs_merge_file_path}{agg_filename}", 
                                    low_memory=False)

    tripid_w_starttime = full_trip_stop_schedule.groupby('trip_id')\
                        .agg({'trip_start_time':'min'})\
                        .reset_index()


    #cheap and dirty way to make the route_name to id dictionary
    route_name_to_id_dict = dict(zip(full_routes_gtfs.route_short_name.tolist(),
                                full_routes_gtfs.route_id.tolist()))

    logging.info("grabbing arguments")
    route_of_interest_input = sys.argv[1]
    logging.info("route of interest argument = {}".format(route_of_interest_input))

    if route_of_interest_input == 'all':
        
        # route progress .csv is created so that you can a) track progress and b) start where you left off
        # this could also be stored locally but I run this on a virtual computer so it's
        # handy to send it to S3
        s3_prefix = "progress/"
        csv_name = "route_progress_status.csv"
        route_status_file_key = s3_prefix+csv_name
        route_status_df = get_csv_s3_make_df(route_status_file_key)

        remaining_routes = route_status_df[route_status_df['status']=='not_started']['route_name'].unique()
    else:
        route_of_interest_input = route_of_interest_input.strip("[]").split(",")
        remaining_routes = route_of_interest_input
    month_list = ['201809', '201810', '201811']
    all_route_positions = get_positions_months(month_list)
    #
    for route_of_interest in remaining_routes:
        logging.info("starting work on route name = {}".format(route_of_interest))
        route_of_interest_id = route_name_to_id_dict[route_of_interest]

        logging.info("running get_positions_months_route_id")
        single_route_positions = all_route_positions[all_route_positions['route_id']==route_of_interest_id].copy()
        
        if single_route_positions.empty:
            logging.info("skipping route name {} no positions".format(route_of_interest))
            get_send_route_progress_file(route_of_interest, 'no_positions')
            pass

        else:
            single_route_positions = convert_index_to_pct(single_route_positions)
            single_route_positions = add_time_index_columns(single_route_positions)

            direction_id_list = [0,1]
            for direction in direction_id_list:
                shape_id_of_interest, trip_headsign = get_most_used_shape_id_per_direction(full_trip_stop_schedule, route_of_interest_id, direction)

                logging.info("starting process for direction = {}, shape_id = {}".format(direction, shape_id_of_interest))
                logging.info("make one route_vertex_geo from shape_id {}".format(shape_id_of_interest))
                route_vertex_geo = make_geopandas_shape_df(full_shapes_gtfs, shape_id_of_interest)

                positions_w_trips = {}
                for name, group in full_trips_gtfs.groupby(['start_gtfs_date','end_gtfs_date']):
                    #print(name)
                    positions_w_trips[name] = join_positions_with_gtfs_trips(single_route_positions, group, name[0], name[1])
                for idx, dict_group in enumerate(positions_w_trips.keys()):
                    #print(dict_group)
                    if positions_w_trips[dict_group].empty:
                        pass
                    else:
                        if idx == 0:
                            unpacked_positions_full = positions_w_trips[dict_group].copy()
                        else:
                            unpacked_positions_full = unpacked_positions_full.append(positions_w_trips[dict_group])
                
                unpacked_positions_one_shape = unpacked_positions_full[unpacked_positions_full['shape_id']==shape_id_of_interest]
                logging.info("getting close nodes")
                positions_w_near_node_df = get_close_node_process(unpacked_positions_one_shape, route_vertex_geo)
                logging.info("datetime transform")
                positions_w_near_node_datetime = datetime_transform_df(positions_w_near_node_df)

                logging.info("join with gtfs schedule on shape_pt_sequence & calculate times from trip_start_time")
                position_w_node_schedule = join_tripstart(positions_w_near_node_datetime, 
                                                            full_trip_stop_schedule, 
                                                            tripid_w_starttime)

                #it's helpful to have this distance for debugging
                position_w_node_schedule['distance_btw_veh_and_shape'] = position_w_node_schedule\
                                                                .apply(lambda x: calc_distance(x['vehicle_lat'],
                                                                x['vehicle_long'], 
                                                                x['shape_pt_lat'],
                                                               x['shape_pt_lon']), axis=1)
                

                csv_name = 'transformed/route_{}_{}_shape_{}_node_trips_w_nearest_2018-08-15_2018-12-11.csv'.format(
                                                route_of_interest,"".join(trip_headsign.replace("/","-").split(" ")) ,shape_id_of_interest)
                s3_prefix = "route_shape_files/"
                logging.info("sending file to s3")
                send_output_df_to_s3(position_w_node_schedule, s3_prefix,  csv_name)
                
            get_send_route_progress_file(route_of_interest, 'done')
    send_log_to_s3(log_filename)