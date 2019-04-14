import os
import pandas as pd
import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points
import networkx as nx
import multiprocessing
from datetime import datetime
import multiprocessing
import matplotlib.pyplot as plt 

##################################################################################
#                        All steps in a row
##################################################################################

def get_full_trips_all_steps(gtfs_stops,
                            gtfs_stop_times,
                            gtfs_trips,
                            gtfs_shapes,
                            gtfs_routes,
                            input_dict,
                            raw_positions_df):
    '''
    '''
    positions_pacific = convert_index_to_pct(raw_positions_df)
    positions_pacific = add_time_index_columns(positions_pacific)
    print("getting month of interest: {}".format(input_dict['yearmonth_of_interest']))
    positions_month_pct = positions_pacific.loc[input_dict['yearmonth_of_interest']]
    print("getting route of interest: {}".format(input_dict['route_id_of_interest']))
    key_route_positions = positions_month_pct[
                        (positions_month_pct['route_id'] == input_dict['route_id_of_interest'])].copy()

    trip_stops_w_name_route = trip_stop_schedule(gtfs_stops, 
                                             gtfs_stop_times, 
                                             gtfs_trips, 
                                             gtfs_routes)

    print("join position_df with trips GTFS")
    positions_w_trips = join_positions_with_gtfs_trips(key_route_positions, 
                                                   gtfs_trips, 
                                                   input_dict['gtfs_startdate'], 
                                                   input_dict['gtfs_enddate'])

    print("get route vertex geo and graph G")
    route_vertex_geo, G = get_route_vertex_graph(gtfs_shapes, input_dict['one_shape_id'])

    print("get unique trip list and position geo")

    unique_trip_list, positions_w_trips_geo = get_trip_from_shape_id(input_dict['one_shape_id'], 
                                                                         positions_w_trips)

    print("datetime transform position geo")
    positions_w_trips_geo = datetime_transform_positions_df(positions_w_trips_geo)

    print("getting full edge df - print progress every 100 trips")
    full_edge_df = create_full_edge_df(unique_trip_list, 
                                   positions_w_trips_geo, 
                                   route_vertex_geo, G, 
                                   input_dict['one_shape_id'])

    print("select only position rows with stops")
    full_edge_only_stops = full_edge_transformations_stopsonly(full_edge_df, 
                                                 route_vertex_geo, 
                                                 trip_stops_w_name_route)

    return full_edge_only_stops


##################################################################################
#           quick way to find the route_id for a given route_name
##################################################################################

def get_select_routeid_name(routes_gtfs, routes_of_interest_list):
    '''
    INPUT
    ------
    routes_gtfs = One Bus Away data for all routes
    routes_of_interest_list = python list of routes to select from larger dataset
                            e.g. ['D Line', 'E Line', '8']
    
    OUTPUT
    ------
    routes_of_interest_df = df of only selected routes
    route_id_to_name_dict = id key, name value
    route_name_to_id_dict = name key, id value
    '''
    route_id_to_name_dict = {}
    route_name_to_id_dict = {}
    for i, route in enumerate(routes_of_interest_list):
        if i == 0:
            routes_of_interest_df = routes_gtfs[routes_gtfs['route_short_name'] == route]
            route_id_to_name_dict[routes_of_interest_df['route_id'].values[0]] = route
            route_name_to_id_dict[route] = routes_of_interest_df['route_id'].values[0]
        else:
            partial_df = routes_gtfs[routes_gtfs['route_short_name'] == route]
            routes_of_interest_df = routes_of_interest_df.append(partial_df)
            route_id_to_name_dict[partial_df['route_id'].values[0]] = route
            route_name_to_id_dict[route] = partial_df['route_id'].values[0]
    return (routes_of_interest_df, route_id_to_name_dict, route_name_to_id_dict)

##################################################################################
#TODO this function is not done
##################################################################################

def parallelize_dataframe(df, func, func_list):
    '''
    NOT DONE
    '''
    num_partitions = 10 #number of partitions to split dataframe
    num_cores = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(num_cores)
    full_data = pd.concat(pool.map(func, func_list))
    pool.close()
    pool.join()
    #return data

##################################################################################
#    function to return count of shape_ids associated with a given route
##################################################################################

def get_shapes_count_from_route_id(trips_df, route_id):
    '''
    '''
    shape_trips_df = trips_df[trips_df['route_id'] == route_id].copy()
    
    return shape_trips_df.groupby(['route_id','shape_id', 'direction_id']).agg({'shape_id':'count'})

##################################################################################
#    function to add unique_trip_id to each row
##################################################################################

def get_unique_trip_id(row):
    unique_trip = str(row['year'])+"_"+str(row['month'])+"_"+str(row['day'])+"_"+str(row['trip_id'])+"_"+str(row['vehicle_id'])
    return unique_trip

##################################################################################
#    transform vehicle locations into geopandas dataframe with 
#    geometry column = vehicle point 
##################################################################################

def create_vehicle_geo(vehicle_location_df, shape_id):
    '''
    INPUT
    --------
    vehicle_location_df = vehicle location dataframe - 1 row per One Bus Away observation
    shape_id = one shape_id to filter on

    OUTPUT
    ---------
    vehicle_location_geo = geopandas dataframe - geometry = vehicle point
    '''
    vehicle_location_for_shape = vehicle_location_df[vehicle_location_df['shape_id'] == shape_id].copy()
    vehicle_location_for_shape.loc[:,'month_day_trip_veh'] = vehicle_location_for_shape.apply(get_unique_trip_id, axis=1)
    crs = {'init':'epsg:4326'}
    vehicle_geometry = [Point(xy) for xy in zip(vehicle_location_for_shape.vehicle_long, 
                                                    vehicle_location_for_shape.vehicle_lat)]
    vehicle_location_geo = GeoDataFrame(vehicle_location_for_shape, crs=crs, geometry=vehicle_geometry)
    return vehicle_location_geo

##################################################################################
#    get list of all trips along the shape_id
##################################################################################

def get_unique_trip_list_df(vehicle_position_geo, shape_id):
    '''
    INPUT
    -------
    vehicle_location_geo = geopandas dataframe - geometry = vehicle point
    shape_id = one shape_id to filter on

    OUTPUT
    -------
    unique_trip_list <-- all the unique trips (month_day_trip_id)
    '''
    vehicle_position_one_shape = vehicle_position_geo[vehicle_position_geo['shape_id'] == shape_id]
    vehicle_position_one_shape['month_day_trip_veh'] = vehicle_position_one_shape.apply(get_unique_trip_id, axis=1)
    month_day_trip_veh_df = vehicle_position_one_shape.groupby(by='month_day_trip_veh')
    unique_trip_list = list(month_day_trip_veh_df.groups.keys())
    return unique_trip_list

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
    crs = {'init':'epsg:4326'}
    shape_geometry = [Point(xy) for xy in zip(one_shape_df.shape_pt_lon, one_shape_df.shape_pt_lat)]
    route_vertex_geo = GeoDataFrame(one_shape_df, crs=crs, geometry=shape_geometry)
    return route_vertex_geo

##################################################################################
#    create graph network from shape
#       you'll need to install the networkx library for this step
#       #TODO - need to think if we can remove the graph and just use
#       general geopandas/pandas functions
##################################################################################

def create_network_fromshape(route_vertex_geo, shape_id):
    '''
    INPUT
    ------
    route_vertex_geo = king county route shapes, includes vertices for
                        each route shape
    shape_id = given a specific shape_id, create the network graph

    OUTPUT
    --------
    Networkx directional network graph G'''
    shape_name = "route_{}_geo".format(shape_id)
    shape_name = route_vertex_geo[route_vertex_geo['shape_id'] == shape_id].copy()
    shape_name = shape_name.sort_values(by='shape_pt_sequence',
                                                axis=0, ascending=True).copy()
    feature_list = [(x,y,dist) for x, y, dist in zip(shape_name.shape_pt_lon, shape_name.shape_pt_lat, shape_name.shape_dist_traveled)]
    edgelist = []
    list_len = len(feature_list)
    for i, item in enumerate(feature_list):
        if i+1 < list_len:
            x1 = item[0]
            y1 = item[1]
            dist1 = item[2]
            x2 = feature_list[i+1][0]
            y2 = feature_list[i+1][1]
            dist2 = feature_list[i+1][2]
            tot_dist = dist2-dist1
            edgelist.append(((x1,y1),(x2,y2),tot_dist))
    #create a directed graph
    G = nx.DiGraph()
    G.graph['rankdir'] = 'LR'
    G.graph['dpi'] = 120
    G.add_weighted_edges_from(edgelist, weight='dist')
    return G

##################################################################################
#    get graph edge list
#    an edge is the connector between two graph nodes
##################################################################################

def get_edge_list(loc1, loc2, G):
    '''get edge list in between two bus locations'''
    node_list = nx.dijkstra_path(G,loc1,loc2)
    edge_list = []
    for i, node in enumerate(node_list):
        if i+1 < len(node_list):
            node1 = node_list[i]
            node2 = node_list[i+1]
            edge_list.append((node1,node2))
    return edge_list

##################################################################################
#    find closest node to an observed raw bus lat/lon
#   alternatively could use shapely.ops nearest_points
##################################################################################

def get_close_node(raw_loc, route_vertex_geo):
    '''
    INPUT
    -----------
    raw_loc = lat/lon coord in a tuple form. e.g. (-122.30731999999999, 47.625236)
    route_vertex_geo = geodataframe from route shape
    OUTPUT
    -------
    near_node = nearest route node
    ---------
    given a raw GPS coordinate from one bus away, get the closest
    bus route shape vertex node
    we'll later use this node to update the graph attributes
    '''
    veh_pt = Point(raw_loc)
    route_vertex_geo['distance'] = route_vertex_geo.distance(veh_pt)
    route_vertex_geo_sorted = route_vertex_geo.sort_values(by=['distance'], axis=0, ascending=True)
    #filter for distance too far away happens later - keep all distances here
    distance = route_vertex_geo_sorted.iloc[0]['distance']
    near_node = route_vertex_geo_sorted.iloc[0].geometry.coords[:][0]
    node_num = route_vertex_geo_sorted.iloc[0]['shape_pt_sequence']
    return near_node, node_num, distance

##################################################################################
#    STILL IN PROGRESS ##
#       #TODO want to convert all functions so they take advantage of pandas apply
##################################################################################

def get_close_node_apply(veh_pt, route_vertex_geo):
    '''
    INPUT
    -----------
    raw_loc = lat/lon coord in a tuple form. e.g. (-122.30731999999999, 47.625236)
    route_vertex_geo = geodataframe from route shape
    OUTPUT
    -------
    near_node = nearest route node
    ---------
    given a raw GPS coordinate from one bus away, get the closest
    bus route shape vertex node
    we'll later use this node to update the graph attributes
    '''
    route_vertex_geo['distance'] = route_vertex_geo.distance(veh_pt)
    route_vertex_geo_sorted = route_vertex_geo.sort_values(by=['distance'], axis=0, ascending=True)
    #add filter for distance too far away
    distance = route_vertex_geo_sorted.iloc[0]['distance']
    near_node = route_vertex_geo_sorted.iloc[0].geometry.coords[:][0]
    node_num = route_vertex_geo_sorted.iloc[0]['shape_pt_sequence']
    return near_node, node_num, distance

##################################################################################
#    get travel distance along the graph network path
##################################################################################

def get_travel_distance(loc1, loc2, route_vertex_geo, G):
    '''
    INPUT
    -------
    loc1 = raw bus location #1
    loc2 = raw bus location #2 - e.g. (-122.30731999999999, 47.625236)
    route_vertex_geo = geodataframe from route shape
    G = network graph of route shape
    OUTPUT
    -------
    trav_dist = distance traveled along the network between two bus locations'''
    node1, node_num, dist1 = get_close_node(loc1, route_vertex_geo)
    node2, node_num, dist2 = get_close_node(loc2, route_vertex_geo)
    trav_dist = nx.shortest_path_length(G,node1,node2, weight='dist')
    return trav_dist

##################################################################################
#    get all the trips associated with a shape
#   also filter for PEAK times 6AM-9AM, 3PM-7PM
##################################################################################

def get_trip_from_shape_id(one_shape_id, key_routes_positions):
    '''
    '''
    if len(key_routes_positions) == 0:
        print("no trips")
        return ([""],pd.DataFrame())
    else:

        key_routes_veh_trip_geo = create_vehicle_geo(key_routes_positions,
                                                        one_shape_id)

    '''print("selecting trips from 6AM-7PM")
    veh_commuter_trip_geo = key_routes_veh_trip_geo[
                                    ((key_routes_veh_trip_geo['hour']>=6)&
                                    (key_routes_veh_trip_geo['hour']<20))].copy()'''

    unique_trip_list = get_unique_trip_list_df(key_routes_veh_trip_geo, one_shape_id)

    return (unique_trip_list, key_routes_veh_trip_geo)

##################################################################################
#    FUNCTION IN PROGRESS
#   #TODO want to use multiprocessing and apply functions for faster processing
##################################################################################

def create_full_edge_df_apply(unique_trip_list, veh_commuter_trip_geo, route_vertex_geo, G, one_shape_id):
    '''
    '''
    grouped = veh_commuter_trip_geo.groupby('month_day_trip_veh')
    full_edge_df = pd.DataFrame()
    for name, group in grouped:
        partial_edge_df = group.copy()
        partial_edge_df['close_node_tuple'] = partial_edge_df.apply(lambda x: get_close_node(x['geometry'].coords[:][0], 
                                                                                          route_vertex_geo), axis=1)
        if full_edge_df.empty:
            full_edge_df = partial_edge_df.copy()
        else:
            full_edge_df = full_edge_df.append(partial_edge_df)
    return full_edge_df

##################################################################################
#    update all edges in between two bus observations - see update_edges function
#   for more details
##################################################################################


def create_full_edge_df(unique_trip_list, veh_commuter_trip_geo, route_vertex_geo, G, one_shape_id):
    ''' '''
    for trip_idx, unique_trip in enumerate(unique_trip_list):
        unique_trip_geo_df = veh_commuter_trip_geo[
                    veh_commuter_trip_geo['month_day_trip_veh']==unique_trip].copy()
        trip_id = unique_trip_geo_df['trip_id'].unique()[0]
        vehicle_id =unique_trip_geo_df['vehicle_id'].unique()[0]
        route_id = unique_trip_geo_df['route_id'].unique()[0]

        partial_edge_df = update_edges(unique_trip_geo_df, route_vertex_geo, G,
                            trip_id, vehicle_id, route_id, one_shape_id)
        if trip_idx == 0:
            full_edge_df = partial_edge_df.copy()
        else:
            full_edge_df = full_edge_df.append(partial_edge_df)
        if trip_idx % 100 == 0 and trip_idx != 0:
            print(trip_idx)
    return full_edge_df

##################################################################################
#    join gtfs files
##################################################################################

def trip_stop_schedule(gtfs_stops, gtfs_stop_times, gtfs_trips, gtfs_routes):
    '''
    '''
    trip_stops_w_names = gtfs_stop_times.merge(gtfs_stops, how='left',on='stop_id')
    trip_arrival_time = trip_stops_w_names.loc[trip_stops_w_names['stop_sequence']==1,['trip_id','stop_sequence','arrival_time']]\
                        .groupby('trip_id')\
                        .agg({'arrival_time':'max'})\
                        .reset_index()\
                        .rename(columns={'arrival_time':'trip_start_time'})
    trip_stops_w_names = trip_stops_w_names.merge(trip_arrival_time, how='left', on='trip_id')
    
    trip_stops_w_name_route = trip_stops_w_names.merge(gtfs_trips[['trip_id','route_id','direction_id','trip_headsign']], how='left',on='trip_id')
    
    trip_stops_w_name_route = trip_stops_w_name_route.merge(gtfs_routes[['route_id', 'route_short_name', 'route_desc']], how='left', on='route_id')

    return trip_stops_w_name_route

def join_positions_with_gtfs_trips(positions, gtfs_trips, start_gtfs_date, end_gtfs_date):
    '''
    start_gtfs_date = YYYY-MM-DD e.g. '2018-01-01'
    end_gtfs_date = YYYY-MM-DD e.g. '2018-01-17'
    '''
    positions_w_trips = positions.loc[start_gtfs_date:end_gtfs_date].merge(gtfs_trips[['route_id',
                                                                                      'trip_id',
                                                                                     'direction_id',
                                                                                     'shape_id']],
                                                                  how='left',
                                                                 on=['route_id','trip_id'])

    return positions_w_trips

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
#    convert time_pct to datetime and set it as index
#   setting the index, you revert back to UTC 
# - so you need to convert back to PCT
##################################################################################

def datetime_transform_positions_df(positions_w_trips):
    '''
    '''
    positions_w_trips['time_pct'] = positions_w_trips['time_pct'].apply(pd.to_datetime)
    positions_w_trips.set_index('time_pct', inplace=True)
    positions_w_trips.sort_index(inplace=True)
    positions_w_trips = positions_w_trips.tz_localize('UTC')
    positions_w_trips = positions_w_trips.tz_convert('US/Pacific')
    
    return positions_w_trips

##################################################################################
#    create route vertex graph network
##################################################################################

def get_route_vertex_graph(gtfs_shapes, one_shape_id):
    '''
    '''
    route_vertex_geo = make_geopandas_shape_df(gtfs_shapes, 
                                               one_shape_id)

    G = create_network_fromshape(route_vertex_geo, 
                                 one_shape_id)

    return (route_vertex_geo, G)

##################################################################################
#    convert arrival and trip start time to datetime
#       calculate time from schedule start and delay
##################################################################################

def full_edge_transformations_stopsonly(full_edge_df, route_vertex_geo, trip_stops_w_name_route):
    '''
    '''
    full_edge_w_stop_seq = full_edge_df.merge(route_vertex_geo[['shape_pt_lat','shape_pt_lon','shape_pt_sequence']],
                                                            how='left',left_on=['pt1_lat','pt1_lon'],
                                                            right_on=['shape_pt_lat','shape_pt_lon'])
    
    full_edge_schedule = full_edge_w_stop_seq.merge(trip_stops_w_name_route,how='left',
                                                    left_on=['trip_id','route_id','shape_id','shape_pt_sequence'], 
                                                    right_on=['trip_id','route_id','shape_id','stop_sequence'])

    full_edge_schedule['arrival_time'] = full_edge_schedule['arrival_time'].apply(pd.to_datetime)

    full_edge_schedule['trip_start_time'] = full_edge_schedule['trip_start_time'].apply(pd.to_datetime)

    full_edge_only_stops = full_edge_schedule[
                            full_edge_schedule['stop_name'].notnull()].copy()
    
    #take the time at every stop and subtract the scheduled start time
    full_edge_only_stops['time_from_scheduled_start'] = (((full_edge_only_stops.loc[:,'time_at_node'].dt.hour)*60+
                                                    full_edge_only_stops.loc[:,'time_at_node'].dt.minute+
                                                    (full_edge_only_stops.loc[:,'time_at_node'].dt.second)/60) - 
                                                    ((full_edge_only_stops.loc[:,'trip_start_time'].dt.hour)*60+
                                                    full_edge_only_stops.loc[:,'trip_start_time'].dt.minute+
                                                    (full_edge_only_stops.loc[:,'trip_start_time'].dt.second)/60))

    full_edge_only_stops['delay'] = (((full_edge_only_stops.loc[:,'time_at_node'].dt.hour)*60+
                                                    full_edge_only_stops.loc[:,'time_at_node'].dt.minute+
                                                    (full_edge_only_stops.loc[:,'time_at_node'].dt.second)/60) - 
                                                    ((full_edge_only_stops.loc[:,'arrival_time'].dt.hour)*60+
                                                    full_edge_only_stops.loc[:,'arrival_time'].dt.minute+
                                                    (full_edge_only_stops.loc[:,'arrival_time'].dt.second)/60))

    return full_edge_only_stops

def full_edge_transformations_alledges(full_edge_df, route_vertex_geo, trip_stops_w_name_route):
    '''
    '''
    full_edge_w_stop_seq = full_edge_df.merge(route_vertex_geo[['shape_pt_lat','shape_pt_lon','shape_pt_sequence']],
                                                            how='left',left_on=['pt1_lat','pt1_lon'],
                                                            right_on=['shape_pt_lat','shape_pt_lon'])
    
    full_edge_schedule = full_edge_w_stop_seq.merge(trip_stops_w_name_route,how='left',
                                                    left_on=['trip_id','route_id','shape_id','shape_pt_sequence'], 
                                                    right_on=['trip_id','route_id','shape_id','stop_sequence'])

    full_edge_schedule['arrival_time'] = full_edge_schedule['arrival_time'].apply(pd.to_datetime)

    full_edge_schedule['trip_start_time'] = full_edge_schedule['trip_start_time'].apply(pd.to_datetime)
    
    #take the time at every stop and subtract the scheduled start time
    full_edge_schedule['time_from_scheduled_start'] = (((full_edge_schedule.loc[:,'time_at_node'].dt.hour)*60+
                                                    full_edge_schedule.loc[:,'time_at_node'].dt.minute+
                                                    (full_edge_schedule.loc[:,'time_at_node'].dt.second)/60) - 
                                                    ((full_edge_schedule.loc[:,'trip_start_time'].dt.hour)*60+
                                                    full_edge_schedule.loc[:,'trip_start_time'].dt.minute+
                                                    (full_edge_schedule.loc[:,'trip_start_time'].dt.second)/60))

    return full_edge_schedule

##################################################################################
#    NEEDS WORK
#   plotting helper function 
##################################################################################

def plot_distance_vs_start_time(ax,
                                color,
                                label_suffix,
                                full_edge_only_stops, input_dict, hour_list_interest, dow_list_interest):
    '''
    '''
 
    new_unique_trip_list = list(
                            full_edge_only_stops[
                            (full_edge_only_stops['hour'].isin(hour_list_interest))&\
                            (full_edge_only_stops['dow'].isin(dow_list_interest))]
                            ['month_day_trip_veh'].unique())
    
    special_hour_trips = full_edge_only_stops.loc[full_edge_only_stops.month_day_trip_veh.isin(new_unique_trip_list)].copy()
    
    
    special_hour_trips = special_hour_trips.groupby(['month_day_trip_veh','shape_dist_traveled'])\
                                .agg({'time_from_scheduled_start':'max'})\
                                .reset_index()
            
    special_hour_trips_avg = special_hour_trips.groupby(['shape_dist_traveled'])\
                            .agg({'time_from_scheduled_start':'mean'})\
                            .reset_index()
            
    special_hour_trips_95 = special_hour_trips.groupby(['shape_dist_traveled'])\
                            .quantile(.95)\
                            .reset_index()

    special_hour_trips_05 = special_hour_trips.groupby(['shape_dist_traveled'])\
                        .quantile(.05)\
                        .reset_index()
            
    #ax.plot(special_hour_trips['time_from_scheduled_start'].values,
    #            special_hour_trips['shape_dist_traveled'].values, 
    #           c='black',
    #           alpha=0.25)
    ax.plot(special_hour_trips_avg['time_from_scheduled_start'].values,
                special_hour_trips_avg['shape_dist_traveled'].values, 
                c=color,
            alpha=0.6,
            label='average_{}'.format(label_suffix))
    ax.plot(special_hour_trips_95['time_from_scheduled_start'].values,
                special_hour_trips_95['shape_dist_traveled'].values, 
                c=color,
            alpha=0.2,
            label='quantile .95_{}'.format(label_suffix))
    ax.plot(special_hour_trips_05['time_from_scheduled_start'].values,
                special_hour_trips_05['shape_dist_traveled'].values, 
                c=color,
            alpha=0.2,
            label='quantile .05_{}'.format(label_suffix))
    ax.set_xlabel('time_from_scheduled_start',fontsize=14)
    ax.set_ylabel('distance_traveled_on_route', fontsize=14)
    
    ax.legend()
    return ax

##################################################################################
#    NEEDS WORK
#   function to interpolate between bus observations and update
#   a "time_at_node" for each stop
#   limitation - only need bus observations to update the edges/nodes
#   some locations have more updates than others  
##################################################################################

def update_edges(vehicle_geo, route_vertex_geo, G,
                    trip_id, vehicle_id, route_id, shape_id):
    #(unique_trip_geo_df, route_vertex_geo, G,
    #trip_id, vehicle_id, route_id, shape_id)
    '''still need a function to separate all the vehicle data for one route into unique
    trips to update the graph'''
    len_veh_locs = len(vehicle_geo)
    vehicle_geo_sorted = vehicle_geo.sort_index()
    month_day_trip_veh = vehicle_geo_sorted['month_day_trip_veh'].unique()[0]
    full_edge_df = pd.DataFrame()
    error_counter = 1
    for veh_row_idx, row in enumerate(vehicle_geo_sorted.iterrows()):
        if veh_row_idx + 1 < len_veh_locs:
            loc1 = vehicle_geo_sorted['geometry'].iloc[veh_row_idx].coords[:][0]
            loc2 = vehicle_geo_sorted['geometry'].iloc[veh_row_idx+1].coords[:][0]
            node1, node_num1, dist1 = get_close_node(loc1, route_vertex_geo)
            node2, node_num2, dist2 = get_close_node(loc2, route_vertex_geo)
            #print("node_num1 {}, node_num2 {}, dist1 {}, dist2 {}".format(node_num1, node_num2,
            #                                                              dist1, dist2))
            if (node_num1 < node_num2) and (dist1 < 0.2) and (dist2 < 0.2):
                '''print("coord1 {}, coord2 {} --> closest coord1 {}, coord2 {}".format(loc1, loc2,
                                                                                     node1, node2))
                print("node_num1 {}, node_num2 {}, dist1 {}, dist2 {}".format(node_num1, node_num2,
                                                                             dist1, dist2))'''
                try:
                    trav_dist = get_travel_distance(node1, node2, route_vertex_geo, G)
                    time1 = vehicle_geo_sorted.index[veh_row_idx]
                    time2 = vehicle_geo_sorted.index[veh_row_idx+1]
                    #print("time1 = {}, time2 = {}".format(time1,time2))
                    time_delta = time2 - time1
                    time_delta_hours = time_delta.total_seconds() / (60 * 60)
                    time_delta_half = time_delta.total_seconds() / 2
                    time_midway = time1 + pd.Timedelta('{} seconds'.format(time_delta_half))
                    hour = time_midway.hour
                    dow = time_midway.dayofweek
                    day = time_midway.day
                    month = time_midway.month
                    time_id = "{}_{}".format(dow, hour)
                    #travel rate in miles per hour
                    trav_rate_update = trav_dist/(time_delta_hours*5280)
                    '''need to find all edges in between loc1 and loc2 and update them'''
                    edge_list = get_edge_list(node1, node2, G)
                    #print("len edge_list for row #{} = {}".format(i,len(edge_list)))
                    edge_for_upload = []
                    col_list = ['month_day_trip_veh',
                                'pt1_lon', 'pt1_lat', 'pt2_lon', 'pt2_lat',
                                'dist_to_pt1','dist_to_pt2',
                                'start_time', 'end_time', 'mid_time', 'time_at_node',
                                'hour', 'dow', 'day', 'month', 'travel_rate',
                                'dist_btw_observ','edge_length','real_observ',
                                'trip_id', 'vehicle_id', 'route_id','shape_id']
                    time_at_node = time1
                    for edge_idx, edge in enumerate(edge_list):
                        #add a value 'real_pt' to keep track of actual observations vs interpolated observations
                        node1 = edge[0]
                        node1_lon = edge[0][0]
                        node1_lat = edge[0][1]
                        node2 = edge[1]
                        node2_lon = edge[1][0]
                        node2_lat = edge[1][1]
                        edge_length = G.get_edge_data(node1,node2)['dist']
                        if edge_idx == 0:
                            real_pt = True
                        else:
                            real_pt = False
                            #edge_length is in feet, travel_rate_update mph
                            travel_rate_ft_per_sec = trav_rate_update*(5280)*(1/(60*60))
                            time_at_node += pd.Timedelta('{} seconds'.format(edge_length/travel_rate_ft_per_sec))
                            '''print("travel_rate_fps {}, time at node #{} - {} - ".format(travel_rate_ft_per_sec, 
                                                                                     edge_idx,
                                                                                     time_at_node))'''
                        info_tuple = (month_day_trip_veh,
                                        node1_lon, node1_lat, node2_lon,
                                            node2_lat,dist1, dist2,
                                          time1, time2, time_midway,time_at_node,
                                            hour, dow, day, month, trav_rate_update,
                                              trav_dist, edge_length, real_pt,
                                            trip_id, vehicle_id, route_id, shape_id)
                        edge_for_upload.append(info_tuple)
                    edge_df = pd.DataFrame(edge_for_upload, columns=col_list)
                    if veh_row_idx == 0:
                        full_edge_df = edge_df.copy()
                    elif full_edge_df.empty:
                        full_edge_df = edge_df.copy()
                    else:
                        full_edge_df = full_edge_df.append(edge_df)

                    #print("writing to GCP {}-{}".format(time_midway, trip_id))
                except nx.NetworkXNoPath:
                    #print("we have an exception")
                    time1 = vehicle_geo_sorted['veh_time_pct'].iloc[veh_row_idx]
                    day = time1.day
                    month = time1.month
                    output_str = (
                    "{}{}\n{}{}\n{}{}\n{}{}\n\
                    ".format('node1 -', node1,
                            'node2 -', node2,
                            'shape_id -', shape_id,
                            'error_num - ', error_counter))
                    time_str = str(month)+"_"+str(day)
                    file_path = './bad_network_nodes.txt'
                    with open(file_path, "a") as f:
                        f.write("month_day - "+time_str)
                        f.write("\n")
                        f.write(output_str)
                        f.write("\n")
                    error_counter += 1
    return full_edge_df