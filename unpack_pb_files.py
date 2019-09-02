"""
Script by Ben Malnor for converting raw gtfs .pb files into more user-friendly .zip files.

There are three .pb files per minute, one for each API endpoint:
endpoints = {'position': 'gtfs_realtime/vehicle-positions-for-agency/{agency}.pb',
               'alert': 'gtfs_realtime/alerts-for-agency/{agency}.pb',
               'update': 'gtfs_realtime/trip-updates-for-agency/{agency}.pb'}

^^ all processes in the subsequent notebooks focus on 'position' files but it's worth
knowing that there are two other endpoints.

The general process for using the script is:
- download raw from S3 to local/EC2
- clean / put in .csv
- zip csv
- upload zip to “unpacked” S3
- delete local zip

^^ so the files will sit on your local/EC2 while they’re being processed
 and then they get deleted to make room for the next batch

Note from Ben:
I usually run in on an EC2 that has more cores for the parallel processing.
I think it should work locally as long as you have access to both buckets.
"""

from google.transit import gtfs_realtime_pb2
from google.protobuf.json_format import MessageToJson, MessageToDict
import json
import os
import pandas as pd
import numpy as np
import io
import subprocess
import multiprocessing
import boto3
import datetime
import pytz
from pytz import timezone
import sys
import logging

logging.basicConfig(filename='unpack_pb.log',level=logging.DEBUG)

bucket_name = os.environ["BUS_BUCKET_NAME"]


def make_single_day_position_zip_from_pb(date):
    '''
    INPUT
    ---------
    date - datetime format
    '''
    day = '{:02d}'.format(date.day)
    month = '{:02d}'.format(date.month)
    year = str(date.year)
    #print(year, month, day)
    day_temp_storage_path = "./temp_data_storage_{}_{}".format(month, day)

    aws_base_command = 'aws --quiet s3 sync s3://{}/one_bus_away_raw/{}/{}/'.format(bucket_name,
                                                                year,
                                                                month
                                                                )

    logging.info("starting process for {}/{}/{} data".format(
                                                year,
                                                month,
                                                day))

    print("starting process for {}/{}/{} data".format(
                                                year,
                                                month,
                                                day))
    #remove the temporary folder if it exists
    os.system('rm -r {}'.format(day_temp_storage_path))

    #create a temporary folder to store a day's worth of downloads
    os.system('mkdir {}'.format(day_temp_storage_path))

    os.system(aws_base_command+"{} {} --quiet".format(day,
                                        day_temp_storage_path))

    print("finished downloading {}/{}/{} files".format(
                                                year,
                                                month,
                                                day))

    full_position_df = position_files_to_df(day_temp_storage_path)

    csv_file_name = day_temp_storage_path+"/{}_{}_{}_positions.csv".format(year,
                                                                        month,
                                                                        day)
    zip_file_name = day_temp_storage_path+"/{}_{}_{}_positions.zip".format(year,
                                                                        month,
                                                                        day)
    #print("making zip")
    make_zip_csv(full_position_df,
                                csv_file_name,
                                zip_file_name)
    put_zip_s3(zip_file_name,year,month)

    #print("zip made")

    print("finished process for {}/{}/{} data".format(
                                                year,
                                                month,
                                                day))

    logging.info("finished process for {}/{}/{} data".format(
                                                year,
                                                month,
                                                day))

    #remove the temporary folder after for loop is finished
    os.system('rm -r {}'.format(day_temp_storage_path))

def make_single_day_update_zip_from_pb(date):
    '''
    INPUT
    ---------
    date - datetime format
    '''
    day = '{:02d}'.format(date.day)
    month = '{:02d}'.format(date.month)
    year = str(date.year)
    #print(year, month, day)
    day_temp_storage_path = "./temp_data_storage_{}_{}".format(month, day)

    aws_base_command = 'aws --quiet s3 sync s3://{}/one_bus_away_raw/{}/{}/'.format(bucket_name,
                                                                year,
                                                                month
                                                                )

    print("starting process for {}/{}/{} data".format(
                                                year,
                                                month,
                                                day))
    #remove the temporary folder if it exists
    os.system('rm -r {}'.format(day_temp_storage_path))

    #create a temporary folder to store a day's worth of downloads
    os.system('mkdir {}'.format(day_temp_storage_path))

    os.system(aws_base_command+"{} {} --quiet".format(day,
                                        day_temp_storage_path))

    print("finished downloading {}/{}/{} files".format(
                                                year,
                                                month,
                                                day))

    full_update_df = update_files_to_df(day_temp_storage_path)

    csv_file_name = day_temp_storage_path+"/{}_{}_{}_updates.csv".format(year,
                                                                        month,
                                                                        day)
    zip_file_name = day_temp_storage_path+"/{}_{}_{}_updates.zip".format(year,
                                                                        month,
                                                                        day)
    #print("making zip")
    make_zip_csv(full_update_df,
                                csv_file_name,
                                zip_file_name)
    put_zip_s3(zip_file_name,year,month)

    #print("zip made")

    print("finished process for {}/{}/{} data".format(
                                                year,
                                                month,
                                                day))

    #remove the temporary folder after for loop is finished
    os.system('rm -r {}'.format(day_temp_storage_path))

def make_date_range(start_date, end_date):
    start_datetime = datetime.datetime.strptime(start_date, '%m/%d/%Y')

    end_datetime = datetime.datetime.strptime(end_date, '%m/%d/%Y')

    step = datetime.timedelta(days=1)

    date_list = []
    while start_datetime <= end_datetime:
        temp_date = start_datetime.date()
        #print(temp_date)
        start_datetime += step
        date_list.append(temp_date)
    return date_list

def position_files_to_df(day_temp_storage_path):
    folder_list = [f for f in os.listdir(day_temp_storage_path) if not f.startswith(".")]
    for i, folder in enumerate(folder_list):
        folder_path = os.path.join(day_temp_storage_path,folder)
        #print("working on #{} in folder_path {}".format(i,folder_path))
        all_subfiles_list = [f for f in os.listdir(folder_path)]
        pb_file_list = list(filter(lambda x: 'position' in x,
                                        all_subfiles_list))
        position_list, bad_update_header_list = make_vehicle_list(pb_file_list, folder_path)

        partial_position_df = pd.DataFrame(position_list)
        if i == 0:
            full_position_df = partial_position_df.copy()
        else:
            full_position_df = full_position_df.append(partial_position_df)
        #print("finished #{} in folder_path {}".format(i,folder_path))
    full_position_df = make_clean_position_pandas(full_position_df)
    return full_position_df

def make_vehicle_list(pb_file_list, folder_path):
    vehicle_list = []
    bad_vehicle_header_list = []
    dirname = folder_path
    for pb_file in pb_file_list:
        pb_file_path = os.path.join(dirname, pb_file)
        with open(pb_file_path, 'rb') as f:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(f.read())
            json_obj = MessageToJson(feed)
        json_parsed = json.loads(json_obj)
        if 'entity' in json_parsed.keys():
            for vehicles_idx in range(len(json_parsed['entity'])):
                vehicle_dict = {}
                j_out = json_parsed['entity'][vehicles_idx]
                if 'vehicle' in j_out.keys():
                    if 'position' in j_out['vehicle'] and 'trip' in j_out['vehicle']:
                        vehicle_dict['vehicle_id'] = j_out['vehicle']['vehicle']['id']
                        vehicle_dict['timestamp'] = j_out['vehicle']['timestamp']
                        vehicle_dict['trip_id'] = j_out['vehicle']['trip']['tripId']
                        vehicle_dict['route_id'] = j_out['vehicle']['trip']['routeId']
                        vehicle_dict['vehicle_lat'] = j_out['vehicle']['position']['latitude']
                        vehicle_dict['vehicle_long'] = j_out['vehicle']['position']['longitude']
                        #trip_id = j_out['vehicle']['trip']['tripId']
                        #route_id = j_out['vehicle']['trip']['routeId']
                        #vehicle_dict['shape_id'] = get_shape_id_from_triproute(trip_id, route_id, schedule_df)
                        vehicle_list.append(vehicle_dict)
                    else:
                        bad_vehicle_header_list.append(json_parsed['header'])
                else:
                    bad_vehicle_header_list.append(json_parsed['header'])
        else:
            bad_vehicle_header_list.append(json_parsed['header'])
    return vehicle_list, bad_vehicle_header_list

def make_clean_position_pandas(position_df):
    position_df['route_id'] = position_df['route_id'].astype(int)
    position_df['trip_id'] = position_df['trip_id'].astype(int)
    position_df['vehicle_id'] = position_df['vehicle_id'].astype(int)
    position_df = position_df.drop_duplicates(['trip_id','vehicle_id','route_id','timestamp'],keep='first')
    position_df['time_utc'] = pd.to_datetime(position_df['timestamp'], unit='s')
    position_df['time_pct'] = position_df.apply(update_timestamp, axis=1)
    return position_df

def make_update_list(pb_file_list, folder_path):
    update_list = []
    bad_update_header_list = []
    dirname = folder_path
    for pb_file in pb_file_list:
        pb_file_path = os.path.join(dirname, pb_file)
        with open(pb_file_path, 'rb') as f:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(f.read())
            dict_obj = MessageToDict(feed)
        if 'entity' in dict_obj.keys():
            for update_idx in range(len(dict_obj['entity'])):
                update_dict = {}
                j_in = json.dumps(dict_obj['entity'][update_idx])
                j_out = json.loads(j_in)
                if 'tripUpdate' in j_out.keys():
                    update_dict['delay'] = j_out['tripUpdate']['delay']
                    update_dict['stop_update_departure'] = j_out['tripUpdate']['stopTimeUpdate'][0]['departure']['time']
                    update_dict['stop_id'] = j_out['tripUpdate']['stopTimeUpdate'][0]['stopId']
                    update_dict['timestamp'] = j_out['tripUpdate']['timestamp']
                    update_dict['route_id'] = j_out['tripUpdate']['trip']['routeId']
                    update_dict['trip_id'] = j_out['tripUpdate']['trip']['tripId']
                    update_dict['vehicle_id'] = j_out['tripUpdate']['vehicle']['id']
                    update_list.append(update_dict)
                else:
                    bad_update_header_list.append(dict_obj['header'])
        else:
            bad_update_header_list.append(dict_obj['header'])
    return update_list, bad_update_header_list

def make_clean_update_pandas(update_df):

    update_df['route_id'] = update_df['route_id'].astype(int)
    update_df['stop_id'] = update_df['stop_id'].astype(int)
    update_df['trip_id'] = update_df['trip_id'].astype(int)
    update_df['vehicle_id'] = update_df['vehicle_id'].astype(int)
    update_df = update_df.drop_duplicates(['trip_id','vehicle_id','timestamp','stop_id'],keep='first')
    update_df = update_df.sort_values('delay',
                          ascending=False).groupby(
                                                    ['trip_id',
                                                     'vehicle_id',
                                                     'stop_id']
                                                    ).tail(1) #grab only the top value (largest delay)
    update_df.reset_index(drop=True,inplace=True)
    update_df['time_utc'] = pd.to_datetime(update_df['timestamp'], unit='s')
    update_df['time_pct'] = update_df.apply(update_timestamp, axis=1)
    return update_df

def update_timestamp(row):
    time_utc = row['time_utc'].tz_localize(timezone('UTC'))
    time_pct = time_utc.astimezone(timezone('US/Pacific'))
    return time_pct

def update_files_to_df(day_temp_storage_path):
    folder_list = [f for f in os.listdir(day_temp_storage_path) if not f.startswith(".")]
    for i, folder in enumerate(folder_list):
        folder_path = os.path.join(day_temp_storage_path,folder)
        #print("working on #{} in folder_path {}".format(i,folder_path))
        all_subfiles_list = [f for f in os.listdir(folder_path)]
        pb_file_list = list(filter(lambda x: 'update' in x,
                                        all_subfiles_list))
        update_list, bad_update_header_list = make_update_list(pb_file_list, folder_path)

        partial_update_df = pd.DataFrame(update_list)
        if i == 0:
            full_update_df = partial_update_df.copy()
        else:
            full_update_df = full_update_df.append(partial_update_df)
        #print("finished #{} in folder_path {}".format(i,folder_path))
    full_update_df = make_clean_update_pandas(full_update_df)
    return full_update_df

def make_zip_csv(full_update_df, csv_file_name, zip_file_name):
    full_update_df.to_csv(csv_file_name, index=False)
    os.system('zip {} {}'.format(zip_file_name, csv_file_name))

def put_zip_s3(zip_file_name,year,month):
    '''
    '''
    bucket_suffix = "unpacked/{}/{}/".format(year,month)
    aws_base_command = 'aws s3 cp {} s3://{}/{}'.format(zip_file_name, bucket_name,
                                                                    bucket_suffix
                                                                    )
    os.system(aws_base_command)

if __name__ == "__main__":
    '''
    usage - python unpack_pb_files.py <update or position> <start_date> <end_date>
    e.g. python unpack_pb_files.py "position" "12/01/2018" "12/17/2018"
    the above will "unpack" position endpoint .pb files from 12/01/2018 - 12/17/2018
    and put them in s3_bucket_base/unpacked/year/month/ folder
    '''

    file_option = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]

    date_list = make_date_range(start_date, end_date)

    n_pools = multiprocessing.cpu_count() - 2

    pool = multiprocessing.Pool(4)

    if file_option == 'update':
        pool.map(make_single_day_update_zip_from_pb, date_list)
    elif file_option == 'position':
        pool.map(make_single_day_position_zip_from_pb, date_list)

    else:
        print("incorrect file_type option - must be update or position")
