import pandas as pd
import numpy as np
import os
import string

##################################################################################
#   Collection of aggregation functions to use with pandas .agg() function,
#   which does not play well with things that aren't hashable,
#   so we need to use tuples.
##################################################################################

def list_all(objects: pd.Series) -> tuple:
    """Aggregates a collection of objects by putting them in a tuple."""
    return tuple(objects)

def list_all_sorted(comparable_objects: pd.Series) -> tuple:
    """Aggregates a collection of comparable objects by sorting them and putting them in a tuple."""
    return tuple(sorted(comparable_objects))

def count_unique(hashable_objects: pd.Series) -> int:
    """Aggregates hashable objects by counting the unique ones."""
    return hashable_objects.nunique() #len(set(hashable_objects))

def list_unique(hashable_objects: pd.Series) -> tuple:
    """Aggregates hashable objects by putting the unique ones in a tuple."""
    return tuple(hashable_objects.unique())

def list_unique_sorted(hashable_comparable_objects: pd.Series) -> tuple:
    """Aggregates hashable objects by sorting the unique ones and putting them in a tuple.
    """
    return tuple(sorted(hashable_comparable_objects.unique()))

class StaticGTFS:
    """
    Class to store static GTFS tables posted on a specific date.
    """

    def __init__(self, directory, extension='.txt', post_date='infer', timezone='US/Pacific'):
        """
        Initialize tables from files stored in a directory.
        """
        self.directory = directory
        self.table_names = []
        with os.scandir(directory) as dir_entries:
            for dir_entry in dir_entries:
                if dir_entry.name.endswith(extension):
                    table_name = dir_entry.name[:-len(extension)]
                    self.table_names.append(table_name)
                    table = pd.read_csv(dir_entry.path)
                    setattr(self, table_name, table)

        # Sort the table names in place
        self.table_names.sort()

        if post_date=='infer': #Try to infer date from directory
            # Remove non-alphanumeric characters from directory to make pattern-matching easier
            dir_alnum = ''.join(ch for ch in directory if ch.isalnum())
            self.post_date = pd.to_datetime(dir_alnum, format='%Y%m%d', exact=False)
        else:
            self.post_date = pd.to_datetime(post_date)

        # Localize the start date to the given timezone, and convert it to UTC
        self.post_date = self.post_date.tz_localize(timezone).tz_convert('UTC')

    def route_ids(self, *route_short_names):
        """
        Return a Series indexed by route short names with values equal to the route id,
        which can be used like a dictionary to index DataFrames indexed by route id's
        by route names instead.
        If any route short names are passed as arguments, only those routes will be
        included in the result.
        """
        route_data = self.routes[['route_short_name', 'route_id']]
        if len(route_short_names) > 0:
            route_short_names = [str(name) for name in route_short_names]
            route_data = route_data.loc[route_data.route_short_name.isin(route_short_names),:]
        return route_data.set_index('route_short_name')['route_id']

    def route_ids_from_names(self, *route_short_names):
        if len(route_short_names) == 0: #Get all the id's if none were passed
            route_short_names = self.routes.route_short_name
        else:
            route_short_names = [str(name) for name in route_short_names]

        return self.routes.loc[self.routes.route_short_name.isin(route_short_names),
                ['route_short_name', 'route_id']].set_index('route_short_name')

    def routes_by_name(self, *route_short_names):
        """
        Extracts the most saliet data from the routes table (name, id, description),
        and indexes the table by route_short_name.
        """
        route_data = self.routes[['route_short_name', 'route_desc', 'agency_id', 'route_id']]
        if len(route_short_names) > 0:
            route_short_names = [str(name) for name in route_short_names]
            route_data = route_data.loc[route_data.route_short_name.isin(route_short_names),:]
        return route_data.set_index('route_short_name')

    # def trip_headsign_and_direction_id_for_routes(route_short_names):
    #     return trips.loc[trips.route_short_name.isin(route_short_names), ]

    def trips_by_route_and_direction(self, *route_short_names):
        """
        Returns a dataframe aggregated and indexed by route names and direction id's,
        containing the route description, the list of trip headsigns for each direction,
        the count of shapes for each direction, and the count of trips for each direction.
        """
        route_data = self.routes[['route_id', 'route_short_name', 'route_desc']]
        if len(route_short_names) > 0:
            route_short_names = [str(name) for name in route_short_names]
            route_data = route_data.loc[route_data.route_short_name.isin(route_short_names),:]

        # print(route_data)
        return self.trips.merge(
                route_data, on='route_id'
            ).groupby(
                by=['route_short_name', 'direction_id']
            ).agg(
                {'route_desc': 'max', # There should be only one route description per route
                'trip_headsign': list_unique_sorted, #lambda x: x.unique(),
                'shape_id': count_unique,
                'trip_id': 'count',
                'block_id': count_unique,
                'trip_short_name': list_unique_sorted,
                'peak_flag': list_unique_sorted,
                'fare_id': list_unique_sorted
                }
            ).rename(
                columns={
                'shape_id': 'shape_count',
                'trip_id': 'trip_count',
                'block_id': 'block_count',
                }
            )

    def trips_by_route_and_shape(self, *route_short_names):
        """
        Returns a dataframe aggregated and indexed by route names and direction id's,
        containing the route description, the list of trip headsigns for each direction,
        the count of shapes for each direction, and the count of trips for each direction.
        """
        route_data = self.routes[['route_id', 'route_short_name']]
        if len(route_short_names) > 0:
            route_short_names = [str(name) for name in route_short_names]
            route_data = route_data.loc[route_data.route_short_name.isin(route_short_names),:]
        return self.trips.merge(
                route_data, on='route_id'
            ).groupby(
                by=['route_short_name', 'shape_id']
            ).agg(
                {'direction_id': 'max', #There should be only one direction per shape
                'trip_headsign': list_unique, #lambda x: tuple(x.unique()),
                'trip_id': 'count',
                'block_id': count_unique, #lambda x: x.nunique(),
                'trip_short_name': list_unique_sorted, #lambda x: tuple(sorted(x.unique())),
                'peak_flag': list_unique_sorted, #lambda x: tuple(sorted(x.unique())),
                'fare_id': list_unique_sorted, #lambda x: tuple(sorted(x.unique())),
                }
            ).rename(
                columns={
                'trip_id': 'trip_count',
                'block_id': 'block_count',
                }
            )
    def trips_for_route_and_direction(self, route_short_name, direction_id):
        """Return trip data for all the trips for the given route and direction."""
        route_data = self.routes.loc[self.routes.route_short_name == str(route_short_name),
            ['route_short_name', 'route_id']]
        return route_data.merge(
            self.trips.loc[self.trips.direction_id == direction_id],
            on = 'route_id'
        )

    def trip_ids_for_route_and_direction(self, route_short_name, direction_id):
        """Return trip ids for all the trips for the given route and direction."""
        # route_data = self.routes.loc[
        #     self.routes.route_short_name == str(route_short_name), 'route_id']
        # return route_data.merge(
        #     self.trips.loc[self.trips.direction_id == direction_id, ['route_id','trip_id']],
        #     on = 'route_id'
        # ).trip_id
        route_id = self.route_ids(route_short_name).values[0]
        return self.trips.loc[(self.trips.route_id == route_id)
            & (self.trips.direction_id == direction_id), 'trip_id'
            ]


    def stops_by_id(self, stop_ids=None):
        """Returns the salient data from the stops table, with index set to `stop_id`.
        Filters by the given stop ids if any are passed.
        """
        stops_data = self.stops[
                ['stop_id', 'stop_name', 'zone_id', 'stop_lat', 'stop_lon']
            ].set_index('stop_id')
        if stop_ids is not None:
            stops_data = stops_data.loc[stop_ids]
        return stops_data

class StaticGTFSHistory:
    """
    Class to store a collection of static GTFS tables posted historically.
    """

    def __init__(self, gtfs_list):
        """
        Creates a GTFS history from a list of GTFS objects.
        """
        #Save a dictionary mapping dates to gtfs objects
        self.date_to_gtfs = {gtfs.post_date: gtfs for gtfs in gtfs_list}

        #These are redundant but might be convenient
        self.dates = sorted(gtfs.post_date for gtfs in gtfs_list)
        self.gtfs_list = gtfs_list
