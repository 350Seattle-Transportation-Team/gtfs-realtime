import pandas as pd
import numpy as np
import os
import string

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

    def route_ids_from_names(self, *route_short_names):
        if len(route_short_names) == 0: #Get all the id's if none were passed
            route_short_names = self.routes.route_short_name
        else:
            route_short_names = [str(name) for name in route_short_names]

        return self.routes.loc[self.routes.route_short_name.isin(route_short_names),
                ['route_short_name', 'route_id']].set_index('route_short_name')

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
