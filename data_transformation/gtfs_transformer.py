import pandas as pd
import numpy as np
import os

class GTFS:
    """
    Class to store GTFS tables.
    """

    def __init__(self, directory, extension='.txt', start_date=None, end_date=None):
        """
        Initialize from files stored in a directory.
        """
        self.directory = directory
        with os.scandir(directory) as dir_entries:
            for dir_entry in dir_entries:
                if dir_entry.name.endswith(extension):
                    table_name = dir_entry.name[:-len(extension)]
                    table = pd.read_csv(dir_entry.path)
                    setattr(self, table_name, table)

        # self.start_date = pd.to_datetime(start_date) if start_date is not None else None
        if start_date is not None:
            self.start_date = start_date
        if end_date is not None:
            self.end_date = end_date
