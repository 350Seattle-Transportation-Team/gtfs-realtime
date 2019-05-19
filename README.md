# Background
We have an AWS Lambda Function getting One Bus Away API responses for King County Metro.
```
agency = '1'  # this is the agency ID for King County Metro
base_url = 'http://pugetsound.onebusaway.org/api/'
endpoints = {'position': 'gtfs_realtime/vehicle-positions-for-agency/{agency}.pb',
           'alert': 'gtfs_realtime/alerts-for-agency/{agency}.pb',
           'update': 'gtfs_realtime/trip-updates-for-agency/{agency}.pb'}
```

We're focusing on the `position` files. If you call the One Bus Away API's position endpoint at any point in time, it will return vehicle locations (lat/lon) for buses travelling along their routes at that time:
```
entity {
  id: "2"
  vehicle {
    trip {
      trip_id: "34746616"
      route_id: "100136"
    }
    position {
      latitude: 47.657772
      longitude: -122.14334
    }
    timestamp: 1528154042
    vehicle {
      id: "7222"
    }
  }
}
```

We can plot one set of vehicle positions along a route (in this case, route 7). In the gtfs, there's a file describing this particular vehicle's (vehicle_id) journey, or trip (trip_id) at a given time of day going a specific direction along route 7.

<img src="/images/trip_1.png" width="300"/>

We can plot another trip here:

<img src="/images/trip_2.png" width="300"/>

If we plot the trips together you can see that the vehicle locations are not consistent one trip to another.

<img src="/images/trip_1_and_2.png" width="300"/>

# Description of scripts/notebooks

All Andrew's S3 data is available at `s3://realtime-buses/datasets/`. To copy both datasets to your computer, use `aws s3 cp s3://realtime-buses/datasets/ . --recursive`

Additionally, notebooks 2 and 3 reference GTFS feeds that should be downloaded from https://transitfeeds.com/p/king-county-metro/73 and unzipped to `data/source/gtfs_YYYYMMDD` folders.

- `download_raw_locations.sh`: downloads Ben's raw data for January 2018
- `01_transform_source_data.ipynb`: transforms said data into a pandas DataFrame indexed on the datetime - the output of this is available on S3 in file `positions_201801.h5`
- `02_transform_e_locations.ipynb`: selects northbound E-line vehicles and calculates `closest_stop_id` (used in future analysis) - the output of this is available on S3 in file `e_northbound_locations_2018-01.h5`
- `03_e_segment_analysis.ipynb`: transforms data into a shape that will let us calculate time between two stops for northbound E (denny/aurora and 46th/aurora), then generates histograms for the distribution those commute times

# Environment setup

In a fresh Python 3.6 env:

```
pip install pandas geopandas numpy shapely fiona six pyproj tables matplotlib tqdm geopy
```

In more detail, assuming you have successfully installed [Anaconda](https://www.anaconda.com/download/) on your system:

## Mac OS X

On Mac, you can set up a Python 3.6 environment using `conda`, but you need to install the above packages with `pip`.

```bash
#Create a new conda environment named `realtime-buses` with Python version 3.6 (3.7 does not work) and the ipython kernel
conda create --name realtime-buses python=3.6 ipykernel
#Activate the new environment
source activate realtime-buses
#Use pip to nstall modules needed for geopandas
pip install geopandas numpy pandas shapely fiona six pyproj tables matplotlib tqdm geopy
#Install the kernel for the new environment (for the current user) so Jupyter will detect it
ipython kernel install --user --name realtime-buses --display-name "Python 3.6 (realtime-buses)"
#Or... not sure what the difference is:
#python -m ipykernel install --user --name realtime-buses --display-name "Python 3.6 (realtime-buses)"

#Launch Jupyter in your browser. The directory from which you type the command will be
#the top level directory in your Jupyter session, and you can navigate down from there
#if needed. Click on a .ipynb file to open it, or click the 'New' button to create
#a new notebook. You may have to explicitly select the "Python 3.6 (realtime-buses)" kernel.
jupyter notebook
```

## Windows

On Windows, it should work to install everything with `conda`. Instead of `tables`, install `pytables` (this is needed to work with `.h5` files).

```shell
#Create a new conda environment named `realtime-buses` with Python version 3.6 (3.7 does not work) and the ipython kernel
conda create --name realtime-buses python=3.6 ipykernel
#Activate the new environment
conda activate realtime-buses
#On Windows, instead of pip:
conda install geopandas numpy pandas shapely fiona six pyproj pytables matplotlib tqdm geopy
#Install the kernel for the new environment (for the current user) so Jupyter will detect it
ipython kernel install --user --name realtime-buses --display-name "Python 3.6 (realtime-buses)"
#Launch Jupyter in your browser. The directory from which you type the command will be
#the top level directory in your Jupyter session, and you can navigate down from there
#if needed. Click on a .ipynb file to open it, or click the 'New' button to create
#a new notebook. You may have to explicitly select the "Python 3.6 (realtime-buses)" kernel.
jupyter notebook
```
Windows trouble shooting notes:
1. If you followed the conda install's bad advice not to add conda to your path, you may have to add a bunch of stuff to your path (e.g. if you get an HTTP error involving ssl not found).  I (Alice) found that I had to run setx PATH "%path%";c:\Users\Alice\Anaconda3\;c:\Users\Alice\Anaconda3\scripts;c:\Users\Alice\Anaconda3\condabin.
2. If you get a PackagesNotFound error for geopy:\
  ```conda config --append channels conda-forge```\
  ```conda install geopy```

