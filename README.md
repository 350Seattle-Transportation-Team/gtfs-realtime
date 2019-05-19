# Background
We have an AWS Lambda Function pulling One Bus Away API responses for King County Metro.
```
agency = '1'  # this is the agency ID for King County Metro
base_url = 'http://pugetsound.onebusaway.org/api/'
endpoints = {'position': 'gtfs_realtime/vehicle-positions-for-agency/{agency}.pb',
           'alert': 'gtfs_realtime/alerts-for-agency/{agency}.pb',
           'update': 'gtfs_realtime/trip-updates-for-agency/{agency}.pb'}
```

We're focusing on the `position` files. If you call [Ping] the One Bus Away API's position endpoint at any point in time, it will return the last recorded vehicle location (lat/lon) for buses travelling along their routes at that time:
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
*The time and distance between bus location records vary. We're pinging the One Bus Away API every minute.*

We can plot one set of vehicle positions along a route (in this case, route 7). In the gtfs, there's a file describing this particular vehicle's (vehicle_id) journey, or trip (trip_id) at a given time of day going a specific direction along route 7.

<img src="/images/trip_1.png" width="300"/>

We can plot another trip here:

<img src="/images/trip_2.png" width="300"/>

If we plot the trips together you can see that the vehicle locations are not consistent one trip to another.

<img src="/images/trip_1_and_2.png" width="300"/>

Let's zoom into the route and see what other information we have:

<img src="/images/position_zoom_in.png" width="500"/>

For each route, the gtfs gives us: 
- route vertex points (all the lat/lon coordinates that make up a route's "shape")
- and an indication whether the route vertex is associated with a bus stop

From the image above, there are some important things to note:
- route vertex points occur at every street intersection
- if there's a bus stop, another route vertex point occurs at the same location along the centerline of the street
- the bus `positions` or `vehicle locations` do not line up directly with either the route vertex points or the bus stops

Our goal is to better understand how buses travel along their routes. One natural question is "How fast is the bus moving along the route?" To get speed, we need change in location (we have that with multiple position records) and distance traveled. To think about getting "distance traveled", it's helpful to look at the picture below:
<img src="/images/between_positions.png" width="800"/>

In the picture, you'll see two vehicle positions along a route. The vehicle was at location 1 first at time, t1. Then the vehicle traveled along the route and arrived at location 2 at time, t2. We can find distance traveled in 2 ways:
1. Naive approach - we can take the straight line distance between location 1 and location 2 (ignoring the actual route). This will work if observations are close together but if two observations that are far apart and the route is non-linear, this naive approach will have a lot of error.
2. Route aware - we can find the nearest route shape vertex to the vehicle location. The gtfs gives us shape distance traveled between each route vertex so we can calculate the distance traveled by taking
shape_distance_traveled<sub>loc2</sub>-shape_distance_traveled<sub>loc1</sub>

Since bus riders are more familiar with distances and timing between stops, it's helpful to contextualize everything around bus stops. There are two ways we are doing this process:
1. Find the nearest route vertex point to each vehicle location. If the nearest route vertex point is a bus stop, keep that row in the dataset. Otherwise, remove the observation (row) from the dataset.
2. Find the nearest route vertex point to each vehicle location. Find the distance and time between route vertex points. Interpolate when the vehicle `would have been` at the bus stop in between route vertex points.

Please see the instructions below to set up your python environment and get started with the code. 

# Description of scripts/notebooks

Note: the data is stored using AWS. If AWS is unfamiliar, scroll down to "Using Amazon Web Services to access Ben’s data" below, for a little tutorial.  

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
#If you don't have Anaconda installed, install it from here. NOTE: if you don't check the box for adding Conda folders to your path, you will likely have trouble later.
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
  ```conda install geopy```\
  
**Using Amazon Web Services to access Ben’s data**
If you don’t have an AWS account, sign up at https://aws.amazon.com/ \
If you don’t have the AWS cli\
install it from https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html \
At this point, if aws commands in the cli return “Unable to locate credentials”, you need to get Amazon Identity and Access Management (IAM) credentials\
And set up the credentials as described here: https://docs.aws.amazon.com/IAM/latest/UserGuide/getting-started_create-admin-group.html, under "creating an aws use..."r.  You will be asked for a user name and password.\
This will generate an email, use the sign-in url in the email, and the user name and password you created in the previous step.\
At this point you are in the console, it has a link to the Identity and Access Management (IAM) console.  Go there.\
And get an access key id and access key as described here: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration \
Double click on users \
Check  box, then double click on a user (e.g. Admin), \
Select the Security Credentials tab, and click Get Access Key\
Now back in your dos command prompt, type \
    Aws configure\
And supply your credentials\
Now just grab the data from https://github.com/350Seattle-Transportation-Team/gtfs-realtime (green download button) 

