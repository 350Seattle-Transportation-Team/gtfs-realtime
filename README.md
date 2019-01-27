## Description of scripts/notebooks

All Andrew's S3 data is available at `s3://realtime-buses/datasets/`. To copy both datasets to your computer, use `aws s3 cp s3://realtime-buses/datasets/* .`

Additionally, notebooks 2 and 3 reference GTFS feeds that should be downloaded from https://transitfeeds.com/p/king-county-metro/73 and unzipped to `data/source/gtfs_YYYYMMDD` folders.

- `download_raw_locations.sh`: downloads Ben's raw data for January 2018
- `01_transform_source_data.ipynb`: transforms said data into a pandas DataFrame indexed on the datetime - the output of this is available on S3 in file `positions_201801.h5`
- `02_transform_e_locations.ipynb`: selects northbound E-line vehicles and calculates `closest_stop_id` (used in future analysis) - the output of this is available on S3 in file `e_northbound_locations_2018-01.h5`
- `03_e_segment_analysis.ipynb`: transforms data into a shape that will let us calculate time between two stops for northbound E (denny/aurora and 46th/aurora), then generates histograms for the distribution those commute times

## Environment setup

In a fresh Python 3.6 env:

```
pip install pandas geopandas numpy shapely fiona six pyproj tables matplotlib tqdm
```
