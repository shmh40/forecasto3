# TOAR: merging the downloaded environmental and chemical data.

# This script merges the environmental and chemical data downloaded from 
# the v1_scrape_dma8 and v1_scrape_env scripts, putting them into a single dataframe, 
# one with nans and one without nans. 
# The dataframes include datetimes and station attributes. 

# Our imports for this script

import pandas as pd
import numpy as np

from functools import reduce

# Need to swap the country in here, or create a list and loop through.
# Need to have dma8 or dma8_non_strict for the chemical data, 
# defined by the string sampling here.

country = 'Switzerland'
sampling = 'dma8_non_strict' # or 'dma8'


# read in both the dropnaed and total data for both env and dma8 

country_dma8_df = pd.read_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/country/'+country+'/dma8/'+sampling+'_data.csv')
country_dma8_df_dropna = pd.read_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/country/'+country+'/dma8/'+sampling+'_dropna_data.csv')

country_env_df = pd.read_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/country/'+country+'/env/env_data.csv')
country_env_df_dropna = pd.read_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/country/'+country+'/env/env_dropna_data.csv')

print('Data loading complete')

#define list of DataFrames
dfs = [country_dma8_df, country_env_df]

#merge all DataFrames into one
merged_env_dma8_df = reduce(lambda  left,right: pd.merge(left,right,on=['datetime', 'station_name', 'country', 'lat', 'lon', 'alt', 'station_etopo_alt', 
                                                                        'station_rel_etopo_alt', 'station_type', 'landcover', 'toar_category', 
                                                                        'pop_density', 'max_5km_pop_density', 'max_25km_pop_density', 
                                                                        'nightlight_1km', 'nightlight_max_25km', 'nox_emi', 'omi_nox'],
                                            how='outer'), dfs)

print('Merging complete')

# here replacing -1.0s and -999.0s with nans...allows easier dropping of nan values

merged_env_dma8_df = merged_env_dma8_df.replace(-1.0, np.nan)
merged_env_dma8_df = merged_env_dma8_df.replace(-999.0, np.nan)


### Check the length of the data having dropped nans.
merged_env_dma8_df_dropna = merged_env_dma8_df.dropna()
print('Check on length of dropnaed data:', len(merged_env_dma8_df_dropna))

# Here we are setting up the time_idx aspect of the work.

# merged_env_dma8_df_dropna.dtypes
merged_env_dma8_df['datetime'] = pd.to_datetime(merged_env_dma8_df['datetime'], format='%Y-%m-%d')
merged_env_dma8_df['raw_time_idx'] = merged_env_dma8_df['datetime'].apply(lambda x: x.toordinal())
print(max(merged_env_dma8_df['raw_time_idx']))
merged_env_dma8_df['time_idx_large_temp'] = merged_env_dma8_df['raw_time_idx'] + 1000000

# here we are doing the timeidx for each station, as required for PyTorch Forecasting.

new_data = pd.DataFrame()

for s in list(merged_env_dma8_df['station_name'].unique()):
    data_subset = merged_env_dma8_df[merged_env_dma8_df["station_name"] == s]
    data_subset['time_idx_new'] = data_subset['time_idx_large_temp'] - max(data_subset['raw_time_idx'])
    # print(max(data_subset['time_idx_new']))
    new_data = pd.concat([new_data, data_subset])
    
# set a new column, called time_idx, to act as the time_idx for PyTorch Forecasting.

new_data["time_idx"] = new_data['time_idx_new']

# sort the dataframe by station_name and by time_idx, and temp and no2, and drop duplicates if any.

new_data_sorted = new_data.sort_values(['station_name', 'time_idx', 'temp', 'no2'], ignore_index=True) 
new_data_sorted_drop_dups = new_data_sorted.drop_duplicates(subset=['datetime', 'station_name'], keep='first')

# Check no duplicates.
assert new_data_sorted_drop_dups['station_name'].nunique() == new_data_sorted_drop_dups['time_idx'].value_counts().max()

# could create the directory here, but this has already been done in previous scripts.

new_data_sorted_drop_dups.to_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/country/'+country+'/'+country+'_'+sampling+'_all_data_timeidx_drop_dups.csv', index=False)

print('Data saved to csv')

# this dataframe (which has nans in NO and NO2), can then be dealt with before ingesting into algorithm, depending on need for NO or NO2 for example.