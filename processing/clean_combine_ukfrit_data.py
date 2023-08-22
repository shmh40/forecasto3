# Combining data from the UK, France and Italy into a singe dataset for training
# Do this process for datasets with and without NO2.

import pandas as pd

pd.set_option('display.max_columns', None)

# Read in data

uk_data = pd.read_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/country/UK/UK_dma8_non_strict_all_data_timeidx_drop_dups.csv')
france_data = pd.read_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/country/France/France_dma8_non_strict_all_data_timeidx_drop_dups.csv')
italy_data = pd.read_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/country/Italy/Italy_dma8_non_strict_all_data_timeidx_drop_dups.csv')

# check total number of rows
print(uk_data.shape[0] + italy_data.shape[0] + france_data.shape[0])

# drop unused columns

uk_data = uk_data.drop(['time_idx_large_temp', 'time_idx_new', 'totprecip'], axis=1)
france_data = france_data.drop(['time_idx_large_temp', 'time_idx_new', 'totprecip'], axis=1)
italy_data = italy_data.drop(['time_idx_large_temp', 'time_idx_new', 'totprecip'], axis=1)

# drop nans in necessary columns
uk_data_dropna_meteo = uk_data.dropna(subset=['o3', 'temp', 'press', 'u', 'v', 'pblheight', 'relhum', 'cloudcover'])
france_data_dropna_meteo = france_data.dropna(subset=['o3', 'temp', 'press', 'u', 'v', 'pblheight', 'relhum', 'cloudcover'])
italy_data_dropna_meteo = italy_data.dropna(subset=['o3', 'temp', 'press', 'u', 'v', 'pblheight', 'relhum', 'cloudcover'])

print(uk_data_dropna_meteo.shape[0] + france_data_dropna_meteo.shape[0] + italy_data_dropna_meteo.shape[0])

# Concatenate the data and ignore the index
combined_data = pd.concat([uk_data_dropna_meteo, france_data_dropna_meteo, italy_data_dropna_meteo], ignore_index=True)

# replace small number missing values in altitude columns with ~means
combined_data['station_etopo_alt'] = combined_data['station_etopo_alt'].fillna(value = 200)
combined_data['station_rel_etopo_alt'] = combined_data['station_rel_etopo_alt'].fillna(value = 50)

# create new time indexes for PyTorch Forecasting

# merged_env_dma8_df_dropna.dtypes
combined_data['datetime'] = pd.to_datetime(combined_data['datetime'], format='%Y-%m-%d')
combined_data['raw_time_idx'] = combined_data['datetime'].apply(lambda x: x.toordinal())
print(max(combined_data['raw_time_idx']))
combined_data['time_idx_large_temp'] = combined_data['raw_time_idx'] + 1000000

# here we are doing the timeidx for each station.

new_data = pd.DataFrame()

for s in list(combined_data['station_name'].unique()):
    data_subset = combined_data[combined_data["station_name"] == s]
    data_subset['time_idx_new'] = data_subset['time_idx_large_temp'] - max(data_subset['raw_time_idx'])
    # print(max(data_subset['time_idx_new']))
    new_data = pd.concat([new_data, data_subset])
    
# set a new column, called time_idx, to act as the time_idx!

new_data["time_idx"] = new_data['time_idx_new']

# save the processed dateframe, ready for training

## save this dataframe

new_data.to_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/merged_euro_clean/uk_france_italy_o3_nans_no2_no_non_strict_drop_dups.csv', index=False)