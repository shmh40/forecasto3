# Final cleaning and combination of country datasets.
# We do this for datasets both with and without NO2.

# imports
import pandas as pd

pd.set_option('display.max_columns', None)

# Read in data from a particular country. Can do for all countries.

country = 'Norway'

country_data = pd.read_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/country/'+country+'/'+country+'_dma8_non_strict_all_data_timeidx_drop_dups.csv')


# Check number of duplicate rows.

duplicateRows = country_data[country_data.duplicated()]
print('Number of duplicate rows:', len(duplicateRows))

# drop the columns that we are not interested in.

country_data = country_data.drop(['time_idx_large_temp', 'time_idx_new', 'totprecip'], axis=1)
# drop rows with nans in the meteorological columns and in o3 column.
country_data_dropna_meteo = country_data.dropna(subset=['o3', 'temp', 'press', 'u', 'v', 'pblheight', 'relhum', 'cloudcover'])

# fill missing altitude values with mean altitudes
country_data_dropna_meteo['station_etopo_alt'] = country_data_dropna_meteo['station_etopo_alt'].fillna(value = country_data_dropna_meteo.station_etopo_alt.mean())
country_data_dropna_meteo['station_rel_etopo_alt'] = country_data_dropna_meteo['station_rel_etopo_alt'].fillna(value = country_data_dropna_meteo.station_rel_etopo_alt.mean())
country_data_dropna_meteo['alt'] = country_data_dropna_meteo['alt'].fillna(value = country_data_dropna_meteo.alt.mean())

# save this dataframe

country_data_dropna_meteo.to_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/country/'+country+'/'+country+'_dma8_non_strict_all_data_timeidx_drop_dups_drop_nas_in_o3_and_met_columns.csv', index=False)

# Now we individual files that can be used for training and testing.
