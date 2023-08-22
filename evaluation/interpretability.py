# Script to examine uncertainty and interpretability in TFT.

# imports

# basic imports
import warnings
from typing import Any, Dict, Union
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

# pytorch and pytorch_forecasting imports
from pytorch_forecasting import (Baseline, TemporalFusionTransformer,
                                 TimeSeriesDataSet)
from pytorch_forecasting.data import TimeSeriesDataSet, TorchNormalizer
from pytorch_forecasting.metrics import MASE, Metric
from pytorch_forecasting.utils import to_list

from scipy import stats

## scikit-learn imports
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import RobustScaler


warnings.filterwarnings("ignore")  # avoid printing out absolute paths


pd.set_option('max_columns', None)

### Reading in the data scraped from TOAR here.
data = pd.read_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/merged_euro_clean/uk_france_italy_o3_nans_no2_no_non_strict_drop_dups.csv')

# set 2 negative ozone values we have to a small positive number.
data.o3[data.o3 < 0] = 0.01
# log transform pblheight.
data['pblheight'] = np.log(1 + data['pblheight'])

print('Data loaded')
print('No. of unique stations =', data['station_name'].nunique())

# GLOBALS for the models

num_workers = 4
batch_size = 128  # set this between 32 to 128

# choose forecast length - common for studies of this type
max_prediction_length = 4
# choose look-back length
max_encoder_length = 21

# define some functions that we use to load the datasets, and to evaluate performance.

#  we have robust scaling for all variables, including ozone, 
# and we log transform ozone and pblheight
def load_prepare_ukfrit_data_for_year_robust_features(data, last_day_of_training):
    
    '''
    This function automates the splitting of data by years, into train, validation and test sets.
    last_day_of_training is a string representing a date which is the last day of the training set, of the form 'YYYY-MM-DD' e.g. '2006-12-31' for the last day of 2006.
    The function returns a the training TimeSeriesDataSet, and the train, val and test DataLoaders
    
    '''
    
    # prepare the data here to have different train/val/test splits...
    training_cutoff_end_of_year = pd.to_datetime(last_day_of_training).toordinal()
    val_cutoff_end_of_year = (pd.to_datetime(last_day_of_training) + pd.offsets.DateOffset(years=1)).toordinal()
    test_cutoff_end_of_year = (pd.to_datetime(last_day_of_training) + pd.offsets.DateOffset(years=2)).toordinal()
    
    print(training_cutoff_end_of_year)
    print(val_cutoff_end_of_year)
    print(test_cutoff_end_of_year)

    # prepare the data here to have different train/val/test splits...
    train_data_raw_pre = data[lambda x: x.raw_time_idx < training_cutoff_end_of_year]
    train_data_raw_post = data[lambda x: x.raw_time_idx > test_cutoff_end_of_year]
    train_data_raw = pd.concat([train_data_raw_pre, train_data_raw_post])

    val_data_raw_pre = data[lambda x: x.raw_time_idx <= (val_cutoff_end_of_year)]
    val_data_raw = val_data_raw_pre[lambda x: x.raw_time_idx > (training_cutoff_end_of_year)]

    test_data_raw_pre = data[lambda x: x.raw_time_idx <= (test_cutoff_end_of_year)]
    test_data_raw = test_data_raw_pre[lambda x: x.raw_time_idx > (val_cutoff_end_of_year)]
    
    print('Train min index:', train_data_raw['raw_time_idx'].min())
    print('Train max index:', train_data_raw['raw_time_idx'].max())
    print('Val min index:', val_data_raw['raw_time_idx'].min())
    print('Val max index:', val_data_raw['raw_time_idx'].max())
    print('Test min index:', test_data_raw['raw_time_idx'].min())
    print('Test max index:', test_data_raw['raw_time_idx'].max())

    print('Train data percentage:', train_data_raw.shape[0]/(train_data_raw.shape[0]+val_data_raw.shape[0]+test_data_raw.shape[0]))
    print('Val data percentage:', val_data_raw.shape[0]/(train_data_raw.shape[0]+val_data_raw.shape[0]+test_data_raw.shape[0]))
    print('Test data percentage:', test_data_raw.shape[0]/(train_data_raw.shape[0]+val_data_raw.shape[0]+test_data_raw.shape[0]))
    
    # Now I need to create the appropriate TimeSeriesDataSets.

    # create the training time-series dataset
    training = TimeSeriesDataSet(
        train_data_raw,
        time_idx="raw_time_idx", # swap between raw_time_idx and time_idx depending if we are doing absolute yearly split or per station yearly -> do below time_varying_known_reals
        target="o3",
        group_ids=["station_name"],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["station_type"],
        static_reals=["landcover", "pop_density", "nox_emi", "alt", 
                    "station_etopo_alt", "station_rel_etopo_alt", "omi_nox", "max_5km_pop_density",
                    "max_25km_pop_density", "nightlight_1km", "nightlight_max_25km", "toar_category"],
        time_varying_known_categoricals=[],
        variable_groups={},  # group of categorical variables can be treated as one variable
        #time_varying_known_reals=["time_idx", "cloudcover", "relhum", "press", "temp", "v", "u", "pblheight"] , 
        time_varying_known_reals=["raw_time_idx", "cloudcover", "relhum", "press", "temp", "v", "u", "pblheight"], 
        time_varying_unknown_categoricals=[], 
        time_varying_unknown_reals=["o3",],
        target_normalizer=TorchNormalizer(method='robust', center=False, transformation="log1p"),  # use softplus and normalize across the whole train set  
        scalers={"cloudcover": RobustScaler(), "temp": RobustScaler(), "press": RobustScaler(), "relhum": RobustScaler(),
            "pblheight": RobustScaler(), "u": RobustScaler(), "v": RobustScaler()}, 
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=None 
    )
    
    print('****** Which scalers are we using? ********')
    print('Feature scaling:', training.scalers)
    print('Target scaling:', training.target_normalizer)
    
    validation = TimeSeriesDataSet.from_dataset(training, val_data_raw, predict=False, stop_randomization=True)
    testing = TimeSeriesDataSet.from_dataset(training, test_data_raw, predict=False, stop_randomization=True)

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) # possibly train = False here
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=num_workers, pin_memory=True)
    test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=num_workers, pin_memory=True)
    
    return training, train_dataloader, val_dataloader, test_dataloader    

# this function for all our test countries

# this function for all our test countries

# note we need to make sure we are making these datasets from the original training dataset...

def load_prepare_test_country_data(country, training_timeseriesdataset, last_day_of_training):
    data = pd.read_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/country/'+country+'/'+country+'_dma8_non_strict_all_data_timeidx_drop_dups_drop_nas_in_o3_and_met_columns.csv')
    print('No. of unique stations =', data['station_name'].nunique())
    
    # set the 2 negative ozone values we have to a small positive number...
    data.o3[data.o3 < 0] = 0.01

    # log transform pblheight...as this improved model performance...
    data['pblheight'] = np.log(1 + data['pblheight'])

    # prepare the data
    
    # just turn datetime to pandas datetime so we can select on it for spring and summer
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d')

    # prepare the data here to have different train/val/test splits...
    training_cutoff_end_of_year = pd.to_datetime(last_day_of_training).toordinal()
    val_cutoff_end_of_year = (pd.to_datetime(last_day_of_training) + pd.offsets.DateOffset(years=1)).toordinal()
    test_cutoff_end_of_year = (pd.to_datetime(last_day_of_training) + pd.offsets.DateOffset(years=2)).toordinal()
    
    print(training_cutoff_end_of_year)
    print(val_cutoff_end_of_year)
    print(test_cutoff_end_of_year)
    
    # prepare the data here to have different train/val/test splits...
    train_data_raw_pre = data[lambda x: x.raw_time_idx < training_cutoff_end_of_year]
    train_data_raw_post = data[lambda x: x.raw_time_idx > test_cutoff_end_of_year]
    train_data_raw = pd.concat([train_data_raw_pre, train_data_raw_post])

    val_data_raw_pre = data[lambda x: x.raw_time_idx <= (val_cutoff_end_of_year)]
    val_data_raw = val_data_raw_pre[lambda x: x.raw_time_idx > (training_cutoff_end_of_year)]

    test_data_raw_pre = data[lambda x: x.raw_time_idx <= (test_cutoff_end_of_year)]
    test_data_raw = test_data_raw_pre[lambda x: x.raw_time_idx > (val_cutoff_end_of_year)]
    
    print('Min datetime of test data:', test_data_raw.datetime.min())
    print('Max datetime of test data:', test_data_raw.datetime.max())
    
    print('Train min index:', train_data_raw['raw_time_idx'].min())
    print('Train min index:', train_data_raw['raw_time_idx'].max())
    print('Val min index:', val_data_raw['raw_time_idx'].min())
    print('Val max index:', val_data_raw['raw_time_idx'].max())
    print('Test min index:', test_data_raw['raw_time_idx'].min())
    print('Test max index:', test_data_raw['raw_time_idx'].max())
    
    print('Train data percentage:', train_data_raw.shape[0]/(train_data_raw.shape[0]+val_data_raw.shape[0]+test_data_raw.shape[0]))
    print('Val data percentage:', val_data_raw.shape[0]/(train_data_raw.shape[0]+val_data_raw.shape[0]+test_data_raw.shape[0]))
    print('Test data percentage:', test_data_raw.shape[0]/(train_data_raw.shape[0]+val_data_raw.shape[0]+test_data_raw.shape[0]))
    
    # note this needs to be the original training dataset in from_dataset...
    training_country = TimeSeriesDataSet.from_dataset(training_timeseriesdataset, train_data_raw, predict=False, stop_randomization=True)
    validation_country = TimeSeriesDataSet.from_dataset(training_timeseriesdataset, val_data_raw, predict=False, stop_randomization=True)
    testing_country = TimeSeriesDataSet.from_dataset(training_timeseriesdataset, test_data_raw, predict=False, stop_randomization=True)

    train_dataloader = training_country.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) # possibly train = False here
    val_dataloader = validation_country.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=num_workers, pin_memory=True)
    test_dataloader = testing_country.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=num_workers, pin_memory=True)
    
    return train_dataloader, val_dataloader, test_dataloader

def evaluate_baseline_predictor(testing_dataset):
    
    '''
    Compute baseline predictor persistence model scores, and return the test data and the baseline predictions.
    
    Arguments: the DataLoader for the data that we want to evaluate with the Baseline persistence model.
    Output: prints the scores, and returns the actual test data and the Baseline predictions.
    
    '''
    
    actuals = torch.cat([y for x, (y,weight) in iter(testing_dataset)])
    baseline_predictions = Baseline().predict(testing_dataset)
    
    # just check they are the same shape
    print('Check they are the same shape')
    print(actuals.shape)
    print(baseline_predictions.shape)
    assert actuals.shape == baseline_predictions.shape
    
    # determine the number of rows where we have zeroes
    zeroes_in_row_actuals = (actuals == 0).sum(dim=1)
    print('No. of rows with zeroes in actuals:', (zeroes_in_row_actuals > 0).sum())
    # determine where we have nans in the baseline predictions
    nans_in_row_preds = torch.isnan(baseline_predictions).sum(dim=1)
    print('No. of rows with nans in preds:', (nans_in_row_preds > 0).sum())
    # get the row indicies where we have zeroes in the actuals, as these cause the nans in the preds

    all_row_indicies_with_zero_actuals = np.where((actuals==0))[0]
    row_indicies_zero_actuals = np.unique(all_row_indicies_with_zero_actuals)

    # check that this matches the number of rows with zeroes
    assert row_indicies_zero_actuals.shape[0] == (zeroes_in_row_actuals > 0).sum()
    # now delete rows with zeroes from actuals, then delete the same rows from baseline_predictions
    actuals_del_rows_with_zeroes = np.delete(actuals, row_indicies_zero_actuals, axis=0)
    baseline_preds_del_rows_with_zeroes = np.delete(baseline_predictions, row_indicies_zero_actuals, axis=0)

    all_row_indicies_with_nan_preds = np.where((baseline_preds_del_rows_with_zeroes==np.nan))[0]
    row_indicies_nan_preds = np.unique(all_row_indicies_with_nan_preds)
    
    # now delete rows with nans from preds, then delete the same rows from actuals
    actuals_del_rows_with_zeroes_and_nans = np.delete(actuals_del_rows_with_zeroes, row_indicies_nan_preds, axis=0)
    baseline_preds_del_rows_with_zeroes_and_nans = np.delete(baseline_preds_del_rows_with_zeroes, row_indicies_nan_preds, axis=0)    
    
    # check that both arrays now contain no zeroes and no nans
    print('No. of zeroes in new actuals:', (actuals_del_rows_with_zeroes_and_nans == 0).sum())
    print('No. of nans in new preds:', torch.isnan(baseline_preds_del_rows_with_zeroes_and_nans).sum())

    # check the shapes of the new arrays are the same
    assert actuals_del_rows_with_zeroes_and_nans.shape == baseline_preds_del_rows_with_zeroes_and_nans.shape
    
    print('')
    print('Baseline preds skill scores')
    print('')

    print('Baseline R2 score =', r2_score(actuals_del_rows_with_zeroes_and_nans[0::4, :].flatten(), baseline_preds_del_rows_with_zeroes_and_nans[0::4, :].flatten()))
    print('Baseline MSE =', mean_squared_error(actuals_del_rows_with_zeroes_and_nans[0::4, :].flatten(), baseline_preds_del_rows_with_zeroes_and_nans[0::4, :].flatten()))
    print('Baseline RMSE =', np.sqrt(mean_squared_error(actuals_del_rows_with_zeroes_and_nans[0::4, :].flatten(), baseline_preds_del_rows_with_zeroes_and_nans[0::4, :].flatten())))
    print('Baseline MAE =', mean_absolute_error(actuals_del_rows_with_zeroes_and_nans[0::4, :].flatten(), baseline_preds_del_rows_with_zeroes_and_nans[0::4, :].flatten()))
    print('Baseline MAPE =', mean_absolute_percentage_error(actuals_del_rows_with_zeroes_and_nans[0::4, :].flatten(), baseline_preds_del_rows_with_zeroes_and_nans[0::4, :].flatten()))

    print('Baseline R2 score one day ahead =', r2_score(actuals_del_rows_with_zeroes_and_nans[:, 0], baseline_preds_del_rows_with_zeroes_and_nans[:, 0]))
    print('Baseline R2 score no repeats =', r2_score(actuals_del_rows_with_zeroes_and_nans[0::4, :].flatten(), baseline_preds_del_rows_with_zeroes_and_nans[0::4, :].flatten()))
    
    return actuals_del_rows_with_zeroes_and_nans, baseline_preds_del_rows_with_zeroes_and_nans

def perform_model_predictions(testing_dataset, model):
    
    '''
    Takes test data and a trained TFT (or other e.g. LSTM) model, and evaluates the skill of the predictions.
    '''

    # load up the actuals again
    actuals = torch.cat([y for x, (y,weight) in iter(testing_dataset)])
    # here we make the predictions
    predictions = model.predict(testing_dataset)
    
    # determine the number of rows where we have zeroes
    zeroes_in_row_actuals = (actuals == 0).sum(dim=1)
    print('No. of rows with zeroes in actuals:', (zeroes_in_row_actuals > 0).sum())

    # determine where we have nans in the baseline predictions
    nans_in_row_preds = torch.isnan(predictions).sum(dim=1)
    print('No. of rows with nans in preds:', (nans_in_row_preds > 0).sum())

    # get the row indicies where we have zeroes in the actuals, as these cause the nans in the preds

    all_row_indicies_with_zero_actuals = np.where((actuals==0))[0]
    row_indicies_zero_actuals = np.unique(all_row_indicies_with_zero_actuals)

    # check that this matches the number of rows with zeroes
    assert row_indicies_zero_actuals.shape[0] == (zeroes_in_row_actuals > 0).sum()

    # now delete rows with zeroes from actuals, then delete the same rows from baseline_predictions
    actuals_del_rows_with_zeroes = np.delete(actuals, row_indicies_zero_actuals, axis=0)
    preds_del_rows_with_zeroes = np.delete(predictions, row_indicies_zero_actuals, axis=0)
    
    all_row_indicies_with_nan_preds = np.where((preds_del_rows_with_zeroes==np.nan))[0]
    row_indicies_nan_preds = np.unique(all_row_indicies_with_nan_preds)
    
    # now delete rows with nans from preds, then delete the same rows from actuals
    actuals_del_rows_with_zeroes_and_nans = np.delete(actuals_del_rows_with_zeroes, row_indicies_nan_preds, axis=0)
    preds_del_rows_with_zeroes_and_nans = np.delete(preds_del_rows_with_zeroes, row_indicies_nan_preds, axis=0)    
    
    # check that both arrays now contain no zeroes and no nans
    print('No. of zeroes in new actuals:', (actuals_del_rows_with_zeroes_and_nans == 0).sum())
    print('No. of nans in new preds:', torch.isnan(preds_del_rows_with_zeroes_and_nans).sum())

    # check the shapes of the new arrays are the same
    assert actuals_del_rows_with_zeroes_and_nans.shape == preds_del_rows_with_zeroes_and_nans.shape

    return actuals_del_rows_with_zeroes_and_nans, preds_del_rows_with_zeroes_and_nans

def evaluate_metrics(actuals, preds):
    
    # calculate and store evaluation metrics
    r = stats.pearsonr(actuals[0::4, :].flatten(), preds[0::4, :].flatten())
    r2 = r2_score(actuals[0::4, :].flatten(), preds[0::4, :].flatten())
    mse = mean_squared_error(actuals[0::4, :].flatten(), preds[0::4, :].flatten())
    rmse = np.sqrt(mean_squared_error(actuals[0::4, :].flatten(), preds[0::4, :].flatten()))
    mae = mean_absolute_error(actuals[0::4, :].flatten(), preds[0::4, :].flatten())
    mape = mean_absolute_percentage_error(actuals[0::4, :].flatten(), preds[0::4, :].flatten())
    
    
    # Print evaluation metrics
    print('')
    print('Trained model predictions skill scores')
    print('')
    
    print('Model r score =', r[0])
    print('Model R2 score =', r2)
    print('Model MSE =', mse)
    print('Model RMSE =', rmse)
    print('Model MAE =', mae)
    print('Model MAPE =', mape)
    
    print('Model R2 score one day ahead =', r2_score(actuals[:, 0], preds[:, 0]))
    print('Model R2 score no repeats =', r2_score(actuals[0::4, :].flatten(), preds[0::4, :].flatten()))
    
    return r[0], r2, mse, rmse, mae, mape


# evaluation metrics by percentile...

def evaluate_metrics_percentile(actuals, preds, percentile):
    
    # getting percentile values
    actuals_no_repeats = actuals[0::4, :].flatten()
    preds_no_repeats = preds[0::4, :].flatten()
    
    actuals_percentile_value = np.percentile(actuals_no_repeats, percentile)
    actuals_above_percentile = actuals_no_repeats[actuals_no_repeats > actuals_percentile_value]
    preds_above_percentile = preds_no_repeats[actuals_no_repeats > actuals_percentile_value]
    
    print(actuals.shape)
    print(actuals_no_repeats.shape)
    print(preds_no_repeats.shape)
    print(percentile, "th percentile of observed ozone data:", actuals_percentile_value)
    print(actuals_above_percentile.shape)
    print(preds_above_percentile.shape)
    
    # calculate and store evaluation metrics
    r = stats.pearsonr(actuals_above_percentile, preds_above_percentile)
    r2 = r2_score(actuals_above_percentile, preds_above_percentile)
    mse = mean_squared_error(actuals_above_percentile, preds_above_percentile)
    rmse = np.sqrt(mean_squared_error(actuals_above_percentile, preds_above_percentile))
    mae = mean_absolute_error(actuals_above_percentile, preds_above_percentile)
    mape = mean_absolute_percentage_error(actuals_above_percentile, preds_above_percentile)
    
    
    # Print evaluation metrics
    print('')
    print('Trained model predictions skill scores for the', percentile, 'th percentile')
    print('')
    
    print('Model r score =', r[0])
    print('Model R2 score =', r2)
    print('Model MSE =', mse)
    print('Model RMSE =', rmse)
    print('Model MAE =', mae)
    print('Model MAPE =', mape)
    
    print('Model R2 score one day ahead =', r2_score(actuals_above_percentile, preds_above_percentile))
    print('Model R2 score no repeats =', r2_score(actuals_above_percentile, preds_above_percentile))
    
    return r[0], r2, mse, rmse, mae, mape, actuals_above_percentile, preds_above_percentile


def scatter_plot_predictions_observations(actuals, predictions, model_scores, country):
    
    # line of best fit to predictions vs. observations
    coef = np.polyfit(actuals[0::4, :].flatten(), predictions[0::4, :].flatten(), 1)
    poly1d_fn = np.poly1d(coef) 
    
    
    
    fig, ax = plt.subplots(dpi=300)
    #plt.figure(dpi=200)
    x = np.arange(0,max(predictions.max(), actuals.max()))
    ax.plot(x, x, color='k')
    ax.scatter(actuals[0::4, :].flatten(), predictions[0::4, :].flatten(), alpha=0.05)
    ax.plot(actuals[0::4, :].flatten(), poly1d_fn(actuals[0::4, :].flatten()), color='orange', label='Preds vs. obs linear fit')
    ax.text(0.3, 0.9, r'R$^2$ = {1:.2f}, MAE = {4:.1f}, RMSE = {3:.1f}'.format(*model_scores), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
    ax.set_xlabel('Observations / ppb')
    ax.set_ylabel('Predictions / ppb')
    ax.set_title('Ozone predictions against observations for '+country)
    #plt.savefig('per_'+country+'_performance_final.jpg', facecolor="white")
    plt.show()
    
    
def scatter_plot_predictions_observations_percentile(actuals, predictions, percentile, model_scores, country):

    # getting percentile values
    actuals_no_repeats = actuals[0::4, :].flatten()
    preds_no_repeats = predictions[0::4, :].flatten()
    
    actuals_percentile_value = np.percentile(actuals_no_repeats, percentile)
    actuals_above_percentile = actuals_no_repeats[actuals_no_repeats > actuals_percentile_value]
    preds_above_percentile = preds_no_repeats[actuals_no_repeats > actuals_percentile_value]
    
    # line of best fit to predictions vs. observations
    coef = np.polyfit(actuals_above_percentile.flatten(), preds_above_percentile.flatten(), 1)
    poly1d_fn = np.poly1d(coef) 
    
    
    
    fig, ax = plt.subplots(dpi=300)
    #plt.figure(dpi=200)
    x = np.arange(0,max(preds_above_percentile.max(), actuals_above_percentile.max()))
    ax.plot(x, x, color='k')
    ax.scatter(actuals_above_percentile.flatten(), preds_above_percentile.flatten(), alpha=0.05)
    ax.plot(actuals_above_percentile.flatten(), poly1d_fn(actuals_above_percentile.flatten()), color='orange', label='Preds vs. obs linear fit')
    ax.text(0.3, 0.9, r'R$^2$ = {1:.2f}, MAE = {4:.1f}, RMSE = {3:.1f}'.format(*model_scores), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
    ax.set_xlabel('Observations / ppb')
    ax.set_ylabel('Predictions / ppb')
    #ax.set_title('Ozone predictions against observations for '+country)
    ax.set_title('Ozone preds against obs for '+country)
    #plt.savefig('per_'+country+'_percentile_performance_final.jpg', facecolor="white")
    plt.show()
    

def hexbin_plot_predictions_observations(actuals, predictions, model_scores, country):
    
    # line of best fit to predictions vs. observations
    coef = np.polyfit(actuals[0::4, :].flatten(), predictions[0::4, :].flatten(), 1)
    poly1d_fn = np.poly1d(coef) 
    
    
    
    fig, ax = plt.subplots(dpi=300)
    #plt.figure(dpi=200)
    x = np.arange(0,max(predictions.max(), actuals.max()))
    ax.plot(x, x, color='k')
    hb = ax.hexbin(actuals, predictions, gridsize = 100, cmap ='Blues') 
    cb = plt.colorbar(hb) 
    ax.plot(actuals[0::4, :].flatten(), poly1d_fn(actuals[0::4, :].flatten()), color='orange', label='Preds vs. obs linear fit')
    ax.text(0.38, 0.9, r'R$^2$ = {1:.2f}, MAE = {4:.1f}, RMSE = {3:.1f}'.format(*model_scores), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
    ax.set_xlabel('Observations / ppb')
    ax.set_ylabel('Predictions / ppb')
    #ax.set_title('Ozone predictions against observations for '+country)
    ax.set_title('Ozone preds against obs for '+country)
    #plt.savefig('hexbin_per_'+country+'_performance_final.jpg', facecolor="white")
    plt.show()
    
def log_hexbin_plot_predictions_observations(actuals, predictions, model_scores, country):
    
    # line of best fit to predictions vs. observations
    coef = np.polyfit(actuals[0::4, :].flatten(), predictions[0::4, :].flatten(), 1)
    poly1d_fn = np.poly1d(coef) 
    
    
    
    fig, ax = plt.subplots(dpi=300)
    #plt.figure(dpi=200)
    x = np.arange(0,max(predictions.max(), actuals.max()))
    ax.plot(x, x, color='k')
    hb = ax.hexbin(actuals, predictions, gridsize = 100, cmap ='Blues', bins='log') 
    cb = plt.colorbar(hb) 
    ax.plot(actuals[0::4, :].flatten(), poly1d_fn(actuals[0::4, :].flatten()), color='orange', label='Preds vs. obs linear fit')
    ax.text(0.38, 0.9, r'R$^2$ = {1:.2f}, MAE = {4:.1f}, RMSE = {3:.1f}'.format(*model_scores), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
    ax.set_xlabel('Observations / ppb')
    ax.set_ylabel('Predictions / ppb')
    #ax.set_title('Ozone predictions against observations for '+country)
    ax.set_title('Ozone preds against obs for '+country)

    #plt.savefig('log_hexbin_per_'+country+'_performance_final.jpg', facecolor="white")
    plt.show()

# plotting interpretability from PyTorch-Forecasting
def plot_prediction_base(
    self,
    x: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    idx: int = 0,
    add_loss_to_title: Union[Metric, torch.Tensor, bool] = False,
    show_future_observed: bool = True,
    ax=None,
    quantiles_kwargs: Dict[str, Any] = None,
    prediction_kwargs: Dict[str, Any] = None,
) -> plt.Figure:
    """
    Plot prediction of prediction vs actuals
    Args:
        x: network input
        out: network output
        idx: index of prediction to plot
        add_loss_to_title: if to add loss to title or loss function to calculate. Can be either metrics,
            bool indicating if to use loss metric or tensor which contains losses for all samples.
            Calcualted losses are determined without weights. Default to False.
        show_future_observed: if to show actuals for future. Defaults to True.
        ax: matplotlib axes to plot on
        quantiles_kwargs (Dict[str, Any]): parameters for ``to_quantiles()`` of the loss metric.
        prediction_kwargs (Dict[str, Any]): parameters for ``to_prediction()`` of the loss metric.
    Returns:
        matplotlib figure
    """
    # all true values for y of the first sample in batch
    encoder_targets = to_list(x["encoder_target"])
    decoder_targets = to_list(x["decoder_target"])
    # get predictions
    if prediction_kwargs is None:
        prediction_kwargs = {}
    if quantiles_kwargs is None:
        quantiles_kwargs = {}
    y_raws = to_list(out["prediction"])  # raw predictions - used for calculating loss
    y_hats = to_list(self.to_prediction(out, **prediction_kwargs))
    y_quantiles = to_list(self.to_quantiles(out, **quantiles_kwargs))
    # for each target, plot
    figs = []
    for y_raw, y_hat, y_quantile, encoder_target, decoder_target in zip(
        y_raws, y_hats, y_quantiles, encoder_targets, decoder_targets
    ):
        y_all = torch.cat([encoder_target[idx], decoder_target[idx]])
        max_encoder_length = x["encoder_lengths"].max()
        y = torch.cat(
            (
                y_all[: x["encoder_lengths"][idx]],
                y_all[max_encoder_length : (max_encoder_length + x["decoder_lengths"][idx])],
            ),
        )
        # move predictions to cpu
        y_hat = y_hat.detach().cpu()[idx, : x["decoder_lengths"][idx]]
        y_quantile = y_quantile.detach().cpu()[idx, : x["decoder_lengths"][idx]]
        y_raw = y_raw.detach().cpu()[idx, : x["decoder_lengths"][idx]]
        # move to cpu
        y = y.detach().cpu()
        # create figure
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        n_pred = y_hat.shape[0]
        x_obs = np.arange(-(y.shape[0] - n_pred), 0)
        x_pred = np.arange(n_pred)
        prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
        obs_color = next(prop_cycle)["color"]
        pred_color = next(prop_cycle)["color"]
        # plot observed history
        if len(x_obs) > 0:
            if len(x_obs) > 1:
                plotter = ax.plot
            else:
                plotter = ax.scatter
            plotter(x_obs, y[:-n_pred], label="observed", c=obs_color)
        if len(x_pred) > 1:
            plotter = ax.plot
        else:
            plotter = ax.scatter
        # plot observed prediction
        if show_future_observed:
            plotter(x_pred, y[-n_pred:], label=None, c=obs_color)
        # plot prediction
        plotter(x_pred, y_hat, label="predicted", c=pred_color)
        # plot predicted quantiles
        plotter(x_pred, y_quantile[:, y_quantile.shape[1] // 2], c=pred_color, alpha=0.15)
        for i in range(y_quantile.shape[1] // 2):
            if len(x_pred) > 1:
                ax.fill_between(x_pred, y_quantile[:, i], y_quantile[:, -i - 1], alpha=0.15, fc=pred_color)
            else:
                quantiles = torch.tensor([[y_quantile[0, i]], [y_quantile[0, -i - 1]]])
                ax.errorbar(
                    x_pred,
                    y[[-n_pred]],
                    yerr=quantiles - y[-n_pred],
                    c=pred_color,
                    capsize=1.0,
                )
        if add_loss_to_title is not False:
            if isinstance(add_loss_to_title, bool):
                loss = self.loss
            elif isinstance(add_loss_to_title, torch.Tensor):
                loss = add_loss_to_title.detach()[idx].item()
            elif isinstance(add_loss_to_title, Metric):
                loss = add_loss_to_title
            else:
                raise ValueError(f"add_loss_to_title '{add_loss_to_title}'' is unkown")
            if isinstance(loss, MASE):
                loss_value = loss(y_raw[None], (y[-n_pred:][None], None), y[:n_pred][None])
            elif isinstance(loss, Metric):
                loss_value = loss(y_raw[None], (y[-n_pred:][None], None))
            else:
                loss_value = loss
            ax.set_title(f"Loss {loss_value}")
        ax.set_xlabel("Time index (days)")
        #fig.legend()
        figs.append(fig)
    # return multiple of target is a list, otherwise return single figure
    if isinstance(x["encoder_target"], (tuple, list)):
        return figs
    else:
        return fig

# format plotting interpretability from PyTorch-Forecasting

def plot_prediction5(
    self,
    x: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    idx: int,
    plot_attention: bool = True,
    add_loss_to_title: bool = False,
    show_future_observed: bool = True,
    ax=None,
    **kwargs,
) -> plt.Figure:
    """
    Plot actuals vs prediction and attention

    Args:
        x (Dict[str, torch.Tensor]): network input
        out (Dict[str, torch.Tensor]): network output
        idx (int): sample index
        plot_attention: if to plot attention on secondary axis
        add_loss_to_title: if to add loss to title. Default to False.
        show_future_observed: if to show actuals for future. Defaults to True.
        ax: matplotlib axes to plot on
    Returns:
        plt.Figure: matplotlib figure
    """
    # plot prediction as normal
    fig = plot_prediction_base(best_tft,
        x=x_test_full,
        out=raw_predictions_test_full,
        idx=idx,
        #add_loss_to_title=False,
        show_future_observed=True,
        ax=ax,
        **kwargs,
    )
    #print(fig)
    # add attention on secondary axis
    if plot_attention:
        interpretation = self.interpret_output(out.iget(slice(idx, idx + 1)))
        for f in to_list(fig):
            ax = f.axes[0]
            ax2 = ax.twinx()
            ax2.set_ylabel("Attention")
            encoder_length = x["encoder_lengths"][0]
            ax2.plot(
                torch.arange(-encoder_length, 0),
                interpretation["attention"][0, -encoder_length:].detach().cpu(),
                alpha=0.2,
                color="k",
                label='attention'
            )
            f.legend(loc=(0.12,0.15))
            f.tight_layout()
    return fig

# load tft weights from checkpoint

best_tft = TemporalFusionTransformer.load_from_checkpoint('/home/jovyan/hot_ozone/forecasting/scripts/ozone-forecasting/xgixahi5/checkpoints/epoch=117-step=3775.ckpt')

country_name = 'UK, France and Italy'

training, train_dataloader, val_dataloader, test_dataloader = load_prepare_ukfrit_data_for_year_robust_features(data, '2010-12-31')

# produce the actuals and the baseline predictions, and score them
ukfrit_actuals, ukfrit_baseline = evaluate_baseline_predictor(test_dataloader)

# reproduce the actuals and then the predictions
#ukfrit_actuals_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p, ukfrit_preds_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p = perform_model_predictions(test_dataloader, best_tft)

# make predictions
# raw predictions are a dictionary from which all kinds of information including quantiles can be extracted
raw_predictions_test_full, x_test_full = best_tft.predict(test_dataloader, mode="raw", return_x=True)

raw_predictions_test_full.prediction.shape

# these prediction contain no nans

assert not torch.isnan(raw_predictions_test_full.prediction).any()

# Let's inspect how the predictive uncertainty changes over the days into the future...

# I think I wantpredictiondifference between the 1st and last quantile for days 1, 2, 3 and 4

day1_uncertainty_1st_last_quantile = raw_predictions_test_full.prediction[:, 0, 6] - raw_predictions_test_full.prediction[:, 0, 0]
day2_uncertainty_1st_last_quantile = raw_predictions_test_full.prediction[:, 1, 6] - raw_predictions_test_full.prediction[:, 1, 0]
day3_uncertainty_1st_last_quantile = raw_predictions_test_full.prediction[:, 2, 6] - raw_predictions_test_full.prediction[:, 2, 0]
day4_uncertainty_1st_last_quantile = raw_predictions_test_full.prediction[:, 3, 6] - raw_predictions_test_full.prediction[:, 3, 0]

day1_uncertainty_middle_quantile = raw_predictions_test_full.prediction[:, 0, 4] - raw_predictions_test_full.prediction[:, 0, 2]
day2_uncertainty_middle_quantile = raw_predictions_test_full.prediction[:, 1, 4] - raw_predictions_test_full.prediction[:, 1, 2]
day3_uncertainty_middle_quantile = raw_predictions_test_full.prediction[:, 2, 4] - raw_predictions_test_full.prediction[:, 2, 2]
day4_uncertainty_middle_quantile = raw_predictions_test_full.prediction[:, 3, 4] - raw_predictions_test_full.prediction[:, 3, 2]

day1_uncert_1st_last_quantile_mean = day1_uncertainty_1st_last_quantile.mean()
day2_uncert_1st_last_quantile_mean = day2_uncertainty_1st_last_quantile.mean()
day3_uncert_1st_last_quantile_mean = day3_uncertainty_1st_last_quantile.mean()
day4_uncert_1st_last_quantile_mean = day4_uncertainty_1st_last_quantile.mean()

day1_uncert_1st_last_quantile_std = day1_uncertainty_1st_last_quantile.std()
day2_uncert_1st_last_quantile_std = day2_uncertainty_1st_last_quantile.std()
day3_uncert_1st_last_quantile_std = day3_uncertainty_1st_last_quantile.std()
day4_uncert_1st_last_quantile_std = day4_uncertainty_1st_last_quantile.std()

day1_uncert_middle_quantile_mean = day1_uncertainty_middle_quantile.mean()
day2_uncert_middle_quantile_mean = day2_uncertainty_middle_quantile.mean()
day3_uncert_middle_quantile_mean = day3_uncertainty_middle_quantile.mean()
day4_uncert_middle_quantile_mean = day4_uncertainty_middle_quantile.mean()

day1_uncert_middle_quantile_std = day1_uncertainty_middle_quantile.std()
day2_uncert_middle_quantile_std = day2_uncertainty_middle_quantile.std()
day3_uncert_middle_quantile_std = day3_uncertainty_middle_quantile.std()
day4_uncert_middle_quantile_std = day4_uncertainty_middle_quantile.std()

print('Day 1 edge quantile uncertainty mean:', day1_uncert_1st_last_quantile_mean)
print('Day 2 edge quantile uncertainty mean:', day2_uncert_1st_last_quantile_mean)
print('Day 3 edge quantile uncertainty mean:', day3_uncert_1st_last_quantile_mean)
print('Day 4 edge quantile uncertainty mean:', day4_uncert_1st_last_quantile_mean)

print('Day 1 edge quantile uncertainty std:', day1_uncert_1st_last_quantile_std)
print('Day 2 edge quantile uncertainty std:', day2_uncert_1st_last_quantile_std)
print('Day 3 edge quantile uncertainty std:', day3_uncert_1st_last_quantile_std)
print('Day 4 edge quantile uncertainty std:', day4_uncert_1st_last_quantile_std)

print('Day 1 middle quantile uncertainty mean:', day1_uncert_middle_quantile_mean)
print('Day 2 middle quantile uncertainty mean:', day2_uncert_middle_quantile_mean)
print('Day 3 middle quantile uncertainty mean:', day3_uncert_middle_quantile_mean)
print('Day 4 middle quantile uncertainty mean:', day4_uncert_middle_quantile_mean)

print('Day 1 middle quantile uncertainty std:', day1_uncert_middle_quantile_std)
print('Day 2 middle quantile uncertainty std:', day2_uncert_middle_quantile_std)
print('Day 3 middle quantile uncertainty std:', day3_uncert_middle_quantile_std)
print('Day 4 middle quantile uncertainty std:', day4_uncert_middle_quantile_std)

for idx in range(4):  # plot 10 examples
    fig, ax5 = plt.subplots(dpi=200)
    ax5.set_ylabel('Ozone / ppb', fontsize=12)
    #ax5.text(-0.1, 1.05, 'B', weight='bold', horizontalalignment='center', 
    # verticalalignment='center', fontsize=28, transform = ax5.transAxes)
    #ax5.set_xticks(ticks, fontsize=12)
    #ax5.set_yticks(ticks, fontsize=12)
    ax5.set_title('TFT individual station forecast',fontsize=14)
    plot_prediction5(best_tft, x=x_test_full, out=raw_predictions_test_full, idx=idx, 
                     add_loss_to_title=False, show_future_observed=True, ax=ax5)
    fig.savefig('low_ozone_attention_values_euro_unseen.png', facecolor='white')


# need to make a little plot of these uncertainties over days...

# make two line plots of different data, with the x axis labelled day 1, day 2, etc

# make some data
x = np.arange(1, 5)

ukfrit_wide = np.array([25.8126, 27.3828, 28.4185, 29.0680])
ukfrit_narrow = np.array([7.3715, 7.9426, 8.3361, 8.5936])

euro_wide = np.array([26.2953, 27.8835, 28.8809, 29.5070])
euro_narrow = np.array([7.5233, 8.1115, 8.4982, 8.7481])


# make a figure
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

# plot the data
ax.scatter(x, ukfrit_wide, label='UK/FR/IT: 7th quantile - 1st quantile', alpha=0.3)
ax.scatter(x, ukfrit_narrow, label='UK/FR/IT: 5th quantile - 3rd quantile', alpha=0.3)

ax.scatter(x, euro_wide, label='EURO: 7th quantile - 1st quantile', alpha=0.3)
ax.scatter(x, euro_narrow, label='EURO: 5th quantile - 3rd quantile', alpha=0.3)


#ax.scatter(x, aus_first_last, label='AUS 95th quantile - 5th quantile', alpha=0.3)
#ax.scatter(x, aus_middle, label='AUS 65th quantile - 35th quantile', alpha=0.3)
#
#ax.scatter(x, bel_first_last, label='BEL 95th quantile - 5th quantile', alpha = 0.3)
#ax.scatter(x, bel_middle, label='BEL 65th quantile - 35th quantile', alpha = 0.3)
#
#ax.scatter(x, cro_first_last, label='CRO 95th quantile - 5th quantile', alpha = 0.3)
#ax.scatter(x, cro_middle, label='CRO 65th quantile - 35th quantile', alpha = 0.3)
#
#ax.scatter(x, den_first_last, label='DEN 95th quantile - 5th quantile', alpha = 0.3)
#ax.scatter(x, den_middle, label='DEN 65th quantile - 35th quantile', alpha = 0.3)
#
#ax.scatter(x, fin_first_last, label='FIN 95th quantile - 5th quantile', alpha = 0.3)
#ax.scatter(x, fin_middle, label='FIN 65th quantile - 35th quantile', alpha = 0.3)
#
#ax.scatter(x, gre_first_last, label='GRE 95th quantile - 5th quantile', alpha = 0.3)
#ax.scatter(x, gre_middle, label='GRE 65th quantile - 35th quantile', alpha = 0.3)
#
#ax.scatter(x, ned_first_last, label='NED 95th quantile - 5th quantile', alpha = 0.3)
#ax.scatter(x, ned_middle, label='NED 65th quantile - 35th quantile', alpha = 0.3)
#
#ax.scatter(x, nor_first_last, label='NOR 95th quantile - 5th quantile', alpha = 0.3)
#ax.scatter(x, nor_middle, label='NOR 65th quantile - 35th quantile', alpha = 0.3)
#
#ax.scatter(x, pol_first_last, label='POL 95th quantile - 5th quantile', alpha = 0.3)
#ax.scatter(x, pol_middle, label='POL 65th quantile - 35th quantile', alpha = 0.3)
#
#ax.scatter(x, por_first_last, label='POR 95th quantile - 5th quantile', alpha = 0.3)
#ax.scatter(x, por_middle, label='POR 65th quantile - 35th quantile', alpha = 0.3)
#
#ax.scatter(x, sp_first_last, label='SP 95th quantile - 5th quantile', alpha = 0.3)
#ax.scatter(x, sp_middle, label='SP 65th quantile - 35th quantile', alpha = 0.3)
#
#ax.scatter(x, swe_first_last, label='SWE 95th quantile - 5th quantile', alpha = 0.3)
#ax.scatter(x, swe_middle, label='SWE 65th quantile - 35th quantile', alpha = 0.3)
#
#ax.scatter(x, swi_first_last, label='SWI 95th quantile - 5th quantile', alpha = 0.3)
#ax.scatter(x, swi_middle, label='SWI 65th quantile - 35th quantile', alpha = 0.3)


ax.plot(x, ukfrit_wide,)
ax.plot(x, ukfrit_narrow,)

ax.plot(x, euro_wide,)
ax.plot(x, euro_narrow,)

# set the x axis labels
ax.set_xticks(x)
ax.set_xticklabels(['Day {}'.format(i) for i in x])

# set the legend
ax.legend(fontsize=7)
ax.set_title('Predictive uncertainty by day of forecast')
ax.set_xlabel('Day of forecast')
ax.set_ylabel('Predictive uncertainty / ppb')

ax.text(1.07, 1.05, 'A', size=30, weight='bold',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)

ax.set_ylim(0, 40)
# save the figure
fig.savefig('predictive_uncertainty_by_day_compare_ukfrit_euro_labelled.png', facecolor='white')

# show the figure
plt.show()

def raw_preds_uncertainty_euro(country_name):
    
    train_dataloader, val_dataloader, test_dataloader = load_prepare_test_country_data(country_name, training, '2010-12-31')
    raw_predictions_test_full, x_test_full = best_tft.predict(test_dataloader, mode="raw", return_x=True)
    
    # I think I wantpredictiondifference between the 1st and last quantile for days 1, 2, 3 and 4

    day1_uncertainty_1st_last_quantile = raw_predictions_test_full.prediction[:, 0, 6] - raw_predictions_test_full.prediction[:, 0, 0]
    day2_uncertainty_1st_last_quantile = raw_predictions_test_full.prediction[:, 1, 6] - raw_predictions_test_full.prediction[:, 1, 0]
    day3_uncertainty_1st_last_quantile = raw_predictions_test_full.prediction[:, 2, 6] - raw_predictions_test_full.prediction[:, 2, 0]
    day4_uncertainty_1st_last_quantile = raw_predictions_test_full.prediction[:, 3, 6] - raw_predictions_test_full.prediction[:, 3, 0]
    day1_uncertainty_middle_quantile = raw_predictions_test_full.prediction[:, 0, 4] - raw_predictions_test_full.prediction[:, 0, 2]
    day2_uncertainty_middle_quantile = raw_predictions_test_full.prediction[:, 1, 4] - raw_predictions_test_full.prediction[:, 1, 2]
    day3_uncertainty_middle_quantile = raw_predictions_test_full.prediction[:, 2, 4] - raw_predictions_test_full.prediction[:, 2, 2]
    day4_uncertainty_middle_quantile = raw_predictions_test_full.prediction[:, 3, 4] - raw_predictions_test_full.prediction[:, 3, 2]

    day1_uncert_1st_last_quantile_mean = day1_uncertainty_1st_last_quantile.mean()
    day2_uncert_1st_last_quantile_mean = day2_uncertainty_1st_last_quantile.mean()
    day3_uncert_1st_last_quantile_mean = day3_uncertainty_1st_last_quantile.mean()
    day4_uncert_1st_last_quantile_mean = day4_uncertainty_1st_last_quantile.mean()
        
    day1_uncert_middle_quantile_mean = day1_uncertainty_middle_quantile.mean()
    day2_uncert_middle_quantile_mean = day2_uncertainty_middle_quantile.mean()
    day3_uncert_middle_quantile_mean = day3_uncertainty_middle_quantile.mean()
    day4_uncert_middle_quantile_mean = day4_uncertainty_middle_quantile.mean()
    
    first_last_quantiles_days = np.array([day1_uncert_1st_last_quantile_mean, day2_uncert_1st_last_quantile_mean, day3_uncert_1st_last_quantile_mean, day4_uncert_1st_last_quantile_mean])
    middle_quantiles_days = np.array([day1_uncert_middle_quantile_mean, day2_uncert_middle_quantile_mean, day3_uncert_middle_quantile_mean, day4_uncert_middle_quantile_mean])
    
    return first_last_quantiles_days, middle_quantiles_days

aus_first_last, aus_middle = raw_preds_uncertainty_euro("Austria")
bel_first_last, bel_middle = raw_preds_uncertainty_euro("Belgium")
cro_first_last, cro_middle = raw_preds_uncertainty_euro("Croatia")
den_first_last, den_middle = raw_preds_uncertainty_euro("Denmark")
fin_first_last, fin_middle = raw_preds_uncertainty_euro("Finland")
gre_first_last, gre_middle = raw_preds_uncertainty_euro("Greece")
ned_first_last, ned_middle = raw_preds_uncertainty_euro("Netherlands")
nor_first_last, nor_middle = raw_preds_uncertainty_euro("Norway")
pol_first_last, pol_middle = raw_preds_uncertainty_euro("Poland")
por_first_last, por_middle = raw_preds_uncertainty_euro("Portugal")
sp_first_last, sp_middle = raw_preds_uncertainty_euro("Spain")
swe_first_last, swe_middle = raw_preds_uncertainty_euro("Sweden")
swi_first_last, swi_middle = raw_preds_uncertainty_euro("Switzerland")

per_country_skills = pd.DataFrame({'Country': ['Austria', 'Belgium', 'Croatia', 'Denmark', 'Finland', 'Greece', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Spain', 'Sweden', 'Switzerland'], 
                   'r2': [0.82757466, 0.75188547 , 0.862780, 0.68250127, 0.68214668, 0.7989247, 0.71669933, 0.65627735, 0.808919, 0.78460688, 0.74346525, 0.67164714, 0.81969449],
                   'mae': [5.089399, 4.8884, 4.84806, 4.42236, 4.1050, 5.679457, 5.0985, 4.15329, 4.782229, 5.64120, 5.025228, 4.0878, 5.35674],
                   'rmse': [6.5795, 6.74118, 6.1585, 5.9673, 5.324778, 7.54055, 6.68449, 5.35059, 6.24731, 7.56496, 6.6261, 5.39467, 6.91331],
                   'mape': [0.220851, 0, 0.15997, 0.179041, 0.151081, 0.190024, 0, 0.154800, 0.200755, 0.186980, 0.166011, 0.138209, 0.2396889]
                   })

day1_country_uncertainty = np.array([aus_first_last[0], bel_first_last[0], 
                                     cro_first_last[0], den_first_last[0],
                                     fin_first_last[0], gre_first_last[0],
                                     ned_first_last[0], nor_first_last[0],
                                     pol_first_last[0], nor_first_last[0], 
                                     sp_first_last[0], swe_first_last[0], 
                                     swi_first_last[0],])
                                      

# make a figure
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

# plot the data
ax.scatter(per_country_skills['mae'], day1_country_uncertainty, )
#ax.scatter(x, ukfrit_narrow, label='UK/FR/IT 65th quantile - 35th quantile')

# set the legend
#ax.legend()
ax.set_title('Predictive uncertainty for countries by mean absolute error', fontsize=10)
ax.set_xlabel('MAE / ppb', fontsize=10)
ax.set_ylabel('Predictive uncertainty (7th - 1st quantile) / ppb', fontsize=10)

ax.set_ylim(0, 35)

ax.text(1.07, 1.05, 'B', size=30, weight='bold',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)

# save the figure
fig.savefig('predictive_uncertainty_by_country_mae_labelled.png', facecolor='white')

# show the figure
plt.show()