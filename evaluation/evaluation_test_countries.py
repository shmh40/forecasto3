# Evaluation of TFT performance across test countries can be executed with this script.

# imports

# wandb

# basic imports
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# pytorch lightning and forecasting imports
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import TorchNormalizer
from scipy import stats

## scikit-learn imports
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")  # avoid printing out absolute paths


# set options for matplotlib and pandas
#%config InlineBackend.figure_format = 'retina'
#plt.rcParams['figure.figsize'] = 12, 6

pd.set_option('max_columns', None)

## Read in the data for the UK, France and Italy which 
# serves as training and main test data

data = pd.read_csv('path_to_data.csv')

# 2 negative ozone values
data.o3[data.o3 < 0] = 0.01

data['pblheight'] = np.log(1 + data['pblheight'])

print('Data loaded')
print('No. of unique stations =', data['station_name'].nunique())

# GLOBALS for the models

num_workers = 4
batch_size = 128  # set this between 32 to 128

# define forecast length
max_prediction_length = 4
# look-back period
max_encoder_length = 21

## Defining functions that we use to load datasets and evaluate performance.

#  we have robust scaling for all variables, including ozone, 
# and we log transform ozone and pblheight
def load_prepare_ukfrit_data_for_year_robust_features(data, last_day_of_training):
    
    '''
    This function automates the splitting of data by years, into train, validation and 
    test sets.
    last_day_of_training is a string representing a date which is the last day of the 
    training set, of the form 'YYYY-MM-DD' e.g. '2006-12-31' for the last day of 2006.
    The function returns a the training TimeSeriesDataSet, and the train, val and 
    test DataLoaders
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
    

    
    # Now create the appropriate TimeSeriesDataSets.

    # create the training time-series dataset
    training = TimeSeriesDataSet(
        train_data_raw,
        time_idx="raw_time_idx",
        target="o3",
        group_ids=["station_name"],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["station_type"],
        static_reals=["landcover", "pop_density", "nox_emi", "alt", 
                    "station_etopo_alt", "station_rel_etopo_alt", "omi_nox", 
                    "max_5km_pop_density", "max_25km_pop_density", 
                    "nightlight_1km", "nightlight_max_25km", 
                    "toar_category"],
        time_varying_known_categoricals=[],
        variable_groups={},  # group of categorical variables can be treated as one variable
        #time_varying_known_reals=["time_idx", "cloudcover", "relhum", "press", "temp", "v", "u", "pblheight"] ,  
        time_varying_known_reals=["raw_time_idx", "cloudcover", "relhum", "press", 
                                  "temp", "v", "u", "pblheight"], 
        time_varying_unknown_categoricals=[], 
        time_varying_unknown_reals=["o3",],
        target_normalizer=TorchNormalizer(method='robust', center=False, transformation="log1p"),  # use softplus and normalize across the whole train set  
        scalers={"cloudcover": RobustScaler(), "temp": RobustScaler(), "press": RobustScaler(), "relhum": RobustScaler(),
            "pblheight": RobustScaler(), "u": RobustScaler(), "v": RobustScaler()},  # s
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

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) 
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=num_workers, pin_memory=True)
    test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=num_workers, pin_memory=True)
    
    return training, train_dataloader, val_dataloader, test_dataloader    

def evaluate_baseline_predictor(testing_dataset):
    
    '''
    Compute baseline predictor persistence model scores, 
    and return the test data and the baseline predictions.
    
    Arguments: the DataLoader for the data that 
    we want to evaluate with the Baseline persistence model.
    Output: prints the scores, and returns the 
    actual test data and the Baseline predictions.
    
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
    
    print('Baseline r score =', stats.pearsonr(actuals_del_rows_with_zeroes_and_nans[0::4, :].flatten(), baseline_preds_del_rows_with_zeroes_and_nans[0::4, :].flatten()))
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
    Takes test data and a trained TFT (or other e.g. LSTM) model, 
    and evaluates the skill of the predictions.
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
    plt.savefig('per_'+country+'_performance_final.jpg', facecolor="white")
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
    plt.savefig('per_'+country+'_percentile_performance_final.jpg', facecolor="white")
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
    plt.savefig('hexbin_per_'+country+'_performance_final.jpg', facecolor="white")
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

    plt.savefig('log_hexbin_per_'+country+'_performance_final.jpg', facecolor="white")
    plt.show()
    

# Final model evaluation, for European countries. This can be executed for multiple countries by changing the country_name variable.
# This initial TFT is testing on 2012, uses robust scaling, transformation= log + 1, and with log transform of pblheight. The year of testing can be varied, loading appropriate TFT weights. Similarly we can test models without static features, by altering variables in static_reals in the TimeSeriesDataSet, and changing the model weights.

# trained tft weights, trained for this purpose.
best_tft = TemporalFusionTransformer.load_from_checkpoint('/home/jovyan/hot_ozone/forecasting/scripts/ozone-forecasting/1fx21ivy/checkpoints/epoch=129-step=4159.ckpt')

# Then test on other countries

# load appropriate training dataset again for UK/FR/IT

training, train_dataloader, val_dataloader, test_dataloader = load_prepare_ukfrit_data_for_year_robust_features(data, last_day_of_training='2010-12-31')

# this function for all our test countries
# note we need to make sure we are making these datasets from the original training dataset...

def load_prepare_test_country_data(country, training_timeseriesdataset, last_day_of_training):
    data = pd.read_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/country/'+country+'/'+country+'_dma8_non_strict_all_data_timeidx_drop_dups_drop_nas_in_o3_and_met_columns.csv')
    print('No. of unique stations =', data['station_name'].nunique())
    
    # set the 2 negative ozone values we have to a small positive number...
    data.o3[data.o3 < 0] = 0.01
    # log transform pblheight...as this improved model performance...
    data['pblheight'] = np.log(1 + data['pblheight'])


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

    train_dataloader = training_country.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) 
    val_dataloader = validation_country.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=num_workers, pin_memory=True)
    test_dataloader = testing_country.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=num_workers, pin_memory=True)
    
    return train_dataloader, val_dataloader, test_dataloader

# Then run for a particular country, here Austria.
# needs to be capitalised for file paths
country_name = 'Austria'

# load up the data into dataloaders, training, validation and testing
train_dataloader_austria, val_dataloader_austria, test_dataloader_austria = load_prepare_test_country_data(country_name, training, '2010-12-31')

# produce the actuals and the baseline predictions, and score them
austria_actuals, austria_baseline = evaluate_baseline_predictor(test_dataloader_austria)

# reproduce the actuals and then the predictions
austria_actuals, austria_preds = perform_model_predictions(test_dataloader_austria, best_tft)

# evaluate the performance of the TFT
scores = evaluate_metrics(austria_actuals, austria_preds)

# make the plots.

# for the plot
country_name = 'Austria, 2012'

scatter_plot_predictions_observations(austria_actuals, austria_preds, scores, country_name)

hexbin_plot_predictions_observations(austria_actuals, austria_preds, scores, country_name)

log_hexbin_plot_predictions_observations(austria_actuals, austria_preds, scores, country_name)