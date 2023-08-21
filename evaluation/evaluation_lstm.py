## Evaluation of trained LSTM

# This script evaluates the skill of the LSTM on 2012 test data from the UK, France and Italy. This can be adapted to evaluate on countries such as Spain and Poland.

# imports

# wandb
import wandb

# basic imports
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import fsspec
import glob
import copy
from pathlib import Path
import warnings
import os
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")  # avoid printing out absolute paths


# pytorch lightning and forecasting imports
import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, SMAPE, RMSE, PoissonLoss, QuantileLoss, R2
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.models import BaseModel
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.base_model import AutoRegressiveBaseModelWithCovariates
from pytorch_forecasting.models.nn import LSTM
from pytorch_forecasting.models.nn import MultiEmbedding

# pytorch lightning + wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

## scikit-learn imports
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


from scipy import stats


# set options for matplotlib and pandas
#%config InlineBackend.figure_format = 'retina'
#plt.rcParams['figure.figsize'] = 12, 6

pd.set_option('max_columns', None)

# Need to set the number of inputs manually. Class from Pytorch Forecasting package.

class LSTMModelWithCovariates(AutoRegressiveBaseModelWithCovariates):
    def __init__(
        self,
        target: str,
        target_lags: Dict[str, Dict[str, int]],
        static_categoricals: List[str],
        static_reals: List[str],
        time_varying_categoricals_encoder: List[str],
        time_varying_categoricals_decoder: List[str],
        time_varying_reals_encoder: List[str],
        time_varying_reals_decoder: List[str],
        x_reals: List[str],
        x_categoricals: List[str],
        embedding_sizes: Dict[str, Tuple[int, int]],
        embedding_labels: Dict[str, List[str]],
        embedding_paddings: List[str],
        categorical_groups: Dict[str, List[str]],
        n_layers: int,
        hidden_size: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        # arguments target and target_lags are required for autoregressive models
        # even though target_lags cannot be used without covariates
        
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)
        
        # create embedder - can be fed with x["encoder_cat"] or x["decoder_cat"] and will return
        # dictionary of category names mapped to embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.hidden_size,
        )
        
        # calculate the size of all concatenated embeddings + continous variables
        n_features = sum(
            embedding_size for classes_size, embedding_size in self.hparams.embedding_sizes.values()
        ) + len(self.reals)
        
        # this is creating a network that can be fed data
        # here use version of LSTM that can handle zero-length sequences
        self.lstm = LSTM(
            #input_size=self.hparams.input_size * n_features,
            # this input is being changed manually - TO FIX with a line a bit like the above line.
            input_size = 25,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.hparams.hidden_size, 1)

    def encode(self, x: Dict[str, torch.Tensor]):
        # we need at least one encoding step as because the target needs to be lagged by one time step
        # because we use the custom LSTM, we do not have to require encoder lengths of > 1
        # but can handle lengths of >= 1
        assert x["encoder_lengths"].min() >= 1
        input_vector = x["encoder_cont"].clone()
        # lag target by one
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        input_vector = input_vector[:, 1:]  # first time step cannot be used because of lagging

        # determine effective encoder_length length
        effective_encoder_lengths = x["encoder_lengths"] - 1
        # run through LSTM network
        _, hidden_state = self.lstm(
            input_vector, lengths=effective_encoder_lengths, enforce_sorted=False  # passing the lengths directly
        )  # second ouput is not needed (hidden state)
        return hidden_state

    def decode(self, x: Dict[str, torch.Tensor], hidden_state):
        # again lag target by one
        input_vector = x["decoder_cont"].clone()
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        # but this time fill in missing target from encoder_cont at the first time step instead of throwing it away
        last_encoder_target = x["encoder_cont"][
            torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
            x["encoder_lengths"] - 1,
            self.target_positions.unsqueeze(-1),
        ].T
        input_vector[:, 0, self.target_positions] = last_encoder_target

        if self.training:  # training mode
            lstm_output, _ = self.lstm(input_vector, hidden_state, lengths=x["decoder_lengths"], enforce_sorted=False)

            # transform into right shape
            prediction = self.output_layer(lstm_output)
            prediction = self.transform_output(prediction, target_scale=x["target_scale"])

            # predictions are not yet rescaled
            return prediction

        else:  # prediction mode
            target_pos = self.target_positions

            def decode_one(idx, lagged_targets, hidden_state):
                x = input_vector[:, [idx]]
                # overwrite at target positions
                x[:, 0, target_pos] = lagged_targets[-1]  # take most recent target (i.e. lag=1)
                lstm_output, hidden_state = self.lstm(x, hidden_state)
                # transform into right shape
                prediction = self.output_layer(lstm_output)[:, 0]  # take first timestep
                return prediction, hidden_state

            # make predictions which are fed into next step
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=x["target_scale"],
                n_decoder_steps=input_vector.size(1),
            )

            # predictions are already rescaled
            return output

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hidden_state = self.encode(x)  # encode to hidden state
        output = self.decode(x, hidden_state)  # decode leveraging hidden state

        return self.to_network_output(prediction=output)


# define some functions that we use to load the datasets, and to evaluate performance.


#  we have robust scaling for all variables, including ozone, and we log transform ozone and pblheight
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
        time_varying_known_reals=["raw_time_idx", "cloudcover", "relhum", "press", "temp", "v", "u", "pblheight"], # add "no2" later,
        time_varying_unknown_categoricals=[],  ## this can be altered to do future prediction...by moving these to unknown reals
        time_varying_unknown_reals=["o3",],
        # possible options for target_normalizer here
        #target_normalizer=None,
        #target_normalizer=GroupNormalizer(groups=["station_name"], transformation="softplus"),  # use softplus and normalize by group
        target_normalizer=TorchNormalizer(method='robust', center=False, transformation="log1p"),  # use softplus and normalize across the whole train set  
        scalers={"cloudcover": RobustScaler(), "temp": RobustScaler(), "press": RobustScaler(), "relhum": RobustScaler(),
            "pblheight": RobustScaler(), "u": RobustScaler(), "v": RobustScaler()},  # s
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=None # or True, None from Seam8.
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

# this function for all our test countries

def load_prepare_test_country_data(country, training_timeseriesdataset):
    # replace with path
    data = pd.read_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/country/'+country+'/'+country+'_dma8_non_strict_all_data_timeidx_drop_dups_drop_nas_in_o3_and_met_columns.csv')
    print('No. of unique stations =', data['station_name'].nunique())
    
    # prepare the data
    train_data_raw = data[lambda x: x.raw_time_idx < training_cutoff]
    train_val_data_raw = data[lambda x: x.raw_time_idx < (training_cutoff + 365)]
    val_data_raw = train_val_data_raw[lambda x: x.raw_time_idx >= (training_cutoff)]
    test_data_raw = data[lambda x: x.raw_time_idx >= (training_cutoff + 365)]
    
    print('Train min index:', train_data_raw['raw_time_idx'].min())
    print('Train min index:', train_data_raw['raw_time_idx'].max())
    print('Val min index:', val_data_raw['raw_time_idx'].min())
    print('Val max index:', val_data_raw['raw_time_idx'].max())
    print('Test min index:', test_data_raw['raw_time_idx'].min())
    print('Test max index:', test_data_raw['raw_time_idx'].max())
    
    print('Train data percentage:', train_data_raw.shape[0]/(train_data_raw.shape[0]+val_data_raw.shape[0]+test_data_raw.shape[0]))
    print('Val data percentage:', val_data_raw.shape[0]/(train_data_raw.shape[0]+val_data_raw.shape[0]+test_data_raw.shape[0]))
    print('Test data percentage:', test_data_raw.shape[0]/(train_data_raw.shape[0]+val_data_raw.shape[0]+test_data_raw.shape[0]))
    
    training_country = TimeSeriesDataSet.from_dataset(training_timeseriesdataset, train_data_raw, predict=False, stop_randomization=True)
    validation_country = TimeSeriesDataSet.from_dataset(training_timeseriesdataset, val_data_raw, predict=False, stop_randomization=True)
    testing_country = TimeSeriesDataSet.from_dataset(training_timeseriesdataset, test_data_raw, predict=False, stop_randomization=True)

    train_dataloader = training_country.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) 
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

    
    
## here we have the data loading
data = pd.read_csv('path_to_data.csv')

# set the 2 negative ozone values we have to a small positive number.
data.o3[data.o3 < 0] = 0.01

# log transform pblheight
data['pblheight'] = np.log(1 + data['pblheight'])

print('Data loaded')
print('No. of unique stations =', data['station_name'].nunique())

# GLOBALS for the models

num_workers = 4
batch_size = 128  # set this between 32 to 128

# forecast length
max_prediction_length = 4
# choose look back length
max_encoder_length = 21

best_lstm = LSTMModelWithCovariates.load_from_checkpoint('path_to_checkpoint.ckpt')


# for plot titles
country_name = 'UK, FR, IT, LSTM, 2012'

# load up the data into dataloaders, training, validation and testing, feed this function data and the last day of training

training, train_dataloader, val_dataloader, test_dataloader = load_prepare_ukfrit_data_for_year_robust_features(data, '2010-12-31')

# reproduce the actuals and then the predictions
ukfrit_actuals_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p_2012, ukfrit_preds_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p_2012 = perform_model_predictions(test_dataloader, best_lstm)

ukfrit_actuals_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p_2012, ukfrit_baseline_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p_2012 = evaluate_baseline_predictor(test_dataloader)

# evaluate the performance of the LSTM
scores = evaluate_metrics(ukfrit_actuals_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p_2012, ukfrit_preds_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p_2012)

percentile_scores = evaluate_metrics_percentile(ukfrit_actuals_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p_2012, ukfrit_preds_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p_2012, 90)

# make and save plots
scatter_plot_predictions_observations(ukfrit_actuals_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p_2012, ukfrit_preds_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p_2012, scores, country_name)

hexbin_plot_predictions_observations(ukfrit_actuals_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p_2012, ukfrit_preds_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p_2012, scores, country_name)

log_hexbin_plot_predictions_observations(ukfrit_actuals_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p_2012, ukfrit_preds_robust_ozone_log1p_centerfalse_robust_features_pbl_log1p_2012, scores, country_name)

