# Script to train TFT on cleaned TOAR data (produced from preprocessing) from UK, France and Italy.

## This script can also be used to exclude the static features, by emptying the 
## static_reals and static_categoricals when defining the training Time SeriesDataSet.

# N.B.
# To run this excluding sequences with missing timesteps during the 21 look-back period,
# we use None for missing_timesteps here. 
# This is implemented with Seam8's edit https://github.com/jdb78/pytorch-forecasting/issues/1132

# Imports:

# import weights and biases - experiments can be logged with WandB with this script.
import wandb

# basic imports
import numpy as np
import pandas as pd
import warnings

# pytorch lightning and forecasting imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, SMAPE, RMSE, QuantileLoss
## try to import R2
from pytorch_forecasting.metrics import R2

# robust scaler
from sklearn.preprocessing import RobustScaler

# pytorch lightning + wandb
from pytorch_lightning.loggers import WandbLogger

warnings.filterwarnings("ignore")  # avoid printing out absolute paths


print('********************** Imports done ***********************')

# optional wandb login
wandb.login()
wandb.init(project="XXX", entity="YYY")

# define the wandb logger
wandb_logger = WandbLogger(project="XXX", entity="YYY")

print('********************** WandB logger created ***********************')

# here we are reading in the data. Replace with your path.

data = pd.read_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/merged_euro_clean/uk_france_italy_o3_nans_no2_no_non_strict_drop_dups.csv')

# set the 2 anomalous negative values of ozone to a small positive number.
data.o3[data.o3 < 0] = 0.01

# log transforming pblheight to improve performance
data['pblheight'] = np.log(1 + data['pblheight'])

print(data.shape)
print('CSV loaded')
        
# set global prediction and encoder length
max_prediction_length = 4
max_encoder_length = 21

# function to split data by year
def split_data_by_year(data, last_day_of_training):
    
    '''
    This function automates the splitting of data by years, into train, validation and 
    test sets.

    last_day_of_training is a string representing a date which is the last day 
    of the training set, of the form 'YYYY-MM-DD' e.g. '2006-12-31' for the last day of 
    2006.
    
    '''
    
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
    
    return(train_data_raw, val_data_raw, test_data_raw)


# set the year here for the last date of training data, 
# then the year two years after is set as test data.
# for example, 2009-12-31 as the last day of training, then the validation year is 2010,
# test year is 2011, unseen during training. Any data after the 
# testing year is also used as training data
print('Splitting the dataset')
train_data_raw, val_data_raw, test_data_raw = split_data_by_year(data, '2009-12-31')


# use time_idx or raw_time_idx depending on whether we are doing absolute years or per station years
print('Try to make the timeseriesdataset')
training = TimeSeriesDataSet(
    train_data_raw,
    time_idx="raw_time_idx",
    target="o3",
    group_ids=["station_name"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=4,
    max_prediction_length=max_prediction_length,
    static_categoricals=["station_type"],
    static_reals=["landcover", "pop_density", "alt", "nox_emi",
                "station_etopo_alt", "station_rel_etopo_alt",  "max_5km_pop_density", "omi_nox",
                "max_25km_pop_density", "nightlight_1km", "nightlight_max_25km", "toar_category"],
    time_varying_known_categoricals=[],
    variable_groups={},  # group of categorical variables can be treated as one variable
    #time_varying_known_reals=["time_idx", "cloudcover", "relhum", "press", "temp", "v", "u", "pblheight"] , 
    time_varying_known_reals=["raw_time_idx", "cloudcover", "relhum", "press", "temp", "v", "u", "pblheight"], ## this can be altered to do fully future prediction, without forecasted meteorological covariates, by moving these to unknown reals
    time_varying_unknown_categoricals=[],  
    time_varying_unknown_reals=["o3",], # these can be returned to time_varying_known_reals to do infilling
    #target_normalizer=GroupNormalizer(groups=["station_name"], transformation="softplus"),  # use softplus and normalize by group
    target_normalizer=TorchNormalizer(method='robust', center=False, transformation="log1p"), # log transform ozone 
    scalers={"cloudcover": RobustScaler(), "temp": RobustScaler(), "press": RobustScaler(), "relhum": RobustScaler(),
             "pblheight": RobustScaler(), "u": RobustScaler(), "v": RobustScaler()},  # apply robust scaling to variables
    add_target_scales=True, 
    add_encoder_length=True,
    allow_missing_timesteps=None # or True, None is implemented with Seam8's edit https://github.com/jdb78/pytorch-forecasting/issues/1132
)

validation = TimeSeriesDataSet.from_dataset(training, val_data_raw, predict=True, stop_randomization=True)

print('****** Which scalers are we using? ********')
print('Feature scaling:', training.scalers)
print('Target scaling:', training.target_normalizer)

# create dataloaders for model
# set num_workers and batch_size
num_workers = 4
batch_size = 32  # set this between 32 to 128 - if we get thread problems, bus errors...

train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=num_workers, pin_memory=True)  

# configure network and trainer
# we can set early stopping here...
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=25, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate

#logger = TensorBoardLogger("lightning_logs")  
wandb_logger = WandbLogger(project="ozone-forecasting") # logging results to wandb, replace with your project


# Instantiate the trainer with pytorch-lightning

pl.seed_everything(43)
trainer = pl.Trainer(
    max_epochs=3000,
    auto_scale_batch_size="binsearch",
    #precision=16,  # can define precision
    #gpus=2,
    accelerator='gpu', 
    devices=2, # set the number of devices
    strategy="ddp_find_unused_parameters_false",
    #strategy="ddp", # other possible strategy for GPUs
    #plugins=DDPPlugin(find_unused_parameters=False), # other possible strategy for GPUs
    #auto_select_gpus=True, # here we automatically find GPUs if desired
    weights_summary="top",
    gradient_clip_val=0.1,
    limit_train_batches=32,  # comment in for training, running validation every 30 batches
    # fast_dev_run=True,  # include to check that training has no bugs! Very useful before setting off a long run.
    callbacks=[lr_logger, early_stop_callback],
    logger=wandb_logger
)

### Here we define the actual architecture of the TFT model. These hyperparameters determined with Bayesian optimisation.
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.00993328793341612, # determined in finding_lr, usually we do learning_rate = 0.0302
    hidden_size=50, 
    lstm_layers=3, 
    attention_head_size=8, 
    dropout=0.1097384664285541, # between 0.1 and 0.3 are sensible
    hidden_continuous_size=14, 
    optimizer = 'ranger', 
    output_size=7,  # 7 quantiles by default, can choose different quantiles if wished
    loss=QuantileLoss(), 
    log_interval=100, # logging interval
    reduce_on_plateau_patience=4,
    logging_metrics = torch.nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), R2()])  # metrics to be logged to wandb
)


print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

print('************************* Start training *************************')

# for error messages
CUDA_LAUNCH_BLOCKING=1

# fit network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

