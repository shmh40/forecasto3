# Hyperparameter optimisation with Bayesian optimisation, as implemented with Weights and Biases sweeps.

# imports

# import weights and biases
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

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# pytorch lightning and forecasting imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import Trainer

import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, SMAPE, RMSE, PoissonLoss, QuantileLoss
from pytorch_forecasting.metrics import R2
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# pytorch lightning + wandb
from pytorch_lightning.loggers import WandbLogger

WANDB_PROJECT_NAME = "name"

WANDB_DIR = "dir"

wandb_logger = WandbLogger(project="project", entity="username")

# load the data
data = pd.read_csv("path_to_csv")

# set some globals 

max_prediction_length = 4
max_encoder_length = 21  # feasibly this could be something else...

training = TimeSeriesDataSet(
    data[lambda x: x.raw_time_idx <= training_cutoff],
    time_idx="raw_time_idx",
    target="o3",
    group_ids=["station_name"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["station_type"],
    static_reals=["landcover", "pop_density", "alt", "nox_emi",
                "station_etopo_alt", "station_rel_etopo_alt",  "max_5km_pop_density", "omi_nox",
                "max_25km_pop_density", "nightlight_1km", "nightlight_max_25km", "toar_category"],
    time_varying_known_categoricals=[],
    variable_groups={},  # group of categorical variables can be treated as one variable
    #time_varying_known_reals=["time_idx", "cloudcover", "relhum", "press", "temp", "v", "u", "pblheight"] ,  
    time_varying_known_reals=["raw_time_idx", "cloudcover", "relhum", "press", "temp", "v", "u", "pblheight"], 
    time_varying_unknown_categoricals=[],  
    time_varying_unknown_reals=["o3",], 
    target_normalizer=TorchNormalizer(transformation="softplus"),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=None # or True, I am testing this here...with Seam8's edit https://github.com/jdb78/pytorch-forecasting/issues/1132
)

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

num_workers = 4
batch_size = 32  # set this between 32 to 128 - if we get thread problems, bus errors, 

train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=num_workers, pin_memory=True)   # batch_size*10 ?

# configure network and trainer
# we can set early stopping here...
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=30, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate

pl.seed_everything(43)
trainer = pl.Trainer(
    max_epochs=3000,
    auto_scale_batch_size="binsearch",
    gpus=1,
    weights_summary="top",
    gradient_clip_val=0.1,
    limit_train_batches=32,  
    callbacks=[lr_logger, early_stop_callback],
    logger=wandb_logger
    )

with wandb.init(project=WANDB_PROJECT_NAME):
    
    WANDB_DIR = "dir"

    config = wandb.config
    
    tft = TemporalFusionTransformer.from_dataset(
                    training,
                    learning_rate=config.learning_rate, 
                    hidden_size=config.hidden_size, 
                    lstm_layers=config.lstm_layers, 
                    attention_head_size=config.attention_head_size, 
                    dropout=config.dropout, 
                    hidden_continuous_size=config.hidden_continuous_size, 
                    optimizer = config.optimizer, 
                    output_size=7,  
                    loss=QuantileLoss(), 
                    log_interval=100,
                    reduce_on_plateau_patience=4,
                    logging_metrics = torch.nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), R2()])
        )
    
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        )
    
    
