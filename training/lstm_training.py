# Script to train benchmark LSTM model
# Using PyTorch Forecasting to implement this.

# imports

# # basic imports
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# pytorch lightning and forecasting imports
import pytorch_lightning as pl
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    R2,
    RMSE,
    SMAPE,
)
from pytorch_forecasting.models.base_model import (
    AutoRegressiveBaseModelWithCovariates,
)
from pytorch_forecasting.models.nn import LSTM, MultiEmbedding
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

# pytorch lightning + wandb
from pytorch_lightning.loggers import WandbLogger

## import r2_score from scikit-learn
from sklearn.preprocessing import (
    RobustScaler,
)
from torch import nn

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

pd.set_option('max_columns', None)

# define the wandb logger
wandb_logger = WandbLogger(project="project", entity="username")

print('********************** WandB logger created ***********************')

# here we are reading in the data, replace with new path
data = pd.read_csv('/home/jovyan/lustre_scratch/cas/european_data_new_temp/merged_euro_clean/uk_france_italy_o3_nans_no2_no_non_strict_drop_dups.csv')

# set the 2 negative values of ozone to a small positive number. 
data.o3[data.o3 < 0] = 0.01

# Log transform pblheight
data['pblheight'] = np.log(1 + data['pblheight'])

print(data.shape)
print('CSV loaded')

# LSTM class from PyTorch Forecasting. Need to set the number of inputs manually.
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
            input_size = 13,  # nota bene we adjust this accordingly...
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

    
    
### Now let's convert this into a TimeSeriesDataSet

max_prediction_length = 4
max_encoder_length = 21

# split data by year
def split_data_by_year(data, last_day_of_training):
    
    '''
    This function automates the splitting of data by years, into train, validation and 
    test sets.
    last_day_of_training is a string representing a date which is the last day 
    of the training set, of the form 'YYYY-MM-DD' 
    e.g. '2006-12-31' for the last day of 2006.
    
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


# set the year here for the last date of training...
# we then take the year after for validation, and then the year after that for testing,
# then any data after the testing year is also taken as training data

print('Splitting the dataset')
train_data_raw, val_data_raw, test_data_raw = split_data_by_year(data, '2010-12-31')
 

# create the dataset from the pandas dataframe
training = TimeSeriesDataSet(
    train_data_raw,
    time_idx="raw_time_idx",
    target="o3",
    group_ids=["station_name"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=[], #"station_type"],
    static_reals=[], #"landcover", "pop_density", "alt", "nox_emi",
                #"station_etopo_alt", "station_rel_etopo_alt",  "max_5km_pop_density", "omi_nox",
                #"max_25km_pop_density", "nightlight_1km", "nightlight_max_25km", "toar_category"],
    time_varying_known_categoricals=[],
    variable_groups={},  # group of categorical variables can be treated as one variable
    #time_varying_known_reals=["time_idx", "cloudcover", "relhum", "press", "temp", "v", "u", "pblheight"] , 
    time_varying_known_reals=["raw_time_idx", "cloudcover", "relhum", "press", 
                              "temp", "v", "u", "pblheight"], ## this can be altered to do future prediction...by moving these to unknown reals
    time_varying_unknown_categoricals=[],  
    time_varying_unknown_reals=["o3",], # these can be returned to time_varying_known_reals to do infilling
    target_normalizer=TorchNormalizer(method='robust', center=False, 
                                      transformation="log1p"),
    scalers={"cloudcover": RobustScaler(), "temp": RobustScaler(), 
             "press": RobustScaler(), "relhum": RobustScaler(), 
             "pblheight": RobustScaler(), "u": RobustScaler(),
             "v": RobustScaler()}, 
    add_relative_time_idx=True, 
    add_target_scales=True, 
    add_encoder_length=True,
    allow_missing_timesteps=None # or True, None with Seam8's edit https://github.com/jdb78/pytorch-forecasting/issues/1132
)

# create validation set (predict=True) which means to 
# predict the last max_prediction_length points in time
# for each series
validation = TimeSeriesDataSet.from_dataset(training, val_data_raw, predict=True, 
                                            stop_randomization=True)

num_workers = 4
batch_size = 32  # set this between 32 to 128 - if we get thread problems, bus errors...

train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, 
                                          num_workers=num_workers, pin_memory=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, 
                                          num_workers=num_workers, pin_memory=True) 

print('****** Which scalers are we using? ********')
print('Feature scaling:', training.scalers)
print('Target scaling:', training.target_normalizer)

# hyperparameters from optimisation
model = LSTMModelWithCovariates.from_dataset(training, 
                                             n_layers=5, 
                                             hidden_size=100,
                                             logging_metrics = torch.nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])) # logging metics
# summarise the model
print(model.summarize("full"))

# configure network and trainer
# we can set early stopping here...
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=25, 
                                    verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
#logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

# instantiate trainer with pytorch lightning. Can adapt for multi GPU training.
trainer = pl.Trainer(
    max_epochs=3000,
    # testing parallelisation here
    #accelerator='gpu',
    gpus=1,
    #strategy="ddp_find_unused_parameters_false",
    weights_summary="top",
    gradient_clip_val=0.1,
    limit_train_batches=32, 
    #fast_dev_run=True,  # comment in to check that network or dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    #logger=logger,
    logger=wandb_logger
)

# Set the model.

model = LSTMModelWithCovariates.from_dataset(training, 
                                             n_layers=5, 
                                             hidden_size=100, 
                                             logging_metrics = torch.nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), R2()]))

#trainer = Trainer(fast_dev_run=True)

# fit the model.
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)