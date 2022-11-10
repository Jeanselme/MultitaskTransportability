#!/usr/bin/env python
import pandas as pd
import numpy as np
import datetime
import os

import argparse
parser = argparse.ArgumentParser(description = 'Running split.')
parser.add_argument('--dataset', '-d',  type = str, default = 'random', help = 'Dataset to use for training: synthetic, info_nar, info_rand, random, regular, sar')
args = parser.parse_args()

datasets = ['info_nar', 'info_rand', 'original', 'random', 'regular', 'sar']
path = 'SyntheticClinicalPresence/data/'

# Open all data
labs = {}
for dataset in datasets:
    for file in os.listdir(path + dataset + '/'):
        if '.csv' in file:
            labs[dataset + '_' + file[:file.index('.')]] = pd.read_csv(path + dataset + '/' + file, index_col = 0).drop(columns = ['l2'])
labs = pd.concat(labs, names = ['Patient'])

outcomes = pd.read_csv(path + 'labels.csv')
outcomes = pd.concat([outcomes.set_index(dataset + '_' + outcomes.Patient.astype(str)) for dataset in datasets])

# Assign all data in training and testing
training = pd.Series((outcomes.index.get_level_values('Patient').str.contains(args.dataset)) & (outcomes.Patient < 750), index = outcomes.index)
results = 'results_synthetic/{}/'.format(args.dataset)

print('Total patients: {}'.format(len(training)))
print('Training patients: {}'.format(training.sum()))

from experiment import ShiftExperiment

def time_since_last(series):
    # Keep the time of the last observation
    times = series.dropna().index.get_level_values('Time').to_series()

    # Create a series with this time between two observation
    times_last = pd.Series(np.nan, index = series.index.get_level_values('Time'))
    times_last.loc[times] = times.values
    times_last = times_last.ffill()

    # Do the difference between index and time to the previous
    times_last = series.index.get_level_values('Time').to_series() - times_last

    # Replace time of event for time since last
    times_last.loc[times] = series.dropna().index.get_level_values('Time').to_series().diff().loc[times]
    return times_last

def time_to_next(series):
    # Keep the time of the last observation
    times = series.dropna().index.get_level_values('Time').to_series()

    # Create a series with this time between two observation
    times_next = pd.Series(np.nan, index = series.index.get_level_values('Time'))
    times_next.loc[times] = times.values
    times_next = times_next.bfill()

    # Do the difference between index and time to the previous
    times_next = times_next - series.index.get_level_values('Time').to_series()

    # Replace time of event for time since last
    times_next.loc[times] = -series.dropna().index.get_level_values('Time').to_series().diff(periods = -1).loc[times]
    return times_next

def compute(data, f = time_since_last):
    """
        Returns the table of times since last observations
    """
    times = data.groupby('Patient').apply(
        lambda x: pd.concat({c: f(x[c]) for c in x.columns}, axis = 1).sort_index()
        )
    return times

def process(data, labels):
    """
        Extracts mask and interevents
        Preprocesses the time of event and event
    """
    cov = data.copy().astype(float)
    cov = cov.groupby('Patient').ffill() 
    
    patient_mean = data.astype(float).groupby('Patient').mean()
    cov.fillna(patient_mean, inplace=True) 

    pop_mean = patient_mean.mean()
    cov.fillna(pop_mean, inplace=True) 

    # Compute time to the next event (only when observed)
    ie_to = compute(data, time_to_next).fillna(-10)
    ie_since = compute(data, time_since_last).fillna(-10)

    mask = ~data.isna() 
    time_event = pd.DataFrame((labels.Time.loc[data.index.get_level_values(0)] - data.index.get_level_values(1)).values, index = data.index)

    return cov, ie_to, ie_since, mask, time_event, ~labels.Censored.astype(bool)



layers = [[], [50], [50, 50], [50, 50, 50]]

# LOCF
last = labs.groupby('Patient').ffill().groupby('Patient').last().fillna(labs.groupby('Patient').mean().mean()) 

se = ShiftExperiment.create(model = 'deepsurv', 
                    hyper_grid = {"survival_args": [{"layers": l} for l in layers],
                        "lr" : [1e-3, 1e-4],
                        "batch": [100, 250]
                    }, 
                    path = results + 'deepsurv_last',
                    times = [0.5, 1, 1.5, 2]) 

se.train(last, outcomes.Time - 12 * np.pi, ~outcomes.Censored.astype(bool), training)

# Count
count = (~labs.isna()).groupby('Patient').sum() # Compute counts

se = ShiftExperiment.create(model = 'deepsurv', 
                    hyper_grid = {"survival_args": [{"layers": l} for l in layers],
                        "lr" : [1e-3, 1e-4],
                        "batch": [100, 250],
                    }, 
                    path = results + 'deepsurv_count',
                    times = [0.5, 1, 1.5, 2])

se.train(pd.concat([last, count], axis = 1), outcomes.Time - 12 * np.pi, ~outcomes.Censored.astype(bool), training)

hyper_grid = {
        "layers": [1, 2, 3],
        "hidden": [10, 30],
        "survival_args": [{"layers": l} for l in layers],

        "lr" : [1e-3, 1e-4],
        "batch": [100, 250]
    }

# LSTM with value
cov, ie_to, ie_since, mask, time, event = process(labs.copy(), outcomes)

se = ShiftExperiment.create(model = 'joint', 
                    hyper_grid = hyper_grid,
                    path = results + 'lstm_value',
                    times = [0.5, 1, 1.5, 2])

se.train(cov, time, event, training, ie_to, ie_since, mask)

# LSTM with mask and time
labs_selection = pd.concat([labs.copy(), labs.isna().add_suffix('_mask').astype(float), compute(labs, time_since_last).add_suffix('_time')], axis = 1)
cov, ie_to, ie_since, mask, time, event = process(labs_selection, outcomes)


se = ShiftExperiment.create(model = 'joint', 
                    hyper_grid = hyper_grid,
                    path = results + 'lstm_value+time+mask',
                    times = [0.5, 1, 1.5, 2])

se.train(cov, time, event, training, ie_to, ie_since, mask)

# Resampling
labs_resample = labs.copy()
labs_resample = labs_resample.set_index(pd.to_datetime(labs_resample.index.get_level_values('Time'), unit = 'D'), append = True) 
labs_resample = labs_resample.groupby('Patient').resample('1H', level = 2).mean() 
labs_resample.index = labs_resample.index.map(lambda x: (x[0], (x[1] - datetime.datetime(1970,1,1)).total_seconds() / (3600 * 24)))
# Ensure last time step is the same
shift = labs_resample.groupby('Patient').apply(lambda x: x.index[-1][1]) - labs.groupby('Patient').apply(lambda x: x.index[-1][1])
labs_resample.index = labs_resample.index.map(lambda x: (x[0], (x[1] - shift[x[0]])))

cov, ie_to, ie_since, mask, time, event = process(labs_resample, outcomes)

se = ShiftExperiment.create(model = 'joint', 
                    hyper_grid = hyper_grid,
                    path = results + 'lstm+resampled',
                    times = [0.5, 1, 1.5, 2])

se.train(cov, time, event, training, ie_to, ie_since, mask)

hyper_grid_joint = hyper_grid.copy()
hyper_grid_joint.update(
    {
        "weight": [0.1, 0.3, 0.5],
        "temporal": ["point"], 
        "temporal_args": [{"layers": l} for l in layers],
        "longitudinal": ["neural"], 
        "longitudinal_args": [{"layers": l} for l in layers],
        "missing": ["neural"], 
        "missing_args": [{"layers": l} for l in layers],
    }
)

# Joint full
cov, ie_to, ie_since, mask, time, event = process(labs.copy(), outcomes)

se = ShiftExperiment.create(model = 'joint', 
                    hyper_grid = hyper_grid_joint,
                    path = results + 'joint+value',
                    times = [0.5, 1, 1.5, 2])

se.train(cov, time, event, training, ie_to, ie_since, mask)

# Joint GRU-D
hyper_grid_joint_gru = hyper_grid_joint.copy()
hyper_grid_joint_gru["typ"] = ['GRUD']

labs_selection = pd.concat([labs.copy(), labs.isna().add_suffix('_mask').astype(float)], axis = 1)
cov, ie_to, ie_since, mask, time, event = process(labs_selection, outcomes)

se = ShiftExperiment.create(model = 'joint', 
                    hyper_grid = hyper_grid_joint_gru,
                    path = results + 'joint_gru_d+mask',
                    times = [0.5, 1, 1.5, 2])

se.train(cov, time, event, training, ie_to, ie_since, mask)

# Joint with full input
labs_selection = pd.concat([labs.copy(), labs.isna().add_suffix('_mask').astype(float), compute(labs, time_since_last).add_suffix('_time')], axis = 1)
cov, ie_to, ie_since, mask, time, event = process(labs_selection, outcomes)

mask_mixture = np.full(len(cov.columns), False)
mask_mixture[:len(labs.columns)] = True

hyper_grid_joint['mixture_mask'] = [mask_mixture] 

se = ShiftExperiment.create(model = 'joint', 
                    hyper_grid = hyper_grid_joint,
                    path = results + 'joint_value+time+mask',
                    times = [0.5, 1, 1.5, 2])

se.train(cov, time, event, training, ie_to, ie_since, mask)

# Joint GRU-D with full input
labs_selection = pd.concat([labs.copy(), labs.isna().add_suffix('_mask').astype(float), compute(labs, time_since_last).add_suffix('_time')], axis = 1)
cov, ie_to, ie_since, mask, time, event = process(labs_selection, outcomes)

mask_mixture = np.full(len(cov.columns), False)
mask_mixture[:len(labs.columns)] = True

hyper_grid_joint_gru['mixture_mask'] = [mask_mixture] 

se = ShiftExperiment.create(model = 'joint', 
                    hyper_grid = hyper_grid_joint_gru,
                    path = results + 'joint_gru_d_value+time+mask',
                    times = [0.5, 1, 1.5, 2])

se.train(cov, time, event, training, ie_to, ie_since, mask)

# Full Fine Tune
hyper_grid_joint['full_finetune'] = [True] 

se = ShiftExperiment.create(model = 'joint', 
                    hyper_grid = hyper_grid_joint,
                    path = results + 'joint_full_finetune_value+time+mask',
                    times = [0.5, 1, 1.5, 2])

se.train(cov, time, event, training, ie_to, ie_since, mask)

# GRUD
hyper_grid_gru = hyper_grid.copy()
hyper_grid_gru["typ"] = ['GRUD']

labs_selection = pd.concat([labs.copy(), labs.isna().add_suffix('_mask').astype(float)], axis = 1)
cov, ie_to, ie_since, mask, time, event = process(labs_selection, outcomes)

se = ShiftExperiment.create(model = 'joint', 
                    hyper_grid = hyper_grid_gru,
                    path = results + 'gru_d+mask',
                    times = [0.5, 1, 1.5, 2])

se.train(cov, time, event, training, ie_to, ie_since, mask)

hyper_grid_gru["typ"] = ['ODE']

se = ShiftExperiment.create(model = 'joint', 
                    hyper_grid = hyper_grid_gru,
                    path = results + 'ode+mask',
                    times = [0.5, 1, 1.5, 2])

se.train(cov, time, event, training, ie_to, ie_since, mask)