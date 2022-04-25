#!/usr/bin/env python
from tqdm import tqdm
import pandas as pd
import numpy as np

import argparse
parser = argparse.ArgumentParser(description = 'Running split.')
parser.add_argument('--mode', '-m', type = int, default = 0, help = 'Mode for training (1, -1) : (weekend, weekday); (2, -2): (male, female); 0 : Random.', choices = range(-2,3))
parser.add_argument('--sub', '-s', action='store_true', help = 'Run on subset of vitals.')
args = parser.parse_args()



# This number is used only for training, the testing happens only on the first 24 hours to ensure that
# each patient has the same impact on the final performance computation
labs = pd.read_csv('data/labs_first_day_subselection.csv', index_col = [0, 1]) if args.sub else pd.read_csv('data/labs_first_day.csv', index_col = [0, 1], header = [0, 1])
outcomes = pd.read_csv('data/outcomes_first_day{}.csv'.format('_subselection' if args.sub else ''), index_col = 0)

outcomes['Death'] = ~outcomes.Death.isna()

# # Split 
ratio = 0. 
results = 'results_subselection/' if args.mode else 'results/' 
if args.mode == 0:
    print("Applied on Random")
    training = pd.Series(outcomes.index.isin(outcomes.sample(frac = 0.8, random_state = 0).index), index = outcomes.index)
    results += 'random/'
elif args.mode == 1:
    print("Applied on Weekdays")
    training = outcomes.Day <= 4
    results += 'weekdays/'
elif args.mode == -1:
    print("Applied on Weekends")
    training = outcomes.Day > 4
    results += 'weekends/'
    ratio = (1-training).sum() / training.sum()
elif args.mode == 2:
    print("Applied on Male")
    training = outcomes.GENDER == 'M'
    results += 'male/'
    ratio = (1-training).sum() / training.sum()
elif args.mode == -2:
    print("Applied on Female")
    training = outcomes.GENDER == 'F'
    results += 'female/'

results += 'survival_'

print('Total patients: {}'.format(len(training)))
print('Training patients: {}'.format(training.sum()))

from experiment import ShiftExperiment
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

    ie_time = data.groupby("Patient").apply(lambda x: x.index.get_level_values('Time').to_series().diff().fillna(0))
    mask = ~data.isna() 
    time_event = pd.DataFrame((labels.LOS.loc[data.index.get_level_values(0)] - data.index.get_level_values(1)).values, index = data.index)

    return cov, ie_time, mask, time_event, labels.Death


layers = [[], [50], [50, 50], [50, 50, 50]]
# LOCF
last = labs.groupby('Patient').ffill().groupby('Patient').last().fillna(labs.groupby('Patient').mean().mean()) 

se = ShiftExperiment.create(model = 'deepsurv', 
                     hyper_grid = {"survival_args": [{"layers": l} for l in layers],
                        "lr" : [1e-3, 1e-4],
                        "batch": [100, 250]
                     }, 
                     path = results + 'deepsurv_last')


se.train(last, outcomes.Remaining, outcomes.Death, training, oversampling_ratio = ratio)

hyper_grid = {
        "layers": [1, 2, 3],
        "hidden": [10, 30, 60],
        "survival_args": [{"layers": l} for l in layers],

        "lr" : [1e-3, 1e-4],
        "batch": [100, 250]
    }

# LSTM with value
cov, ie, mask, time, event = process(labs.copy(), outcomes)

se = ShiftExperiment.create(model = 'joint', 
                     hyper_grid = hyper_grid,
                     path = results + 'lstm_value')


se.train(cov, time, event, training, ie, mask, oversampling_ratio = ratio)

# LSTM with input
labs_selection = pd.concat([labs.copy(), labs.isna().add_suffix('_mask').astype(float)], axis = 1)
labs_selection['Time'] = labs_selection.index.to_frame().reset_index(drop = True).groupby('Patient').diff().fillna(0).values
cov, ie, mask, time, event = process(labs_selection, outcomes)


se = ShiftExperiment.create(model = 'joint', 
                     hyper_grid = hyper_grid,
                     path = results + 'lstm_value+time+mask')


se.train(cov, time, event, training, ie, mask, oversampling_ratio = ratio)

# Resampling
labs_resample = labs.copy()
labs_resample = labs_resample.set_index(pd.to_datetime(labs_resample.index.get_level_values('Time'), unit = 'D'), append = True) 
labs_resample = labs_resample.groupby('Patient').resample('1H', level = 2).mean() 
labs_resample.index = labs_resample.index.set_levels(labs_resample.index.levels[1].hour / 24, level = 1) 

cov, ie, mask, time, event = process(labs_resample, outcomes) 

se = ShiftExperiment.create(model = 'joint', 
                     hyper_grid = hyper_grid,
                     path = results + 'lstm+resampled')


se.train(cov, time, event, training, ie, mask, oversampling_ratio = ratio)

# GRUD
hyper_grid_gru = hyper_grid.copy()
hyper_grid_gru["typ"] = ['GRUD']

labs_selection = pd.concat([labs.copy(), labs.isna().add_suffix('_mask').astype(float)], axis = 1)
cov, ie, mask, time, event = process(labs_selection, outcomes)

se = ShiftExperiment.create(model = 'joint', 
                     hyper_grid = hyper_grid_gru,
                     path = results + 'gru_d+mask')

se.train(cov, time, event, training, ie, mask, oversampling_ratio = ratio)


hyper_grid_joint = hyper_grid.copy()
hyper_grid_joint.update(
    {
        "weight": [0.1, 0.5],
        "temporal": ["point"], 
        "temporal_args": [{"layers": l} for l in layers],
        "longitudinal": ["neural"], 
        "longitudinal_args": [{"layers": l} for l in layers],
        "missing": ["neural"], 
        "missing_args": [{"layers": l} for l in layers],
    }
)


# Joint
labs_selection = labs.copy()
cov, ie, mask, time, event = process(labs_selection, outcomes)

se = ShiftExperiment.create(model = 'joint', 
                     hyper_grid = hyper_grid_joint,
                     path = results + 'joint+value')


se.train(cov, time, event, training, ie, mask, oversampling_ratio = ratio)


# Joint with input
labs_selection = pd.concat([labs.copy(), labs.isna().add_suffix('_mask').astype(float)], 1)
labs_selection['Time'] = labs_selection.index.to_frame().reset_index(drop = True).groupby('Patient').diff().fillna(0).values
cov, ie, mask, time, event = process(labs_selection, outcomes)

mask_mixture = np.full(len(cov.columns), False)
mask_mixture[:len(labs.columns)] = True

hyper_grid_joint['mixture_mask'] = [mask_mixture] 

se = ShiftExperiment.create(model = 'joint', 
                     hyper_grid = hyper_grid_joint,
                     path = results + 'joint_value+time+mask')


se.train(cov, time, event, training, ie, mask, oversampling_ratio = ratio)