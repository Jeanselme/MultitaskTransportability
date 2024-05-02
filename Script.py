#!/usr/bin/env python
import pandas as pd
import numpy as np
import datetime
import os

import argparse
parser = argparse.ArgumentParser(description = 'Running split.')
parser.add_argument('--mode', '-m', type = int, default = 0, help = 'Mode for training (1, -1) : (weekend, weekday); 0 : Random.', choices = range(-3,4))
parser.add_argument('--sub', '-s', action='store_true', help = 'Run on subset of vitals.')
parser.add_argument('--under', '-u', action='store_true', help = 'Undersample larger set.')
parser.add_argument('--path', '-p',  type = str, default = './', help = 'Path to get data and save results')
parser.add_argument('--debug', action='store_true', help = 'Run on subset of patients.')
parser.add_argument('--model', type = int, default = 0, help = 'Model to run')
args = parser.parse_args()


# This number is used only for training, the testing happens only on the first 24 hours to ensure that
# each patient has the same impact on the final performance computation
labs = pd.read_csv(args.path + 'data/mimic/labs_first_day{}.csv'.format('_subselection' if args.sub else ''), index_col = [0, 1])
outcomes = pd.read_csv(args.path + 'data/mimic/outcomes_first_day{}.csv'.format('_subselection' if args.sub else ''), index_col = 0)

if args.debug:
    outcomes = outcomes.sample(frac = 0.5, random_state = 0).sort_index()
    labs = labs[labs.index.get_level_values(0).isin(outcomes.index)]

if args.under:
    assert args.mode == 1, "Undersampling only works with weekdays as larger set"

# # Split 
path = args.path + 'results_subselection/' if args.sub else args.path + 'results/' 
path += 'mimic/'
if args.mode == 0:
    print("Applied on Random")
    training = pd.Series(outcomes.index.isin(outcomes.sample(frac = 0.8, random_state = 0).index), index = outcomes.index)
    results = path + 'random/'
elif args.mode == 1:
    print("Applied on Weekdays")
    training = outcomes.Day <= 4
    results = path + 'weekdays/'

    if args.under:
        print("Applied on Weekdays with undersampling")
        selection = training[training].sample(n = training.sum() - (~ training).sum(), random_state = 0).index # Compute the number that you have more
        training[selection] = False # Remove from the training set
        results += 'under_'

elif args.mode == -1:
    print("Applied on Weekends")
    training = outcomes.Day > 4
    results = path + 'weekends/'

results += 'survival_'

print('Total patients: {}'.format(len(training)))
print('Training patients: {}'.format(training.sum()))

from experiment import *

# Extract and save data
file = path + 'data.pickle'
if os.path.isfile(file):
    print('Load data non temporal')
    cov, ie_to, ie_since, mask = pickle.load(open(file, 'rb'))
else:
    print('Extract data non temporal')
    cov, ie_to, ie_since, mask = process(labs.copy(), outcomes)
    with open(file, 'wb') as output:
        pickle.dump((cov, ie_to, ie_since, mask), output)
        
# Extract and save data with time
file = path + 'data_mt.pickle'
if os.path.isfile(file):
    print('Load data temporal')
    cov_mt, ie_to_mt, ie_since_mt, mask_mt = pickle.load(open(file, 'rb'))
else:
    print('Extract data temporal')
    labs_selection = pd.concat([labs.copy(), mask.add_suffix('_mask'), ie_since.add_suffix('_time')], axis = 1)
    cov_mt, ie_to_mt, ie_since_mt, mask_mt = process(labs_selection, outcomes)
    with open(file, 'wb') as output:
        pickle.dump((cov_mt, ie_to_mt, ie_since_mt, mask_mt), output)

# Data resampled
file = path + 'data_resampled.pickle'
if os.path.isfile(file):
    print('Load data resampled')
    cov_res, ie_to_res, ie_since_res, mask_res = pickle.load(open(file, 'rb'))
else:
    print('Extract data resampled')
    # Resampling
    labs_resample = labs.copy()
    labs_resample = labs_resample.set_index(pd.to_datetime(labs_resample.index.get_level_values('Time'), unit = 'D'), append = True) 
    labs_resample = labs_resample.groupby('Patient').resample('1H', level = 2).mean() 
    labs_resample.index = labs_resample.index.map(lambda x: (x[0], (x[1] - datetime.datetime(1970,1,1)).total_seconds() / (3600 * 24)))

    # Ensure last time step is the same
    shift = labs_resample.groupby('Patient').apply(lambda x: x.index[-1][1]) - labs.groupby('Patient').apply(lambda x: x.index[-1][1])
    labs_resample.index = labs_resample.index.map(lambda x: (x[0], (x[1] - shift[x[0]])))

    cov_res, ie_to_res, ie_since_res, mask_res = process(labs_resample, outcomes)
    with open(file, 'wb') as output:
        pickle.dump((cov_res, ie_to_res, ie_since_res, mask_res), output)


# Subselect if debug
cov, ie_to, ie_since, mask, time, event = cov.loc[outcomes.index], ie_to.loc[outcomes.index], ie_since.loc[outcomes.index], mask.loc[outcomes.index], outcomes.Remaining, outcomes.Event

# Subselect if debug
cov_mt, ie_to_mt, ie_since_mt, mask_mt, time_mt, event_mt = cov_mt.loc[outcomes.index], ie_to_mt.loc[outcomes.index], ie_since_mt.loc[outcomes.index], mask_mt.loc[outcomes.index], outcomes.Remaining, outcomes.Event

# Subselect if debug
cov_res, ie_to_res, ie_since_res, mask_res, time_res, event_res = cov_res.loc[outcomes.index], ie_to_res.loc[outcomes.index], ie_since_res.loc[outcomes.index], mask_res.loc[outcomes.index], outcomes.Remaining, outcomes.Event

layers = [[], [50], [50, 50], [50, 50, 50]]

# LOCF
if (args.model == 0) or (args.model == 1):
    print('Running LOCF')
    last = labs.groupby('Patient').ffill().groupby('Patient').last()
    last.fillna(last.mean(), inplace = True)

    se = ShiftExperiment.create(model = 'deepsurv', 
                        hyper_grid = {"survival_args": [{"layers": l} for l in layers],
                            "lr" : [1e-3, 1e-4],
                            "batch": [512, 1024]
                        }, 
                        path = results + 'deepsurv_last')

    se.train(last, outcomes.Remaining, outcomes.Event, training)

# Count
if (args.model == 0) or (args.model == 2):
    print('Running Count')
    last = labs.groupby('Patient').ffill().groupby('Patient').last()
    last.fillna(last.mean(), inplace = True)
    count = (~labs.isna()).groupby('Patient').sum() # Compute counts

    se = ShiftExperiment.create(model = 'deepsurv', 
                        hyper_grid = {"survival_args": [{"layers": l} for l in layers],
                            "lr" : [1e-3, 1e-4],
                            "batch": [512, 1024]
                        }, 
                        path = results + 'deepsurv_count')

    se.train(pd.concat([last, count.add_prefix('count_')], axis = 1), outcomes.Remaining, outcomes.Event, training)

hyper_grid = {
        "layers": [1, 2],
        "hidden": [10, 25],
        "survival_args": [{"layers": l} for l in layers],

        "lr" : [1e-3, 1e-4],
        "batch": [512, 1024],
    }

# LSTM with value
if (args.model == 0) or (args.model == 3):
    print('Running LSTM Value')
    se = ShiftExperiment.create(model = 'joint', 
                        hyper_grid = hyper_grid,
                        path = results + 'lstm_value')

    se.train(cov, time, event, training, ie_to, ie_since, mask)

# LSTM with input
if (args.model == 0) or (args.model == 4):
    print('Running LSTM All')
    se = ShiftExperiment.create(model = 'joint', 
                        hyper_grid = hyper_grid,
                        path = results + 'lstm_value+time+mask')

    se.train(cov_mt, time_mt, event_mt, training, ie_to_mt, ie_since_mt, mask_mt)

# LSTM with resampled
if (args.model == 0) or (args.model == 5):
    print('Running Resample')
    se = ShiftExperiment.create(model = 'joint', 
                        hyper_grid = hyper_grid,
                        path = results + 'lstm+resampled')

    se.train(cov_res, time_res, event_res, training, ie_to_res, ie_since_res, mask_res)

# GRUD
hyper_grid_gru = hyper_grid.copy()
hyper_grid_gru["typ"] = ['GRUD']
if (args.model == 0) or (args.model == 6):
    print('Running GRU-D')
    se = ShiftExperiment.create(model = 'joint', 
                        hyper_grid = hyper_grid_gru,
                        path = results + 'gru_d')

    se.train(cov, time, event, training, ie_to, ie_since, mask)

# ODE
hyper_grid_gru = hyper_grid.copy()
hyper_grid_gru["typ"] = ['ODE']
if (args.model == 0) or (args.model == 7):
    print('Running ODE')
    se = ShiftExperiment.create(model = 'joint', 
                        hyper_grid = hyper_grid_gru,
                        path = results + 'ode')

    se.train(cov, time, event, training, ie_to, ie_since, mask)

# Hyper grid
hyper_grid_joint = hyper_grid.copy()
hyper_grid_joint.update(
    {
        "weight": [0.1, 0.3],
        "temporal": ["single"], 
        "temporal_args": [{"layers": l} for l in layers],
        "missing": ["neural"], 
        "missing_args": [{"layers": l} for l in layers],
    }
)

# Joint full
if (args.model == 0) or (args.model == 8):
    print('Running Joint')
    se = ShiftExperiment.create(model = 'joint', 
                        hyper_grid = hyper_grid_joint,
                        path = results + 'joint_value')

    se.train(cov, time, event, training, ie_to, ie_since, mask)

# Joint with full input
obs_mask = np.full(len(cov_mt.columns), False)
obs_mask[:len(labs.columns)] = True

hyper_grid_joint_temporal = hyper_grid_joint.copy()
hyper_grid_joint_temporal['obs_mask'] = [obs_mask] 

if (args.model == 0) or (args.model == 9):
    print('Running Joint All')
    se = ShiftExperiment.create(model = 'joint', 
                        hyper_grid = hyper_grid_joint_temporal,
                        path = results + 'joint_value+time+mask')

    se.train(cov_mt, time_mt, event_mt, training, ie_to_mt, ie_since_mt, mask_mt)

# ##################
# # ABLATION STUDY #
# ##################

hyper_grid_joint = hyper_grid.copy()
hyper_grid_joint.update(
    {
        "weight": [0.1, 0.3],
        "temporal": ["single"], 
        "temporal_args": [{"layers": l} for l in layers],
        "obs_mask": [obs_mask] 
    }
)
if (args.model == 0) or (args.model == 10):
    print('Running Joint Temporal')
    se = ShiftExperiment.create(model = 'joint', 
                        hyper_grid = hyper_grid_joint,
                        path = results + 'joint+time_value+time+mask')

    se.train(cov_mt, time_mt, event_mt, training, ie_to_mt, ie_since_mt, mask_mt)

hyper_grid_joint = hyper_grid.copy()
hyper_grid_joint.update(
    {
        "weight": [0.1, 0.3],
        "missing": ["neural"], 
        "missing_args": [{"layers": l} for l in layers],
        "obs_mask": [obs_mask] 
    }
)
if (args.model == 0) or (args.model == 11):
    print('Running Joint Missing')
    se = ShiftExperiment.create(model = 'joint', 
                        hyper_grid = hyper_grid_joint,
                        path = results + 'joint+missing_value+time+mask')

    se.train(cov_mt, time_mt, event_mt, training, ie_to_mt, ie_since_mt, mask_mt)


# Joint GRU-D
hyper_grid_joint_gru = hyper_grid_joint.copy()
hyper_grid_joint_gru["typ"] = ['GRUD']
if (args.model == 0) or (args.model == 12):
    print('Running Joint GRU-D')
    se = ShiftExperiment.create(model = 'joint', 
                        hyper_grid = hyper_grid_joint_gru,
                        path = results + 'joint_gru_d')

    se.train(cov, time, event, training, ie_to, ie_since, mask)

# Joint ODE
hyper_grid_joint_gru = hyper_grid_joint.copy()
hyper_grid_joint_gru["typ"] = ['ODE']
if (args.model == 0) or (args.model == 13):
    print('Running Joint ODE')
    se = ShiftExperiment.create(model = 'joint', 
                        hyper_grid = hyper_grid_joint_gru,
                        path = results + 'joint_ode')

    se.train(cov, time, event, training, ie_to, ie_since, mask)