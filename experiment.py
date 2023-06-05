from sklearn.model_selection import ParameterSampler, train_test_split
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc

from sklearn.preprocessing import StandardScaler
from models.rnn_joint import RNNJoint
from models.deepsurv import DeepSurv

from pycox.evaluation import EvalSurv

import pandas as pd
import numpy as np
import pickle
import torch
import copy
import os
import io

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
    return times.fillna(-1)

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
    ie_to = compute(data, time_to_next)
    ie_since = compute(data, time_since_last)

    mask = ~data.isna() 
    time_event = pd.DataFrame((labels.Time.loc[data.index.get_level_values(0)] - data.index.get_level_values(1)).values, index = data.index)

    return cov, ie_to, ie_since, mask, time_event, ~labels.Censored.astype(bool)

def normalizeMinMax(data, normalizer = None):
    """
        Apply a max standardization for time data (do not remove min to ensure 0 is 0)
    """
    index = None
    if isinstance(data, list):
        data = np.array(data, dtype = float)
    elif isinstance(data, pd.DataFrame):
        index = data.index
        data = data.values

    mask = data >= 0
    if normalizer is None:
        normalizer = data[mask].max()

    normalized_data = data.copy()
    normalized_data[mask] = normalized_data[mask] / normalizer

    if index is None:
        return normalized_data, normalizer
    else:
        return pd.DataFrame(normalized_data, index = index), normalizer

def select(df, oversample):
    """
        Allows to select from a multi index with over sampling
        Ensure that the each resampled patients will have a different index
    """
    if df.index.nlevels > 1:
        results = {}
        for i, patient in enumerate(oversample):
            patient_df = df.loc[patient]
            results[i] = patient_df
        return pd.concat(results)
    else:
        return df.loc[oversample].reset_index(drop=True)

def evaluate(e, t, risk, train_index, test_index, iterations = 100, horizons = []):
    # Cast names to time
    risk.columns = risk.columns.astype(float)
    times = risk.columns.to_numpy()

    e_train, t_train, risk_train = e.loc[train_index], t.loc[train_index], risk.loc[train_index]
    e_test, t_test, risk_test = e.loc[test_index], t.loc[test_index], risk.loc[test_index]

    et_train = np.array([(e_train[i], t_train[i]) for i in e_train.index],
                     dtype = [('e', bool), ('t', float)])

    cis, rocs, brs = {t: [] for t in horizons}, {t: [] for t in horizons}, {t: [] for t in horizons}
    cis['Overall'], brs['Overall'] = [], []
    total = iterations
    for _ in range(iterations):
        bootstrap = np.random.choice(risk_test.index, size = len(risk_test), replace = True) 
        et_test = np.array([(e_test.loc[j], t_test.loc[j]) for j in bootstrap],
                            dtype = [('e', bool), ('t', float)])

        risk_iteration = risk_test.loc[bootstrap]
        survival_iteration = 1 - risk_iteration

        # Compute cumulated metrics
        km = EvalSurv(1 - risk_train.T, t_train.values, e_train.values, censor_surv = 'km')
        test_eval = EvalSurv(survival_iteration.T, t_test.loc[bootstrap].values, e_test.loc[bootstrap].values, censor_surv = km)

        try:
            cis['Overall'].append(test_eval.concordance_td())
            brs['Overall'].append(test_eval.integrated_brier_score(times))
        except Exception as e:
            continue
        try:
            indexes = [np.argmin(np.abs(times - te)) for te in horizons]
            b = brier_score(et_train, et_test, survival_iteration.iloc[:, indexes], horizons)[1]
            # Concordance and ROC for each time
            for j, (index, time) in enumerate(zip(indexes, horizons)):
                brs[time].append(b[j])
                cis[time].append(concordance_index_ipcw(et_train, et_test, risk_iteration.iloc[:, index], time)[0])
                rocs[time].append(cumulative_dynamic_auc(et_train, et_test, risk_iteration.iloc[:, index], time)[0][0])
        except Exception as e:
            continue
    print("Effective iterations: ", total)
    result = {}
    for horizon in cis:
        result.update({
          ("TD Concordance Index", 'Mean', str(horizon)): np.mean(cis[horizon]),
          ("TD Concordance Index", 'Std', str(horizon)): np.std(cis[horizon]), 
          ("Brier Score", 'Mean', str(horizon)): np.mean(brs[horizon]),
          ("Brier Score", 'Std', str(horizon)): np.std(brs[horizon]),
          ("ROC AUC", 'Mean', str(horizon)): np.mean(rocs[horizon]) if horizon in rocs else np.nan,
          ("ROC AUC", 'Std', str(horizon)): np.std(rocs[horizon]) if horizon in rocs else np.nan,
        })

    return pd.Series({r: result[r] for r in sorted(result)}), cis

class CPU_Unpickler(pickle.Unpickler):
    """
        Allow reloading of a GPU model on a CPU machine
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location = 'cpu')
        else: 
            return super().find_class(module, name)

class ToyExperiment():

    def train(self, *args, **kwargs):
        print("Results already saved")

class ShiftExperiment():

    def __init__(self, model = 'joint', hyper_grid = None, n_iter = 100, 
                random_seed = 0, normalization = True, path = 'results', save = True):
        self.model = model
        self.hyper_grid = list(ParameterSampler(hyper_grid, n_iter = n_iter, random_state = random_seed) if hyper_grid is not None else [{}])
        self.random_seed = random_seed
        
        self.iter = 0
        self.best_nll = np.inf
        self.best_hyper = None
        self.best_model = None
        self.normalization = normalization
        self.path = path
        self.tosave = save

    @classmethod
    def create(cls, model = 'joint', hyper_grid = None, n_iter = 100, 
                random_seed = 0, path = 'results', normalization = True, force = False, save = True):
        print(path)
        if not(force):
            if os.path.isfile(path + '.csv'):
                return ToyExperiment()
            elif os.path.isfile(path + '.pickle'):
                print('Loading previous copy')
                try:
                    obj = cls.load(path + '.pickle')
                    return obj
                except:
                    print('ERROR: Reinitalizing object')
                    os.remove(path + '.pickle')
                    pass
                
        return cls(model, hyper_grid, n_iter, random_seed, normalization, path, save)

    @staticmethod
    def load(path):
        file = open(path, 'rb')
        if torch.cuda.is_available():
            obj = pickle.load(file)
            obj.best_model.unpickle()
            return obj
        else:
            se = CPU_Unpickler(file).load()
            se.best_model.cuda = False
            return se

    @staticmethod
    def save(obj):
        with open(obj.path + '.pickle', 'wb') as output:
            if obj.best_model is None:
                pickle.dump(obj, output)
            else:
                obj.best_model.pickle()
                pickle.dump(obj, output)
                obj.best_model.unpickle()

                
    def save_results(self, predictions, used):
        res = pd.concat([predictions, used], axis = 1)
        
        if self.tosave:
            res.to_csv(open(self.path + '.csv', 'w'))

        return res

    def train(self, covariates, time, event, training, ie_to = None, ie_since = None, mask = None, oversampling_ratio = 0.):
        """
            Model is selected with train / test split and maximum likelihood
            Args:
                covariates (Dataframe n * d): Observed covariates
                
                time (Dataframe n): Time of censoring or event
                event (Dataframe n): Event indicator
                training (Dataframe n): Indicate which points should be used for training
                ie_to (Dataframe n * d): Interevent times (time to next observation)
                ie_since (Dataframe n * d): Interevent times (time since last observation)
                mask (Dataframe n * d): Indicator observation
                oversampling_ratio (float): Over sample data in the training set.
            Returns:
                (Dict, Dict): Dict of fitted model and Dict of observed performances
        """
        assert (covariates.index.get_level_values(0).unique() == training.index).all(), 'Misaligned indexesss'
        self.times = np.sort(np.append(np.linspace(time.min(), time.max(), 100, endpoint = False), [1, 7, 14, 30])) # Evaluate at regular points and the evaluation points

        # Split source domain into train, test and dev
        all_training = training[training].index
        training_index, test_index = train_test_split(all_training, train_size = 0.9, 
                                            random_state = self.random_seed) # For testing
        training_index, dev_index = train_test_split(training_index, train_size = 0.9, 
                                            random_state = self.random_seed) # For parameter tuning
        dev_index, val_index = train_test_split(dev_index, train_size = 0.5, 
                                            random_state = self.random_seed) # For early stopping
        annotated_training = pd.Series("Train", training.index, name = "Use")
        annotated_training.loc[test_index] = "Internal"
        annotated_training[~training] = "External"

        # Normalize data using only training data
        if self.normalization:
            self.normalizer = StandardScaler().fit(covariates.loc[training_index])
            time, self.normalizer_t = normalizeMinMax(time)
            ie_to, self.normalizer_ieto = (None, None) if ie_to is None else normalizeMinMax(ie_to) 
            ie_since, self.normalizer_iesi = (None, None) if ie_to is None else normalizeMinMax(ie_since)
            covariates = pd.DataFrame(self.normalizer.transform(covariates), index = covariates.index)

        # Oversample training data
        oversampling = training_index
        if oversampling_ratio > 0:
            oversampling = pd.Series(training_index).sample(frac = oversampling_ratio, replace = True).values

        # Split data
        train_cov, train_time, train_event = select(covariates, oversampling), select(time, oversampling), \
                                             select(event, oversampling)
        train_iet = None if ie_to is None else select(ie_to, oversampling)
        train_ies = None if ie_since is None else select(ie_since, oversampling)
        train_mask = None if mask is None else select(mask, oversampling)

        dev_cov, dev_time, dev_event = select(covariates, dev_index), select(time, dev_index), \
                                             select(event, dev_index)
        dev_iet = None if ie_to is None else select(ie_to, dev_index)
        dev_ies = None if ie_since is None else select(ie_since, dev_index)
        dev_mask = None if mask is None else select(mask, dev_index)

        val_cov, val_time, val_event = select(covariates, val_index), select(time, val_index), \
                                             select(event, val_index)
        val_iet = None if ie_to is None else select(ie_to, val_index)
        val_ies = None if ie_since is None else select(ie_since, val_index)
        val_mask = None if mask is None else select(mask, val_index)

        # Train on subset one domain
        ## Grid search best params
        for i, hyper in enumerate(self.hyper_grid):
            try:
                if i < self.iter:
                    # When object is reloaded - Avoid to recompute same parameters
                    continue
                model = self._fit(train_cov, train_iet, train_ies, train_mask, train_event, train_time, hyper, 
                                    val_cov, val_iet, val_ies, val_mask, val_event, val_time)

                if model is not None:
                    nll = self._nll(model, dev_cov, dev_iet, dev_ies, dev_mask, dev_event, dev_time)
                    if nll < self.best_nll:
                        self.best_hyper = hyper
                        self.best_model = model
                        self.best_nll = nll

                self.iter += 1
                ShiftExperiment.save(self)
            except KeyboardInterrupt as e:
                print('Interruption -> Return best results')
                break

        return self.save_results(self.predict(covariates, ie_to, ie_since, mask, training.index), annotated_training)

    def predict(self, covariates, ie_to, ie_since, mask, index = None):
        """
            Predicts the risk for each given covariates
            Data MUST be normalized in the same way
            Args:
                covariates (Dataframe): Data on which to train
                ie_to, ie_since (Dataframe n * d): Interevent times (time to next observation, time since last)
                mask (Dataframe n * d): Indicator observation
            Returns:
                Dataframe (n * len(self.time))
        """
        if self.best_model is None:
            raise ValueError('Model not trained - Call .fit')
        eval_times = normalizeMinMax(self.times, self.normalizer_t)[0].flatten().tolist() if self.normalization else self.times.tolist()
        return pd.DataFrame(1 - self.best_model.predict(covariates, ie_to, ie_since, mask, horizon = eval_times, risk = 1, batch = 50), index = index, columns = self.times)
            
    def normalize(self, cov, ie_to, ie_since, time):
        """
            Apply the same normalization than in the training of the model
        """
        time, _ = normalizeMinMax(time, self.normalizer_t)
        ie_to, _ = (None, None) if ie_to is None else normalizeMinMax(ie_to, self.normalizer_ieto) 
        ie_since, _ = (None, None) if ie_to is None else normalizeMinMax(ie_since, self.normalizer_iesi)
        cov = pd.DataFrame(self.normalizer.transform(cov), index = cov.index)
        return cov, ie_to, ie_since, time

    def _fit(self, covariates, ie_to, ie_since, mask, event, time, hyperparameter, val_cov, val_ie_to, val_ie_since, val_mask, val_event, val_time):
        """
            Fits the model on the given data
        """
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        inputdim = len(covariates.columns)
        outputdim = len(event.unique()) - 1

        lr = hyperparameter.pop('lr', 0.0001)
        batch = hyperparameter.pop('batch', 500)
        full = hyperparameter.pop('full_finetune', False)

        if self.model == "joint":
            model = RNNJoint(inputdim, outputdim, **hyperparameter)
            return model.fit(covariates, ie_to, ie_since, mask, event, time,
                             val_cov, val_ie_to, val_ie_since, val_mask, val_event, val_time, lr = lr, batch = batch, full_finetune = full)
        elif self.model == "deepsurv":
            model = DeepSurv(inputdim, outputdim, **hyperparameter)
            return model.fit(covariates, ie_to, ie_since, mask, event, time,
                             val_cov, val_ie_to, val_ie_since, val_mask, val_event, val_time, lr = lr, batch = batch)
        else:
             raise ValueError('Model {} unknown'.format(self.model))
        
    def _nll(self, model, covariates, ie_to, ie_since, mask, event, time):
        """
            Computes the negative loglikelihood of the model on the given data
        """
        return model.loss(covariates, ie_to, ie_since, mask, event, time)