from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.preprocessing import StandardScaler
from models.rnn_joint import RNNJoint
from models.deepsurv import DeepSurv
import pandas as pd
import numpy as np
import pickle
import torch
import os
import io

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
                random_seed = 0, times = [1, 7, 14, 30], normalization = True, path = 'results', save = True):
        self.model = model
        self.hyper_grid = list(ParameterSampler(hyper_grid, n_iter = n_iter, random_state = random_seed) if hyper_grid is not None else [{}])
        self.random_seed = random_seed
        self.times = times
        
        self.iter = 0
        self.best_nll = np.inf
        self.best_hyper = None
        self.best_model = None
        self.normalization = normalization
        self.path = path
        self.tosave = save

    @classmethod
    def create(cls, model = 'joint', hyper_grid = None, n_iter = 100, 
                random_seed = 0, times = [1, 7, 14, 30], path = 'results', normalization = True, force = False, save = True):
        print(path)
        if not(force):
            if os.path.isfile(path + '.csv'):
                return ToyExperiment()
            elif os.path.isfile(path + '.pickle'):
                print('Loading previous copy')
                try:
                    obj = cls.load(path + '.pickle')
                    obj.times = times
                    return obj
                except:
                    print('ERROR: Reinitalizing object')
                    os.remove(path + '.pickle')
                    pass
                
        return cls(model, hyper_grid, n_iter, random_seed, times, normalization, path, save)

    @staticmethod
    def load(path):
        file = open(path, 'rb')
        if torch.cuda.is_available():
            return pickle.load(file)
        else:
            se = CPU_Unpickler(file).load()
            se.best_model.cuda = False
            return se

    @staticmethod
    def save(obj):
        with open(obj.path + '.pickle', 'wb') as output:
            try:
                pickle.dump(obj, output)
            except Exception as e:
                print('Unable to save object')
                
    def save_results(self, predictions, used):
        res = pd.concat([predictions, used], axis = 1)
        
        if self.tosave:
            res.to_csv(open(self.path + '.csv', 'w'))

        return res

    def train(self, covariates, time, event, training, interevent = None, mask = None, oversampling_ratio = 0.):
        """
            Model is selected with train / test split and maximum likelihood

            Args:
                covariates (Dataframe n * d): Observed covariates
                interevent (Dataframe n * 1): Interevent times (time between observation)
                mask (Dataframe n * d): Indicator observation
                time (Dataframe n): Time of censoring or event
                event (Dataframe n): Event indicator
                training (Dataframe n): Indicate which points should be used for training
                oversampling_ratio (float): Over sample data in the training set.

            Returns:
                (Dict, Dict): Dict of fitted model and Dict of observed performances
        """
        # Split source domain into train, test and dev
        all_training = training[training].index
        training_index, test_index = train_test_split(all_training, train_size = 0.9, 
                                            random_state = self.random_seed) # For testing
        training_index, dev_index = train_test_split(training_index, train_size = 0.9, 
                                            random_state = self.random_seed) # For parameter tuning
        training_index, val_index = train_test_split(training_index, train_size = 0.9, 
                                            random_state = self.random_seed) # For early stopping
        annotated_training = pd.Series("Train", training.index, name = "Use")
        annotated_training.loc[test_index] = "Internal"
        annotated_training[~training] = "External"

        # Normalize data using only training data
        if self.normalization:
            self.normalizer = StandardScaler().fit(covariates.loc[training_index])
            covariates = pd.DataFrame(self.normalizer.transform(covariates), index = covariates.index)

        # Oversample training data
        oversampling = training_index
        if oversampling_ratio > 0:
            oversampling = pd.Series(training_index).sample(frac = oversampling_ratio, replace = True).values

        # Split data
        train_cov, train_time, train_event = select(covariates, oversampling), select(time, oversampling), \
                                             select(event, oversampling)
        train_ie = None if interevent is None else select(interevent, oversampling)
        train_mask = None if mask is None else select(mask, oversampling)

        dev_cov, dev_time, dev_event = covariates.loc[dev_index], time.loc[dev_index], \
                                            event.loc[dev_index]
        dev_ie = None if interevent is None else interevent.loc[dev_index]
        dev_mask = None if mask is None else mask.loc[dev_index]

        val_cov, val_time, val_event = covariates.loc[val_index], time.loc[val_index], \
                                            event.loc[val_index]
        val_ie = None if interevent is None else interevent.loc[val_index]
        val_mask = None if mask is None else mask.loc[val_index]

        # Train on subset one domain
        ## Grid search best params
        for i, hyper in enumerate(self.hyper_grid):
            if i < self.iter:
                # When object is reloaded - Avoid to recompute same parameters
                continue
            model = self._fit(train_cov, train_ie, train_mask, train_event, train_time, hyper, 
                                val_cov, val_ie, val_mask, val_event, val_time)

            if model is not None:
                nll = self._nll(model, dev_cov, dev_ie, dev_mask, dev_event, dev_time)
                if nll < self.best_nll:
                    self.best_hyper = hyper
                    self.best_model = model
                    self.best_nll = nll

            self.iter += 1
            ShiftExperiment.save(self)

        return self.save_results(self.predict(covariates, interevent, mask, training.index), annotated_training)

    def predict(self, covariates, interevent, mask, index = None):
        """
            Predicts the risk for each given covariates

            Args:
                covariates (Dataframe): Data on which to train
                interevent (Dataframe n * d): Interevent times (time between observation)
                mask (Dataframe n * d): Indicator observation

            Returns:
                Dataframe (n * len(self.time))
        """
        if self.best_model is None:
            raise ValueError('Model not trained - Call .fit')
        return pd.DataFrame(1 - self.best_model.predict(covariates, interevent, mask, horizon = self.times, risk = 1, batch = 50), index = index, columns = self.times)
            
    def _fit(self, covariates, interevent, mask, event, time, hyperparameter, val_cov, val_ie, val_mask, val_event, val_time):
        """
            Fits the model on the given data
        """
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        inputdim = len(covariates.columns)
        outputdim = len(event.unique()) - 1

        lr = hyperparameter.pop('lr', 0.0001)
        batch = hyperparameter.pop('batch', 500)

        if self.model == "joint":
            model = RNNJoint(inputdim, outputdim, **hyperparameter)
            return model.fit(covariates, interevent, mask, event, time,
                             val_cov, val_ie, val_mask, val_event, val_time, lr = lr, batch = batch)
        elif self.model == "deepsurv":
            model = DeepSurv(inputdim, outputdim, **hyperparameter)
            return model.fit(covariates, event, time,
                             val_cov, val_event, val_time, lr = lr, batch = batch)
        else:
             raise ValueError('Model {} unknown'.format(self.model))
        
    def _nll(self, model, covariates, interevent, mask, event, time):
        """
            Computes the negative loglikelihood of the model on the given data
        """
        return model.loss(covariates, interevent, mask, event, time)

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