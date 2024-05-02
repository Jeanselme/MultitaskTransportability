from .Survival.survival import Survival
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch

class DeepSurv():
    """
        Encapsulator for torch to match sklearn methods
    """
    
    def __init__(self, inputdim, outputdim = 1, cuda = torch.cuda.is_available(), **params):
        self.model = Survival.create('deepsurv', inputdim, outputdim, **params)
    
        if cuda:
            self.model = self.model.cuda()
        self.model = self.model.double()
        self.fitted = False
        self.cuda = cuda

    def pickle(self):
        pass
    def unpickle(self):
        pass

    def loss(self, x, ie_to, ie_since, m, e, t, batch = None):
        x, e, t = self.preprocess(x, e, t)
        return self.model.loss(x, e, t, batch = batch).item() # Only survival loss

    def predict(self, x, ie_to, ie_since, m, horizon = None, risk = 1, batch = None):
        """
        Predict the outcome

        Args:
            x (List of Array or DataFrame n * [t_n * d]): List of Patient's time series
            i (List of Array or DataFrame n * [t_n * d]): List of inter events times 
            m (List of Array or DataFrame n * [t_n * d]): List of mask 

            horizon (List of float): Survival horizon to predict
            risk (int): RIsk to compute (use when competing risks)
            batch (int): Batch size for estimation

        Returns:
            Array: Predictions
        """
        if not self.fitted:
            raise Exception("The model has not been fitted yet.")
        x, _, _= self.preprocess(x)
        return self.model.predict(x, horizon = horizon, risk = risk, batch = batch).detach().cpu().numpy()

    def fit(self, x_train, ie_to_train, ie_since_train, m_train, e_train, t_train,
             x_valid = None, ie_to_valid = None, ie_since_valid = None, m_valid = None, e_valid = None, t_valid = None, **params):
        """
        Fit the model

        Args:
            x (List of Array or DataFrame n * [t_n * d]): List of Patient's time series
            i (List of Array or DataFrame n * [t_n * d]): List of inter events times 
            m (List of Array or DataFrame n * [t_n * d]): List of mask 
            t (List of Array or DataFrame n * [t_n], optional): List of time to event # Used for survival only
            e (List or DataFrame n, optional): List of event (binary). Defaults to None.

        Returns:
            self
        """
        x_train, e_train, t_train = self.preprocess(x_train, e_train, t_train)
        x_valid, e_valid, t_valid = self.preprocess(x_valid, e_valid, t_valid)

        self.model = train_torch_model(self.model,
            x_train, e_train, t_train,
            x_valid, e_valid, t_valid, **params).eval()

        if self.model:
            return self
        else:
            return None
        
    def fit_baseline(self, x_train, ie_to_train, ie_since_train, m_train, e_train, t_train,
                     x_valid = None, ie_to_valid = None, ie_since_valid = None, m_valid = None, e_valid = None, t_valid = None, **params):
        """ 
            Fit the baselines on the given data
        """
        x_train, e_train, t_train = self.preprocess(x_train, e_train, t_train)
        self.model.eval()
        self.model.compute_baseline(x_train, e_train, t_train, batch = 100)
        self.fitted = True
        return self

    def preprocess(self, x, e = None, t = None):
        """
        Preprocess data
            All lists need to have the same size
            
        Returns:
            6 Tensors: Padded Data, Padded Mask, Padded Interevent Time, Time to event, Target, Length
        """
        if x is None: 
            return None, None, None

        x = x.values if (isinstance(x, pd.DataFrame) or isinstance(x, pd.Series)) else x
        x = torch.DoubleTensor(x)

        if e is not None: 
            e = e.values if (isinstance(e, pd.DataFrame) or isinstance(e, pd.Series)) else e
            e = torch.DoubleTensor(e.copy()).unsqueeze(-1)
            if self.cuda:
                e = e.cuda()

        if t is not None:
            t = t.values if (isinstance(t, pd.DataFrame) or isinstance(t, pd.Series)) else t
            t = torch.DoubleTensor(t).unsqueeze(-1)

            if self.cuda:
                t = t.cuda()

        if self.cuda:
            x = x.cuda()
            
        return x, e, t

def train_torch_model(model_torch, 
    x_train, e_train, t_train,
    x_valid, e_valid, t_valid,
    epochs = 1000, lr = 0.0001, batch = 500, patience = 5, 
    weight_decay = 0.001):

    # Initialization parameters
    t_bar = tqdm(range(epochs))
    
    nbatches = int(x_train.shape[0] / batch) + 1 # Number batch
    batch_order = np.arange(x_train.shape[0]) # Index of all data in training
    best_loss = np.inf # Keep track of losses
    best_weight = deepcopy(model_torch.state_dict()) # Keep best parameters
    
    optimizer = torch.optim.Adam(model_torch.parameters(), lr = lr, weight_decay = weight_decay)

    for i in t_bar:
        model_torch.train()
        # Random batch for backprop training
        np.random.shuffle(batch_order)
        train_loss = 0
        for j in range(nbatches):
            order = batch_order[j*batch:(j+1)*batch]
            xb, eb, tb = x_train[order], e_train[order], t_train[order]

            if xb.shape[0] != batch:
                continue

            optimizer.zero_grad()
            loss = model_torch.loss(xb, eb, tb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= nbatches
        
        # Evaluate validation loss - Batch
        if x_valid is None:
            best_weight = deepcopy(model_torch.state_dict())
            continue
        
        model_torch.eval()
        loss = model_torch.loss(x_valid, e_valid, t_valid, batch = batch).item()
        
        t_bar.set_description("Loss survival: {:.3f} - Train: {:.3f}".format(loss, train_loss))
        t_bar.set_postfix({'Minimal loss observed': best_loss})

        if np.isnan(loss):
            print('ERROR - Loss')
            return None

        if loss >= best_loss:
            # If less good than before
            if wait == patience:
                break
            else:
                wait += 1
        else:
            wait = 0

            # Update new best
            best_weight = deepcopy(model_torch.state_dict())
            best_loss = loss
    
    model_torch.load_state_dict(best_weight)            
    return model_torch