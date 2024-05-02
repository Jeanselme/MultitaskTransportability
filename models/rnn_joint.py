from .rnn_joint_torch import RNNJointTorch
from .utils import pandas_to_list, torch_to_pandas, compute_dwa
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch

class RNNJoint():
    """
        Encapsulator for torch to match sklearn methods
    """
    
    def __init__(self, inputdim, outputdim = 1, cuda = torch.cuda.is_available(), **params):
        self.model = RNNJointTorch(inputdim, outputdim, **params)
    
        if cuda:
            self.model = self.model.cuda()
        self.model = self.model.double()
        self.fitted = False
        self.cuda = cuda

    def pickle(self):
        self.model.pickle()
    
    def unpickle(self):
        self.model.unpickle()
        
    def fit(self, x_train, ie_to_train, ie_since_train, m_train, e_train, t_train, 
             x_valid = None, ie_to_valid = None, ie_since_valid = None, m_valid = None, e_valid = None, t_valid = None, **params):
        """
        Fit the model

        Args:
            x (List of Array or DataFrame n * [t_n * d]): List of Patient's time series
            i_to (List of Array or DataFrame n * [t_n * d]): List of inter events times (time to the next event)
            ie_since (List of Array or DataFrame n * [t_n * d]): List of inter events times (time since the last event)
            m (List of Array or DataFrame n * [t_n * d]): List of mask 
            t (List or DataFrame n): List of time to event # Used for survival only
            e (List or DataFrame n): List of event (binary). Defaults to None.

        Returns:
            self
        """
        x_train, ie_to_train, ie_since_train, m_train, e_train, l_train, t_train = self.preprocess(x_train, ie_to_train, ie_since_train, m_train, e_train, t_train)
        x_valid, ie_to_valid, ie_since_valid, m_valid, e_valid, l_valid, t_valid = self.preprocess(x_valid, ie_to_valid, ie_since_valid, m_valid, e_valid, t_valid)

        self.model = train_torch_model(self.model,
            x_train, ie_to_train, ie_since_train, m_train, e_train, l_train, t_train, 
            x_valid, ie_to_valid, ie_since_valid, m_valid, e_valid, l_valid, t_valid, loss_survival = "last", **params)
        self.model = fine_tune_model(self.model, self.model.survival_model.parameters(), 
                x_train, ie_to_train, ie_since_train, m_train, e_train, l_train, t_train, 
                x_valid, ie_to_valid, ie_since_valid, m_valid, e_valid, l_valid, t_valid, loss_name = "survival", observational = False, **params)
        if self.model:
            return self
        else:
            return None
        
    def fit_baseline(self, x_train, ie_to_train, ie_since_train, m_train, e_train, t_train,
                     x_valid = None, ie_to_valid = None, ie_since_valid = None, m_valid = None, e_valid = None, t_valid = None, **params):
        """ 
            Fit the baselines on the given data and finetune models (avoid doing it only on the best model)
        """
        x_train, ie_to_train, ie_since_train, m_train, e_train, l_train, t_train = self.preprocess(x_train, ie_to_train, ie_since_train, m_train, e_train, t_train)
        x_valid, ie_to_valid, ie_since_valid, m_valid, e_valid, l_valid, t_valid = self.preprocess(x_valid, ie_to_valid, ie_since_valid, m_valid, e_valid, t_valid)

        if self.model.temporal:
            self.model = fine_tune_model(self.model, self.model.observational_model.temporal.parameters(),
                x_train, ie_to_train, ie_since_train, m_train, e_train, l_train, t_train, 
                x_valid, ie_to_valid, ie_since_valid, m_valid, e_valid, l_valid, t_valid, loss_name = 'temporal', **params)
        if self.model.longitudinal:
            self.model = fine_tune_model(self.model, self.model.observational_model.longitudinal.parameters(),
                x_train, ie_to_train, ie_since_train, m_train, e_train, l_train, t_train, 
                x_valid, ie_to_valid, ie_since_valid, m_valid, e_valid, l_valid, t_valid, loss_name = 'longitudinal', **params)
        if self.model.missing:
            self.model = fine_tune_model(self.model, self.model.observational_model.missing.parameters(),
                x_train, ie_to_train, ie_since_train, m_train, e_train, l_train, t_train, 
                x_valid, ie_to_valid, ie_since_valid, m_valid, e_valid, l_valid, t_valid, loss_name = 'missing', **params)
        
        self.model.compute_baseline(x_train, ie_to_train, ie_since_train, m_train, e_train, l_train, t_train, batch = None)
        self.fitted = True
        return self
            
    def predict(self, x, ie_to, ie_since, m, horizon = None, risk = 1, batch = None):
        """
        Predict the outcome

        Args:
            x (List of Array or DataFrame n * [t_n * d]): List of Patient's time series
            ie_to (List of Array or DataFrame n * [t_n * d]): List of inter events times (time to the next event)
            ie_since (List of Array or DataFrame n * [t_n * d]): List of inter events times (time since the last event)
            m (List of Array or DataFrame n * [t_n * d]): List of mask 

            horizon (List of float): Survival horizon to predict
            risk (int): RIsk to compute (use when competing risks)
            batch (int): Batch size for estimation

        Returns:
            Array: Predictions
        """
        if not self.fitted:
            raise Exception("The model has not been fitted yet.")
        x, ie_to, ie_since, m, _, l, _ = self.preprocess(x, ie_to, ie_since, m)
        return self.model.predict(x, ie_to, ie_since, m, l, horizon = horizon, risk = risk, batch = batch).detach().cpu().numpy()

    def observation_predict(self, x, ie_to, ie_since, m, batch = None):
        if not self.fitted:
            raise Exception("The model has not been fitted yet.")
        x_transform, ie_to, ie_since, m, _, l, _ = self.preprocess(x, ie_to, ie_since, m)
        return {name: torch_to_pandas(i, x, self.model.obs_mask) for name, i in zip(['Temporal', 'Longitudinal', 'Missing'], self.model.observational_predict(x_transform, ie_to, ie_since, m, l, batch = batch))}

    def loss(self, x, ie_to, ie_since, m, e, t, batch = None):
        x, ie_to, ie_since, m, e, l, t = self.preprocess(x, ie_to, ie_since, m, e, t)
        return self.model.loss(x, ie_to, ie_since, m, e, l, t, batch, observational = False)[0].item() # Only survival loss

    def likelihood_observation_predict(self, x, ie_to, ie_since, m, batch = None):
        """
            Computes the likelihood of the time series (each dimension independently)
        """
        x_transform, ie_to, ie_since, m, _, l, _ = self.preprocess(x, ie_to, ie_since, m)
        return {name: torch_to_pandas(i, x, self.model.obs_mask) for name, i in zip(['Temporal', 'Longitudinal', 'Missing'], self.model.loss(x_transform, ie_to, ie_since, m, _, l, _, batch, survival = False, observational = True, reduction = 'none')[1]['observational'])}

    def feature_importance(self, x, ie_to, ie_since, m, e, t, n = 100, batch = None):
        x_p, ie_to_p, ie_since_p, m_p, e_p, l_p, t_p = self.preprocess(x, ie_to, ie_since, m, e, t)
        global_nll = self.model.loss(x_p, ie_to_p, ie_since_p, m_p, e_p, l_p, t_p, batch)[1]
        if 'observational' in global_nll:
            global_nll =  {'Survival': global_nll['survival'].item(), 
                           'Temporal': global_nll['observational'][0].item(), 
                           'Longitudinal': global_nll['observational'][1].item(), 
                           'Missing': global_nll['observational'][2].item()}
        else:
            global_nll =  {'Survival': global_nll['survival'].item()}

        performances = {c: {j: [] for j in range(x.shape[1])} for c in global_nll}
        for _ in tqdm(range(n)):
            for j in range(x.shape[1]):
                # Permute all values of one feature (between patients and across time)
                x_updated = x.values.copy()
                np.random.shuffle(x_updated[:, j])
                x_updated = pd.DataFrame(x_updated, index = x.index, columns = x.columns)     
                x_p, ie_to_p, ie_since_p, m_p, e_p, l_p, t_p = self.preprocess(x_updated, ie_to, ie_since, m, e, t)

                nll = self.model.loss(x_p, ie_to_p, ie_since_p, m_p, e_p, l_p, t_p, batch)[1]
                performances['Survival'][j].append(nll['survival'].item())

                if 'observational' in nll:
                    performances['Temporal'][j].append(nll['observational'][0].item())
                    performances['Longitudinal'][j].append(nll['observational'][1].item())
                    performances['Missing'][j].append(nll['observational'][2].item())

        return pd.concat({c: pd.Series({x.columns[j]: (np.array(performances[c][j]) - global_nll[c]) / abs(global_nll[c]) for j in performances[c]}) for c in performances})
          

    def preprocess(self, x, ie_to, ie_since, m, e = None, t = None):
        """
        Preprocess data
            All lists need to have the same size
            
        Returns:
            6 Tensors: Padded Data, Padded Mask, Padded Interevent Time, Time to event, Target, Length
        """
        if x is None: 
            return None, None, None, None, None, None

        x = pandas_to_list(x)
        ie_to = pandas_to_list(ie_to)
        ie_since = pandas_to_list(ie_since)
        m = pandas_to_list(m)

        # X and T Padding
        xres, itres, isres, mres, l = [], [], [], [], [len(xi) for xi in x]
        max_length = max(l)
        for xi, iti, isi, mi, li in zip(x, ie_to, ie_since, m, l):
            xres.append(np.concatenate([xi, np.zeros(shape = (max_length - li, xi.shape[1]))]))
            itres.append(np.concatenate([iti, np.full(fill_value = -1, shape = (max_length - li, xi.shape[1]))])) # -1 unobserved
            isres.append(np.concatenate([isi, np.full(fill_value = -1, shape = (max_length - li, xi.shape[1]))]))# -1 unobserved
            mres.append(np.concatenate([mi, np.zeros(shape = (max_length - li, mi.shape[1]))])) # 0 means unobserved
        x = torch.from_numpy(np.array(xres, dtype=float)).double()
        ie_to = torch.from_numpy(np.array(itres, dtype=float)).double()
        ie_since = torch.from_numpy(np.array(isres, dtype=float)).double()
        m = torch.from_numpy(np.array(mres, dtype=float)) > 0.5
        l = torch.LongTensor(l)

        if e is not None: 
            e = e.values if (isinstance(e, pd.DataFrame) or isinstance(e, pd.Series)) else e
            e = torch.DoubleTensor(e.copy()).unsqueeze(-1)
            if self.cuda:
                e = e.cuda()

        if t is not None:
            t = t.values if (isinstance(t, pd.DataFrame) or isinstance(t, pd.Series)) else t
            t = torch.DoubleTensor(t.copy()).unsqueeze(-1)
            if self.cuda:
                t = t.cuda()

        if self.cuda:
            x, ie_to, ie_since, m, l = x.cuda(), ie_to.cuda(), ie_since.cuda(), m.cuda(), l.cuda()
            
        return x, ie_to, ie_since, m, e, l, t

def train_torch_model(model_torch, 
    x_train, ie_to_train, ie_since_train, m_train, e_train, l_train, t_train,
    x_valid, ie_to_valid, ie_since_valid, m_valid, e_valid, l_valid, t_valid,
    loss_survival = "last", epochs = 500, lr = 0.0001, batch = 500, patience = 5, 
    weight_decay = 0.001):

    # Initialization parameters
    t_bar = tqdm(range(epochs + 1))
    nbatches = int(x_train.shape[0] / batch) + 1 # Number batch
    batch_order = np.arange(x_train.shape[0]) # Index of all data in training
    train_loss = 0 # Keep track of losses
    losses, previous_losses, best_loss = None, None, np.inf # Observational weighting (different losses are weighted differently)
    best_weight = deepcopy(model_torch.state_dict()) # Keep best parameters

    optimizer = torch.optim.Adam(model_torch.parameters(), lr = lr, weight_decay = weight_decay)

    for i in t_bar:
        # Evaluate validation loss - Batch
        if x_valid is None:
            best_weight = deepcopy(model_torch.state_dict())
            losses_val = losses
        else: 
            model_torch.eval()
            loss, losses_val = model_torch.loss(x_valid, ie_to_valid, ie_since_valid,
                                    m_valid, e_valid, l_valid, t_valid, survival = loss_survival,
                                    batch = batch, observational = model_torch.observational)
            
            loss = loss.item()
            losses_val = losses_val['observational'].detach().clone() if model_torch.observational else None

            t_bar.set_description("Loss {}: {:.3f} - Training: {:.3f}".format(loss_survival, loss, train_loss))
            t_bar.set_postfix({'Minimal survival observed': best_loss})

            if loss > best_loss:
                if wait == patience:
                    break
                else:
                    wait += 1
            else:
                wait = 0

                # Update new best
                best_weight = deepcopy(model_torch.state_dict())
                best_loss = loss
        if i == epochs:
            break
        
        weights = compute_dwa(losses_val, previous_losses)
        previous_losses = losses_val
        model_torch.train()
        # Random batch for backprop training
        np.random.shuffle(batch_order)
        train_loss = 0.
        for j in range(nbatches):
            order = batch_order[j*batch:(j+1)*batch] # Need to conserve order
            xb, itb, isb, mb, tb, eb, lb = x_train[order], ie_to_train[order], ie_since_train[order], m_train[order],\
                                     t_train[order], e_train[order], l_train[order]
            
            # Reduce size data to all match batch
            lbmax = lb.max()
            xb, itb, isb, mb, tb = xb[:, :lbmax], itb[:, :lbmax], isb[:, :lbmax], mb[:, :lbmax], tb[:, :lbmax]

            if xb.shape[0] != batch:
                continue

            optimizer.zero_grad()
            loss, losses_batch = model_torch.loss(xb, itb, isb, mb, eb, lb, tb, survival = loss_survival,
                        observational = model_torch.observational, weights = weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if model_torch.observational:
               losses = losses_batch['observational'].detach().clone() if (j == 0) else (losses + losses_batch['observational'].detach().clone())
        train_loss /= nbatches
        if model_torch.observational:
            losses /= nbatches

    model_torch.load_state_dict(best_weight)
    model_torch.eval()        
    return model_torch

def fine_tune_model(model_torch, parameters,
    x_train, ie_to_train, ie_since_train, m_train, e_train, l_train, t_train,
    x_valid, ie_to_valid, ie_since_valid, m_valid, e_valid, l_valid, t_valid,
    epochs = 500, lr = 0.0001, batch = 500, patience = 5, 
    weight_decay = 0.001, observational = True, loss_name = ''):

    # Initialization parameters
    t_bar = tqdm(range(epochs + 1))
    nbatches = int(x_train.shape[0] / batch) + 1 # Number batch
    batch_order = np.arange(x_train.shape[0]) # Index of all data in training
    train_loss, best_loss, best_losses = 0, np.inf, None # Keep track of losses
    best_weight = deepcopy(model_torch.state_dict()) # Keep best parameters

    optimizer = torch.optim.Adam(parameters, lr = lr, weight_decay = weight_decay)

    for i in t_bar:
        # Evaluate validation loss - Batch (best model sometimes is the original one)
        if x_valid is None:
            best_weight = deepcopy(model_torch.state_dict())
        else:
            model_torch.eval()
            loss, losses = model_torch.loss(x_valid, ie_to_valid, ie_since_valid,
                                    m_valid, e_valid, l_valid, t_valid,
                                    batch = batch, observational = observational)
            
            loss = loss.item()

            t_bar.set_description("Loss finetune {}: {:.3f} - Training: {:.3f}".format(loss_name, loss, train_loss))
            t_bar.set_postfix({'Minimal survival observed': np.inf if best_losses is None else best_losses['survival'].item()})
            
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
                best_losses = losses
        if i == epochs:
            break

        model_torch.train()
        # Random batch for bSackprop training
        np.random.shuffle(batch_order)
        train_loss = 0.
        for j in range(nbatches):
            order = batch_order[j*batch:(j+1)*batch] # Need to conserve order
            xb, itb, isb, mb, tb, eb, lb = x_train[order], ie_to_train[order], ie_since_train[order], m_train[order],\
                                    t_train[order], e_train[order], l_train[order]
            
            # Reduce size data to all match batch
            lbmax = lb.max()
            xb, itb, isb, mb, tb = xb[:, :lbmax], itb[:, :lbmax], isb[:, :lbmax], mb[:, :lbmax], tb[:, :lbmax]

            if xb.shape[0] == 0:
                continue

            optimizer.zero_grad()
            loss_epoch, _ = model_torch.loss(xb, itb, isb, mb, eb, lb, tb, 
                        observational = observational) # Weights are not important as we aim to reduce all
            loss_epoch.backward()
            optimizer.step()
            train_loss += loss_epoch.item()
        train_loss /= nbatches

    model_torch.load_state_dict(best_weight)
    model_torch.eval()     
    return model_torch