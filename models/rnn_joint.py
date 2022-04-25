from .rnn_joint_torch import RNNJointTorch
from .utils import sort_given_t, pandas_to_list, compute_dwa
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
        
    def fit(self, x_train, i_train, m_train, e_train, t_train, 
             x_valid = None, i_valid = None, m_valid = None, e_valid = None, t_valid = None, **params):
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
        x_train, i_train, m_train, e_train, l_train, t_train = self.preprocess(x_train, i_train, m_train, e_train, t_train)
        x_valid, i_valid, m_valid, e_valid, l_valid, t_valid = self.preprocess(x_valid, i_valid, m_valid, e_valid, t_valid)

        self.model = train_torch_model(self.model,
            x_train, i_train, m_train, e_train, l_train, t_train, 
            x_valid, i_valid, m_valid, e_valid, l_valid, t_valid, **params).eval()

        if self.model:
            self.model.compute_baseline(x_train, i_train, m_train, e_train, l_train, t_train, batch = 100)
            self.fitted = True
            return self
        else:
            return None
            
    def predict(self, x, i, m, horizon = None, risk = 1, batch = None):
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
        x, i, m, _, l, _ = self.preprocess(x, i, m)
        return self.model.predict(x, i, m, l, horizon = horizon, risk  = risk, batch = batch).detach().cpu().numpy()

    def observational_predict(self, x, i, m, batch = None):
        if not self.fitted:
            raise Exception("The model has not been fitted yet.")
        x, i, m, _, l, _ = self.preprocess(x, i, m)
        return [out.detach().cpu().numpy() for out in self.model.observational_predict(x, i, m, l, batch = batch)]

    def loss(self, x, i, m, e, t, batch = None):
        if not self.fitted:
            raise Exception("The model has not been fitted yet.")
        x, i, m, e, l, t = self.preprocess(x, i, m, e, t)
        x, i, m, e, l, t = sort_given_t(x, i, m, e, l, t = t)
        return self.model.loss(x, i, m, e, l, t, batch, observational = False)[0].item() # Only survival loss

    def loss_observational(self, x, i, m, batch = None):
        if not self.fitted:
            raise Exception("The model has not been fitted yet.")
        x, i, m, _, l, _ = self.preprocess(x, i, m)
        return {name: i.item() for name, i in zip(['Temporal', 'Longitudinal', 'Missing'], self.model.loss(x, i, m, None, None, l, batch, survival = False, observational = True)[1]['observational'])}

    def feature_importance(self, x, i, m, e, t, n = 100, batch = None):
        if not self.fitted:
            raise Exception("The model has not been fitted yet.")
        x_p, i_p, m_p, e_p, l_p, t_p = self.preprocess(x, i, m, e, t)
        x_p, i_p, m_p, e_p, l_p, t_p = sort_given_t(x_p, i_p, m_p, e_p, l_p, t = t_p)
        global_nll = self.model.loss(x_p, i_p, m_p, e_p, l_p, t_p, batch)[1]
        if 'observational' in global_nll:
            global_nll =  {'Survival': global_nll['survival'].item(), 
                           'Temporal': global_nll['observational'][0].item(), 
                           'Longitudinal': global_nll['observational'][1].item(), 
                           'Missing': global_nll['observational'][2].item()}
        else:
            global_nll =  {'Survival': global_nll['survival'].item()}

        performances = {c: {j: [] for j in range(x.shape[1])} for c in global_nll}
        for _ in tqdm(range(n)):
            rool = np.random.randint(x.shape[0])
            for j in range(x.shape[1]):
                # Permute all values of one feature (between patients and across time)
                x_p = x.values.copy()
                x_p[:, j] = np.roll(x_p[:, j], rool)
                x_p = pd.DataFrame(x_p, index = x.index)
                
                x_p, i_p, m_p, e_p, l_p, t_p = self.preprocess(x, i, m, e, t)
                x_p, i_p, m_p, e_p, l_p, t_p = sort_given_t(x_p, i_p, m_p, e_p, l_p, t = t_p)

                nll = self.model.loss(x_p, i_p, m_p, e_p, l_p, t_p, batch)[1]
                performances['Survival'][j].append(nll['survival'].item())

                if 'observational' in nll:
                    performances['Temporal'][j].append(nll['observational'][0].item())
                    performances['Longitudinal'][j].append(nll['observational'][1].item())
                    performances['Missing'][j].append(nll['observational'][2].item())

        return {c: {j: (np.array(performances[c][j]) - global_nll[c]) / global_nll[c] for j in performances[c]} for c in performances}
          

    def preprocess(self, x, i, m, e = None, t = None):
        """
        Preprocess data
            All lists need to have the same size
            
        Returns:
            6 Tensors: Padded Data, Padded Mask, Padded Interevent Time, Time to event, Target, Length
        """
        if x is None: 
            return None, None, None, None, None, None

        x = pandas_to_list(x)
        i = pandas_to_list(i)
        m = pandas_to_list(m)

        # X and T Padding
        xres, ires, mres, l = [], [], [], [len(xi) for xi in x]
        max_length = max(l)
        for xi, ii, mi, li in zip(x, i, m, l):
            # Compute time differences between successive points
            xres.append(np.concatenate([xi, np.zeros(shape = (max_length - li, xi.shape[1]))]))
            ires.append(np.concatenate([ii, np.zeros(shape = (max_length - li))]))
            mres.append(np.concatenate([mi, np.zeros(shape = (max_length - li, mi.shape[1]))]))
        x = torch.from_numpy(np.array(xres, dtype=float)).double()
        i = torch.from_numpy(np.array(ires, dtype=float)).double()
        m = torch.from_numpy(np.array(mres, dtype=float)) > 0.5
        l = torch.LongTensor(l)

        if e is not None: 
            e = e.values if (isinstance(e, pd.DataFrame) or isinstance(e, pd.Series)) else e
            e = torch.DoubleTensor(e.copy()).unsqueeze(-1)
            if self.cuda:
                e = e.cuda()

        if t is not None:
            t = pandas_to_list(t)
            t = torch.from_numpy(np.array([ti[li - 1] for ti, li in zip(t, l)])).double().unsqueeze(-1)

            if self.cuda:
                t = t.cuda()

        if self.cuda:
            x, i, m, l = x.cuda(), i.cuda(), m.cuda(), l.cuda()
            
        return x, i, m, e, l, t

def train_torch_model(model_torch, 
    x_train, i_train, m_train, e_train, l_train, t_train,
    x_valid, i_valid, m_valid, e_valid, l_valid, t_valid,
    epochs = 500, pretrain_ite = 500, lr = 0.0001, batch = 500, patience = 5, weight_decay = 0.001):

    # Initialization parameters
    weights = {}
    full = True
    t_bar = tqdm(range(epochs + pretrain_ite))
    
    nbatches = int(x_train.shape[0] / batch) + 1 # Number batch
    batch_order = np.arange(x_train.shape[0]) # Index of all data in training
    previous_loss, best_loss = np.inf, np.inf # Keep track of losses
    best_weight = deepcopy(model_torch.state_dict()) # Keep best parameters

    previous_losses, previous_losses_2 = {}, {} # Observational weighting (different losses are weighted differently)
    
    optimizer = torch.optim.Adam(model_torch.parameters(), lr = lr, weight_decay = weight_decay)

    # Sort batch for likelihood computation
    x_train, i_train, m_train, e_train, l_train, t_train = sort_given_t(x_train, i_train, m_train, e_train, l_train, t = t_train)
    if x_valid is not None:
        x_valid, i_valid, m_valid, e_valid, l_valid, t_valid = sort_given_t(x_valid, i_valid, m_valid, e_valid, l_valid, t = t_valid)

    for i in t_bar:
        if i == pretrain_ite:
            # End pretraining 
            ## Upload best weights and reinitalize losses
            full = False
            wait = 0
            model_torch.load_state_dict(best_weight)
            optimizer = torch.optim.Adam(model_torch.survival_model.parameters(), lr = lr, weight_decay = weight_decay)
        elif full:
            weights = compute_dwa(previous_losses, previous_losses_2)

        model_torch.train()
        # Random batch for backprop training
        np.random.shuffle(batch_order)
        for j in range(nbatches):
            order = np.sort(batch_order[j*batch:(j+1)*batch]) # Need to conserve order
            xb, ib, mb, tb, eb, lb = x_train[order], i_train[order], m_train[order],\
                                     t_train[order], e_train[order], l_train[order]

            if xb.shape[0] == 0:
                continue

            optimizer.zero_grad()
            loss, _ = model_torch.loss(xb, ib, mb, eb, lb, tb, 
                        observational = full, weights = weights)
            loss.backward()
            optimizer.step()
        
        # Evaluate validation loss - Batch
        if x_valid is None:
            best_weight = deepcopy(model_torch.state_dict())
            continue
        
        model_torch.eval()
        previous_losses_2 = previous_losses.copy()
        loss, previous_losses = model_torch.loss(x_valid, i_valid,
                                m_valid, e_valid, l_valid, t_valid, 
                                batch = batch, observational = full)
        
        if full:
            t_bar.set_description("Loss full: {:.3f} - {:.3f}".format(loss.item(), previous_losses['survival'].item()))
        else:
            t_bar.set_description("Loss survival: {:.3f}".format(loss.item()))
        survival_loss = previous_losses['survival'].item()
        
        if np.isnan(survival_loss):
            print('ERROR - Loss')
            return None

        if survival_loss > previous_loss:
            # If less good than before
            if full and (wait == patience):
                pretrain_ite = i + 1
            elif wait == patience:
                break
            else:
                wait += 1
        elif survival_loss < best_loss:
            # Update new best
            best_weight = deepcopy(model_torch.state_dict())
            best_loss = survival_loss
            wait = 0
        else:
            wait = 0
        
        previous_loss = survival_loss
    
    model_torch.load_state_dict(best_weight)            
    return model_torch