import numpy as np
import pandas as pd
import torch.nn as nn
import torch

def pandas_to_list(x):
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        result = []
        for patient in x.index.unique(level = 0):
            selection = x.index.get_level_values(0) == patient
            result.append(x[selection].values)
        return result
    elif isinstance(x, list):
        return x
    else:
        print(x)
        raise ValueError("Data not in the right format")

def create_nn(inputdim, layers, layer_unit = nn.Linear, activation = 'ReLU'):
    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()
    elif activation == 'Tanh':
        act = nn.Tanh()

    modules = []
    prevdim = inputdim

    for hidden in layers:
        modules.append(layer_unit(prevdim, hidden, bias = True))
        modules.append(act)
        prevdim = hidden

    return modules

def get_last_observed(x, l):
    """
        Extracts from xi the li observations
    """
    if x.is_cuda:
        device = x.get_device()
    else:
        device = torch.device("cpu")

    index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device)
    last = index == l.unsqueeze(1).repeat(1, x.size(1))

    return x[last]

def ones_like(x):
    return torch.ones((x.size(0), x.size(1), 1), requires_grad = True, device = x.get_device() if x.is_cuda else 'cpu')

def sort_given_t(*args, t):
    t, order = torch.sort(t, 0, descending = True) # Wants data to be from unobserved to observed
    order = order.squeeze()
    return [arg[order] for arg in args] + [t]

def compute_dwa(previous, previous_2, T = 2):
    """
        Computes the weights given the two last loss 
        Following Dynamic Weighting Average
    """
    if 'observational' not in previous_2 or 'observational' not in previous:
        return {}
    else:
        weights = nn.Softmax(0)(previous['observational'].detach() / (T * previous_2['observational'].detach()))
        return {'observational': weights}

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = False):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.log_weight)
            bound = np.sqrt(1 / np.sqrt(fan_in))
            nn.init.uniform_(self.bias, -bound, bound)
        self.log_weight.data.abs_().sqrt_()

    def forward(self, input):
        if self.bias is not None:
            return nn.functional.linear(input, self.log_weight ** 2, self.bias)
        else:
            return nn.functional.linear(input, self.log_weight ** 2)

class BatchForward(nn.Module):

    def forward_batch(self, *args):
        raise NotImplementedError()

    def predict_batch(self, *args):
        raise NotImplementedError()

    def batch(self, func, *args, batch = None, **kwargs):
        if batch is None:
            batch = args[0].shape[0]

        results = {}
        batches = int(args[0].shape[0] / batch) + 1

        for j in range(batches):
            args_b = [arg[j*batch:(j+1)*batch] for arg in args]
            if args_b[0].shape[0] == 0:
                continue
            
            forwardb = func(*args_b, **kwargs)
            for k, output in enumerate(forwardb):
                if k in results:
                    results[k].append(output)
                else:
                    results[k] = [output]

        for k in results:
            results[k] = torch.cat(results[k], 0)

        return [results[k] for k in results]

    def forward(self, *args, batch = None):
        return self.batch(self.forward_batch, *args, batch = batch)

    def predict(self, *args, batch = None, **kwargs):
        return self.batch(self.predict_batch, *args, batch = batch, **kwargs)[0]
