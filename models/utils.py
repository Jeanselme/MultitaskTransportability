import numpy as np
import pandas as pd
import torch.nn as nn
import torch

def pandas_to_list(x):
    """
        Split pandas dataframe with multi index into list of array
        (Allow to split into multiple patients)
    """
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

def torch_to_pandas(tensor, template, mask = None):
    """
        Reshape the tensor to match the original template

        Args:
            tensor (_type_): _description_
            template (_type_): _description_
    """
    result = []
    mask = [True] * len(template.columns) if mask is None else mask
    for i, patient in enumerate(template.index.unique(level = 0)):
        selection = template.index.get_level_values(0) == patient
        result.append(pd.DataFrame(tensor[i, :np.sum(selection) - 1].cpu().detach().numpy(), index = template.index[selection][:-1], columns = template.columns[mask]))
    return pd.concat(result, axis = 0)

def create_nn(inputdim, layers, layer_unit = nn.Linear, activation = 'ReLU', dropout = 0.):
    """
        Create a simple multi layer perceptron
    """
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
        if dropout > 0:
            modules.append(nn.Dropout(p = dropout))
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

    index = torch.arange(x.size(0)).to(device)
    return x[index, l]

def ones_like(x):
    return torch.ones((x.size(0), x.size(1), 1), requires_grad = True, device = x.get_device() if x.is_cuda else 'cpu')

def sort_given_t(*args, t):
    """
        Sort any parameter (tensor of same size than t) with regard to t
        Necessary for DeepSurv that assume all sorted for loss computation
    """
    t, order = t.sort(dim = 0, descending = True) # Wants data to be from unobserved to observed
    order = order.squeeze()
    return [arg[order] for arg in args] + [t]

def compute_dwa(previous, previous_2, T = 2):
    """
        Computes the weights given the two last loss 
        Following Dynamic Weighting Average
    """
    if previous_2 is None:
        return {}
    else:
        weights = (previous / (T * previous_2)).abs()
        weights = torch.nan_to_num(weights, nan = 0, posinf = 0, neginf = 0) # Null values remove
        weights = nn.Softmax(0)(weights)
        return {'observational': weights}

class PositiveLinear(nn.Module):
    """
        Constraint layer with positive weights for monotonic neural network
    """
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
    """
        Abstract object to simplify batching
    """

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

    def forward(self, *args, batch = None, **kwargs):
        return self.batch(self.forward_batch, *args, batch = batch, **kwargs)

    def predict(self, *args, batch = None, **kwargs):
        return self.batch(self.predict_batch, *args, batch = batch, **kwargs)[0]
