from ..utils import *
from torch.autograd import grad
import torch.nn as nn
import torch

class Temporal():
    """
        Factory object
    """
    @staticmethod
    def create(temporal, inputdim, outputdim, temporal_args = {}, dropout = 0.): 
        if temporal == 'None':
            return None
        elif temporal == 'point':
            return Point(inputdim, outputdim, temporal_args, dropout)
        elif temporal == 'single':
            return PointSingle(inputdim, 1, temporal_args, dropout)
        else:
            raise NotImplementedError()

class Point(BatchForward):

    def __init__(self, inputdim, outputdim, temporal_args = {}, dropout = 0.):
        """
        Args:
            temporal (str, optional): Type of temporal modelling. (When will be the next values ?)
                Possible choices: "point", "mixture".
                Defaults to False.
            inputdim (int): Input dimension (hidden state)
            outputdim (int): Output dimension (original data size)
            temporal_args (dict, optional): Arguments for the model. Defaults to {}.
        """
        super(Point, self).__init__()

        self.inputdim = inputdim
        self.outputdim = outputdim

        temporal_layer = temporal_args.get("layers", [100])
        self.cumulative = nn.Sequential(*create_nn(inputdim + 1, temporal_layer + [outputdim], PositiveLinear, 'Tanh', dropout = dropout)[:-1], nn.Softplus())

    def forward_batch(self, h, i, m, l):
        # Flatten the temporal and patient dimensions -> Keep number of covariates (except for tau)
        tau = torch.flatten(i.abs().min(dim = 2)[0].unsqueeze(-1).clone().detach().requires_grad_(True), 0, 1)
        hidden_tau = torch.flatten(h, 0, 1)
        
        cumulative = self.cumulative(torch.cat((hidden_tau, tau), 1))
        
        gradient = []
        for dim in range(self.outputdim):
          gradient.append(grad(cumulative[:, dim].sum(), tau, create_graph = True)[0].unsqueeze(1))
        gradient = torch.cat(gradient, -1).unsqueeze(-1)

        if h.is_cuda:
            gradient = gradient.cuda()

        return torch.exp(- cumulative).reshape([h.shape[0], h.shape[1], self.outputdim]), \
            gradient.reshape([h.shape[0], h.shape[1], self.outputdim]), \
            cumulative.reshape([h.shape[0], h.shape[1], self.outputdim])

    def loss(self, h, i, m, l, batch = None, reduction = 'mean'):
        _, gradient, cumulative = self.forward(h[:, :-1, :], i[:, :-1, :], m, l, batch = batch)
        submask = i[:, :-1, :] > 0 # -1 because you select the prediction to the next 
        nll = torch.zeros_like(i[:, :-1, :])
        nll[m[:, 1:, :]] = (cumulative - torch.log(gradient.clamp_(1e-10)))[m[:, 1:, :]] # 1 because you measure if next is well predicted
        nll[~m[:, 1:, :]] = cumulative[~m[:, 1:, :]]

        loss = nll * submask

        # Compute difference prediction and real time
        if reduction == 'mean':
            loss = torch.mean(loss.sum(axis = [1, 2]) / submask.sum(axis = [1, 2]))

        return loss
        
class PointSingle(Point):
    # Does not aim to predict each time series but only the time to the next lab test 
    # (as the other dimension of the point process should work)

    def forward_batch(self, h, i, m, l):
        # Flatten the temporal and patient dimensions -> Keep number of covariates (except for tau)
        tau = torch.flatten(i.abs().min(axis = 2)[0], 0, 1).clone().detach().requires_grad_(True).unsqueeze(-1)
        hidden_tau = torch.flatten(h, 0, 1)
        
        cumulative = self.cumulative(torch.cat((hidden_tau, tau), 1))
        gradient = grad(cumulative.sum(), tau, create_graph = True)[0].unsqueeze(1)

        if h.is_cuda:
            gradient = gradient.cuda()

        return torch.exp(- cumulative).reshape([h.shape[0], h.shape[1], self.outputdim]), \
            gradient.reshape([h.shape[0], h.shape[1], self.outputdim]), \
            cumulative.reshape([h.shape[0], h.shape[1], self.outputdim])

    def loss(self, h, i, m, l, batch = None, reduction = 'mean'):
        _, gradient, cumulative = self.forward(h[:, :-1, :], i[:, :-1, :], m, l, batch = batch)
        submask = (m[:, 1:, :].sum(dim = 2) > 0).unsqueeze(2) # If no value is observed - we should not predict (that means we are after the end of observation))
        loss = cumulative - torch.log(gradient.clamp_(1e-10)) # 1 because you measure if next is well predicted / alpha is repeated so ok to take 1
        loss = loss * submask

        # Compute difference prediction and real time
        if reduction == 'mean':
            loss = torch.mean(loss.sum(axis = [1, 2]) / submask.sum(axis = [1, 2]))
        else:
            loss = loss.repeat((1, 1, i.size(2)))

        return loss
        