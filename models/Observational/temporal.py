from ..utils import *
from torch.autograd import grad
import torch.nn as nn
import torch

class Temporal():
    """
        Factory object
    """
    @staticmethod
    def create(temporal, inputdim, outputdim, temporal_args = {}): 
        if temporal == 'None':
            return None
        elif temporal == 'point':
            return Point(inputdim, outputdim, temporal_args)
        else:
            raise NotImplementedError()

class Point(BatchForward):

    def __init__(self, inputdim, outputdim, temporal_args = {}):
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

        temporal_layer = temporal_args['layers'] if 'layers' in temporal_args else [100]
        self.cumulative = nn.Sequential(*create_nn(inputdim + 1, temporal_layer + [outputdim], PositiveLinear, 'Tanh')[:-1], nn.Softplus())

    def forward_batch(self, h, i, m, l):
        # Flatten the temporal and patient dimensions -> Keep number of covariates
        tau = torch.flatten(i[:, :-1, :].clone().detach().requires_grad_(True), 0, 1)
        hidden_tau = torch.flatten(h[:, :-1, :].clone().detach().requires_grad_(True), 0, 1)
        
        cumulative = []
        # Need to iterate to avoid that the time of the other prediciton impacts the predictions
        # Select the dim for which we predict
        for dim in range(self.outputdim):
            cumulative_dim = self.cumulative(torch.cat((hidden_tau, tau[:, dim].unsqueeze(-1)), 1))
            cumulative.append(cumulative_dim[:, dim].unsqueeze(-1))

        cumulative = torch.cat(cumulative, axis = -1)
        gradient = grad(torch.sum(cumulative), tau, create_graph=True)[0] # Verified 
        if h.is_cuda:
            gradient = gradient.cuda()

        return torch.exp(- cumulative).reshape([h.shape[0], h.shape[1] - 1, self.outputdim]), \
            gradient.reshape([h.shape[0], h.shape[1] - 1, self.outputdim]), \
            cumulative.reshape([h.shape[0], h.shape[1] - 1, self.outputdim])

    def loss(self, alpha, h, i, m, l, batch = None, reduction = 'mean'):
        _, gradient, cumulative = self.forward(h, i, m, l, batch = batch)
        mask = (i[:, :-1, :] >= 0) # Predict at all time when will be observed next 
        loss = - torch.sum((alpha * (torch.log(gradient + 1e-8) - cumulative))[mask])

        # Compute difference prediction and real time

        if reduction == 'mean':
            loss /= mask.sum()

        return loss
        