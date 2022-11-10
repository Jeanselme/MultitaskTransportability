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
        elif temporal == 'weibull':
            return Weibull(inputdim, outputdim, temporal_args)
        else:
            raise NotImplementedError()


class Weibull(BatchForward):
    """
        Weibull distribution
    """

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
        super(Weibull, self).__init__()
        self.outputdim = outputdim

        # Only one Weibull
        self.shape = nn.Parameter(-torch.randn(self.outputdim)) # k = exp(shape) 
        self.scale = nn.Parameter(-torch.randn(self.outputdim)) # b = exp(scale)

    def forward_batch(self, h, i, m, l):
        sh, sc = self.shape, self.scale
        i_used = i[:, 1:, :].flatten(0, 1) + 1e-8
        survival_log = - torch.pow(torch.exp(sc) * i_used, torch.exp(sh))
        density_log = sc + sh + (torch.exp(sh) - 1) * (torch.log(i_used) + sc) \
                + survival_log

        # TODO: Check if reshape does what you want
        return torch.exp(survival_log).reshape([h.shape[0], h.shape[1] - 1, self.outputdim]), density_log.reshape([h.shape[0], h.shape[1] - 1, self.outputdim])

    def loss(self, alpha, h, i, m, l, batch = None, reduction = 'mean'):
        _, density = self.forward(h, i, m, l, batch = batch)
        observed = m[:, 1:, :]
        loss = - ((alpha * density)[observed]).sum() 

        if reduction == 'mean':
            loss /= torch.sum(l - 1)
        return loss


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
        tau = i[:, 1:, :]
        hidden_tau = h[:, :-1, :]  # Last point not observed

        # Flatten the temporal and patient dimensions -> Keep number of covariates
        tau = torch.flatten(tau, 0, 1) 
        hidden_tau = torch.flatten(hidden_tau, 0, 1)

        tau.requires_grad = True # For autograd
        cumulative = []
        # Need to iterate to avoid that the time of the other prediciton impacts the predictions
        # Select the dim for which we predict
        for dim in range(self.outputdim):
            cumulative_dim = self.cumulative(torch.cat((hidden_tau, tau[:, dim].unsqueeze(-1)), 1)) - self.cumulative(torch.cat((hidden_tau, torch.zeros_like(tau[:, dim].unsqueeze(-1))), 1))
            cumulative_dim = cumulative_dim[:, dim]
            cumulative_dim = cumulative_dim.reshape([h.shape[0], h.shape[1] - 1, 1])
            cumulative.append(cumulative_dim)
        cumulative = torch.cat(cumulative, axis = -1)

        gradient = grad(torch.sum(cumulative), tau, create_graph=True, retain_graph=True)[0].reshape([h.shape[0], h.shape[1] - 1, self.outputdim])

        if h.is_cuda:
            gradient = gradient.cuda()

        survival = torch.exp(-cumulative)
        return survival, gradient, cumulative

    def loss(self, alpha, h, i, m, l, batch = None, reduction = 'mean'):
        _, gradient, cumulative = self.forward(h, i, m, l, batch = batch)
        observed = m[:, 1:, :]

        with torch.no_grad():
            gradient.clamp_(min = 1e-8)

        # TODO: exact loss, not ELBO
        loss = torch.add(torch.sum((alpha * cumulative)[observed]),
                        -torch.sum((alpha * torch.log(gradient))[observed]))

        if reduction == 'mean':
            loss /= torch.sum(l - 1)

        return loss
        