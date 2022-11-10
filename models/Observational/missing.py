from ..utils import *
import torch.nn as nn
import torch

class Missing():
    """
        Factory object
    """
    @staticmethod
    def create(missing, inputdim, outputdim, missing_args = {}): 
        if missing == 'None':
            return None
        elif missing == 'neural':
            return Neural(inputdim, outputdim, missing_args)
        elif missing == 'bernoulli':
            return Bernoulli(inputdim, outputdim, missing_args)
        else:
            raise NotImplementedError()


class Neural(BatchForward):
    """
        Neural with BCE error
    """

    def __init__(self, inputdim, outputdim, missing_args = {}):
        """
        Args:
            inputdim (int): Input dimension (hidden state)
            outputdim (int): Output dimension (original input dimension)
            missing_args (dict, optional): Arguments for the model. Defaults to {}.
        """
        super(Neural, self).__init__()

        self.inputdim = inputdim
        self.outputdim = outputdim

        missing_layer = missing_args['layers'] if 'layers' in missing_args else [100]
        self.missing = nn.Sequential(*create_nn(inputdim + 1, missing_layer + [outputdim])[:-1], nn.Sigmoid()) # Time might be informative : + 1

    def forward_batch(self, h, i, m, l):
        tau = i[:, 1:, :]
        h_time = h[:, :-1, :]  # Last point not observed

        missing = []
        for dim in range(self.outputdim):
            concat = torch.cat((h_time, tau[:,:, dim].unsqueeze(-1)), 2)
            missing_dim = self.missing(concat)[:,:, dim]
            missing.append(missing_dim.unsqueeze(-1))
        missing = torch.cat(missing, axis = -1)
        return missing,

    def loss(self, alpha, h, i, m, l, batch = None, reduction = 'mean'):
        predictions, = self.forward(h, i, m, l, batch = batch)
        submask = m[:, 1:, :] # First value not even predicted for each time series
        loss = (alpha.flatten() * nn.BCELoss(reduction = "none")(predictions.flatten(), submask.flatten().double())).sum()

        if reduction == 'mean':
            loss /= submask.sum()

        return loss

class Bernoulli(Neural):
    """
        Bernoulli function of time
    """

    def __init__(self, inputdim, outputdim, missing_args = {}):
        """
        Args:
            inputdim (int): Input dimension (hidden state)
            outputdim (int): Output dimension (original input dimension)
            missing_args (dict, optional): Arguments for the model. Defaults to {}.
        """
        super(Bernoulli, self).__init__(inputdim, outputdim, missing_args)
        representation = missing_args['representation'] if 'representation' in missing_args else 50
        self.latent = nn.Parameter(torch.randn(1, 1, representation))
        self.missing = nn.Sequential(nn.Linear(representation + 1, outputdim), nn.Sigmoid())

    def forward_batch(self, h, i, m, l):
        tau = i[:, 1:, :]       
        missing = []
        for dim in range(self.outputdim):
            centroid = torch.cat((self.latent.repeat((len(h), tau.shape[1], 1)), tau[:,:, dim].unsqueeze(-1)), 2)
            missing_dim = self.missing(centroid)[:,:, dim]
            missing.append(missing_dim.unsqueeze(-1))
        missing = torch.cat(missing, axis = -1)
        
        return missing,