import torch.nn as nn
import torch

from .utils import *
from .Observational import *

class Observational(BatchForward):

    def __init__(self, inputdim, outputdim, 
                temporal, temporal_args, 
                longitudinal, longitudinal_args, 
                missing, missing_args, dropout):
        """
        Args:
            inputdim (int): Input dimension (Embedded input)
            outputdim (int, optional): Output dimension (Input dimension)
            
            temporal (str, optional): Type of temporal modelling. (When will be the next values ?)
                Possible choices: "point", "weibull".
                Defaults to False.
            temporal_args (dict, optional): Arguments for the model. Defaults to {}.

            longitudinal (str, optional): Type of temporal modelling. (What will be the next values ?)
                Possible choices: "neural", "gaussian".
                Defaults to False.
            longitudinal_args (dict, optional): Arguments for the model. Defaults to {}.

            missing (str, optional): Type of mask modelling (What will be missing ?).
                Possible choices: "neural", "bernoulli".
                Defaults to False.
            missing_args (dict, optional): Arguments for the model. Defaults to {}.
        """
        super(Observational, self).__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim

        self.temporal = Temporal.create(temporal, inputdim, outputdim, temporal_args, dropout)
        self.longitudinal = Longitudinal.create(longitudinal, inputdim, outputdim, longitudinal_args, dropout)
        self.missing = Missing.create(missing, inputdim, outputdim, missing_args, dropout)

    def forward_batch(self, h, i, m, l):
        """
            Forward through the different networks
        """
        zeros = torch.zeros((h.shape[0], h.shape[1], self.outputdim), device = h.get_device() if h.is_cuda else 'cpu')
        temp_res = (self.temporal.forward_batch(h, i, m, l)[0]) if self.temporal is not None else zeros
        long_res = (self.longitudinal.forward_batch(h, i, m, l)[0]) if self.longitudinal is not None else zeros
        miss_res = (self.missing.forward_batch(h, i, m, l)[0]) if self.missing is not None else zeros

        return temp_res, long_res, miss_res

    def loss(self, h, x, i, m, l, batch = None, reduction = 'mean'):
        zeros = torch.tensor(0., device = x.get_device() if x.is_cuda else 'cpu') if reduction == 'mean' else \
            torch.zeros((h.shape[0], h.shape[1] - 1, self.outputdim), device = x.get_device() if x.is_cuda else 'cpu') # Compute individual likelihood
        loss_temp = self.temporal.loss(h, i, m, l, batch, reduction) if self.temporal is not None else zeros
        loss_long = self.longitudinal.loss(h, x, i, m, l, batch, reduction) if self.longitudinal is not None else zeros
        loss_miss = self.missing.loss(h, i, m, l, batch, reduction) if self.missing is not None else zeros

        return loss_temp, loss_long, loss_miss