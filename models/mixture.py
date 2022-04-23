import torch.nn as nn
import torch

from .utils import *
from .Observational import *

class Mixture(BatchForward):
    """
        Mixture of components for the observational process
    """

    def __init__(self, k, inputdim, outputdim, 
                temporal, temporal_args, 
                longitudinal, longitudinal_args, 
                missing, missing_args):
        """
        Args:
            k (int): Number of components for mixture of observational processes
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
        super(Mixture, self).__init__()
        self.k = k
        self.inputdim = inputdim
        self.outputdim = outputdim

        # Weights for each mixture 
        if self.k > 1:
            self.alphas = nn.Sequential(nn.Linear(inputdim, k), nn.Softmax(dim = 2))
        else:
            self.alphas = torch.ones_like

        self.temporal = nn.ModuleList([Temporal.create(temporal, inputdim, 1, temporal_args) for _ in range(k)])
        self.longitudinal = nn.ModuleList([Longitudinal.create(longitudinal, inputdim, outputdim, longitudinal_args) for _ in range(k)])
        self.missing = nn.ModuleList([Missing.create(missing, inputdim, outputdim, missing_args) for _ in range(k)])

    def forward_batch(self, h, i, m, l):
        """
            Forward through the different networks
        """
        temp_res, long_res, miss_res =  0, 0, 0
        alphas = self.alphas(h[:, :-1])
        for j, (temp, long, miss) in enumerate(zip(self.temporal, self.longitudinal, self.missing)):
            alphas_repeat = alphas[:, :, j].unsqueeze(2).repeat(1, 1, self.outputdim)
            temp_res += (alphas[:, :, j] * temp.forward_batch(h, i, m, l)[0]) if temp is not None else 0
            long_res += (alphas_repeat * long.forward_batch(h, i, m, l)[0]) if long is not None else 0
            miss_res += (alphas_repeat * miss.forward_batch(h, i, m, l)[0]) if miss is not None else 0

        return temp_res, long_res, miss_res, alphas

    def loss(self, h, x, i, m, l, batch = None, reduction = 'mean'):
        loss_temp = loss_long = loss_miss = 0
        alphas = self.alphas(h[:, :-1])
        for j, (temp, long, miss) in enumerate(zip(self.temporal, self.longitudinal, self.missing)):
            # Elbo loss (alpha could be computed exactly)
            alphas_repeat = alphas[:, :, j].unsqueeze(2).repeat(1, 1, x.size(2))
            loss_temp += temp.loss(alphas[:, :, j], h, i, m, l, batch, reduction) if temp is not None else 0
            loss_long += long.loss(alphas_repeat, h, x, i, m, l, batch, reduction) if long is not None else 0
            loss_miss += miss.loss(alphas_repeat, h, i, m, l, batch, reduction) if miss is not None else 0

        return loss_temp, loss_long, loss_miss