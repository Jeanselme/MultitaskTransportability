from ..utils import *
import torch.nn as nn
import torch

class Longitudinal():
    """
        Factory object
    """
    @staticmethod
    def create(longitudinal, inputdim, outputdim, longitudinal_args = {}): 
        if longitudinal == 'None':
            return None
        elif longitudinal == 'neural':
            return Neural(inputdim, outputdim, longitudinal_args)
        elif longitudinal == 'gaussian':
            return Gaussian(inputdim, outputdim, longitudinal_args)
        else:
            raise NotImplementedError()


class Neural(BatchForward):
    """
        Neural with Gaussian error
    """

    def __init__(self, inputdim, outputdim, longitudinal_args = {}):
        """
        Args:
            inputdim (int): Input dimension (hidden state)
            outputdim (int): Output dimension (original input dimension)
            longitudinal_args (dict, optional): Arguments for the model. Defaults to {}.
        """
        super(Neural, self).__init__()

        self.inputdim = inputdim
        self.outputdim = outputdim

        longitudinal_layer = longitudinal_args['layers'] if 'layers' in longitudinal_args else [100]
        self.mean_var = nn.Sequential(*create_nn(inputdim + 1, longitudinal_layer + [2 * outputdim])[:-1])
        self.softplus = nn.Softplus()

    def forward_batch(self, h, i, m, l):
        tau = i[:, 1:].unsqueeze(-1)
        h_time = h[:, :-1, :]  # Last point not observed
        concat = torch.cat((h_time, tau), 2)

        mean_var = self.mean_var(concat)
        mean, var = mean_var[:,:,:self.outputdim], self.softplus(mean_var[:,:,self.outputdim:])

        concat = torch.cat((h_time, torch.zeros_like(tau)), 2)
        mean_0 = self.mean_var(concat)[:,:,:self.outputdim]

        return mean - mean_0, var

    def loss(self, alpha, h, x, i, m, l, batch = None, reduction = 'mean'):
        mean, variance = self.forward(h, i, m, l, batch = batch)
        submask = m[:, 1:, :] # First value not even predicted for each time series
        diff = x[:, 1:, :] - x[:, :-1, :]
        loss = (alpha[submask] * nn.GaussianNLLLoss(reduction = "none", full = True)(mean[submask], diff[submask], variance[submask])).sum()

        if reduction == 'mean':
            loss /= submask.sum()

        return loss


class Gaussian(Neural):
    """
        Gaussian function of time
    """

    def __init__(self, inputdim, outputdim, longitudinal_args = {}):
        """
        Args:
            inputdim (int): Input dimension (hidden state)
            outputdim (int): Output dimension (original input dimension)
            longitudinal_args (dict, optional): Arguments for the model. Defaults to {}.
        """
        super(Gaussian, self).__init__(inputdim, outputdim, longitudinal_args)
        representation = longitudinal_args['representation'] if 'representation' in longitudinal_args else 50
        self.mean_var = nn.Linear(representation + 1, 2 * outputdim)
        self.latent = nn.Parameter(torch.randn(1, 1, representation))

    def forward_batch(self, h, i, m, l):
        tau = i[:, 1:].unsqueeze(-1)
        centroid = torch.cat((self.latent.repeat((len(h), tau.shape[1], 1)), tau), 2)
        mean_var = self.mean_var(centroid)
        mean, var = mean_var[:,:,:self.outputdim], self.softplus(mean_var[:,:,self.outputdim:])

        centroid = torch.cat((self.latent.repeat((len(h), tau.shape[1], 1)), torch.zeros_like(tau)), 2)
        mean_0 = self.mean_var(centroid)[:,:,:self.outputdim]

        return mean - mean_0, var