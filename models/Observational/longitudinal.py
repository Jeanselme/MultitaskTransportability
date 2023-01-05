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
        else:
            raise NotImplementedError()


class Neural(BatchForward):
    """
        Neural with Gaussian error
        Predict the change in value not the value iteself
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
        # Predict next step observation (shorten time)
        concat = torch.cat((h, i.abs().min(dim = 2)[0].unsqueeze(-1)), 2) # Note that should always be positive as there is one observation at least
        mean_var = self.mean_var(concat)
        mean, var = mean_var[:,:,:self.outputdim], self.softplus(mean_var[:,:,self.outputdim:])

        # Remove mean 0 as it predicts change in value
        concat = torch.cat((h, torch.zeros_like(i.min(dim = 2)[0].unsqueeze(-1))), 2)
        mean_0 = self.mean_var(concat)[:,:,:self.outputdim]

        return mean - mean_0, var

    def loss(self, alpha, h, x, i, m, l, batch = None, reduction = 'mean'):
        mean, variance = self.forward(h[:, :-1, :], i[:, :-1, :], m, l, batch = batch)
        submask = m[:, 1:, :] 
        diff = x[:, 1:, :] - x[:, :-1, :] # Compute change in value
        loss = (alpha * nn.GaussianNLLLoss(reduction = "none", full = True)(mean, diff, variance))

        if reduction == 'mean':
            loss = loss[submask].sum() / submask.sum()
        elif reduction == 'none':
            loss = - torch.sum(loss * submask, dim = 1) / submask.sum(dim = 1)

        return loss