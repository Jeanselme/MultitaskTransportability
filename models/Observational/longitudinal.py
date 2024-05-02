from ..utils import *
import torch.nn as nn
import torch

class Longitudinal():
    """
        Factory object
    """
    @staticmethod
    def create(longitudinal, inputdim, outputdim, longitudinal_args = {}, dropout = 0.): 
        if longitudinal == 'None':
            return None
        elif longitudinal == 'neural':
            return Neural(inputdim, outputdim, longitudinal_args, dropout)
        else:
            raise NotImplementedError()


class Neural(BatchForward):
    """
        Neural with Gaussian error
        Predict the change in value not the value iteself
    """

    def __init__(self, inputdim, outputdim, longitudinal_args = {}, dropout = 0.):
        """
        Args:
            inputdim (int): Input dimension (hidden state)
            outputdim (int): Output dimension (original input dimension)
            longitudinal_args (dict, optional): Arguments for the model. Defaults to {}.
        """
        super(Neural, self).__init__()

        self.inputdim = inputdim
        self.outputdim = outputdim

        longitudinal_layer = longitudinal_args.get("layers", [50])
        self.mean = nn.Sequential(*create_nn(inputdim + 1, longitudinal_layer + [outputdim], dropout = dropout)[:-1])

    def forward_batch(self, h, i, m, l):
        # Predict next step observation (shorten time)
        time = i.abs().min(dim = 2)[0].clone().detach().unsqueeze(-1)
        concat = torch.cat((h, time), 2) # Note that should always be positive as there is one observation at least
        mean = self.mean(concat)

        return mean,

    def loss(self, h, x, i, m, l, batch = None, reduction = 'mean'):
        mean, = self.forward(h[:, :-1, :], i[:, :-1, :], m, l, batch = batch)
        loss = m[:, 1:, :] * nn.MSELoss(reduction = "none")(mean, x[:, 1:, :])

        if reduction == 'mean':
            loss = torch.mean(loss.sum(axis = [1, 2]) / m[:, 1:, :].sum(axis = [1, 2]))

        return loss