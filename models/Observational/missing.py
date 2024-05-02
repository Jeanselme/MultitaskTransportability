from ..utils import *
import torch.nn as nn
import torch

class Missing():
    """
        Factory object
    """
    @staticmethod
    def create(missing, inputdim, outputdim, missing_args = {}, dropout = 0.): 
        if missing == 'None':
            return None
        elif missing == 'neural':
            return Neural(inputdim, outputdim, missing_args, dropout)
        else:
            raise NotImplementedError()


class Neural(BatchForward):
    """
        Neural with BCE error
    """

    def __init__(self, inputdim, outputdim, missing_args = {}, dropout = 0.):
        """
        Args:
            inputdim (int): Input dimension (hidden state)
            outputdim (int): Output dimension (original input dimension)
            missing_args (dict, optional): Arguments for the model. Defaults to {}.
        """
        super(Neural, self).__init__()

        self.inputdim = inputdim
        self.outputdim = outputdim

        missing_layer = missing_args.get("layers", [50])
        self.missing = nn.Sequential(*create_nn(inputdim + 1, missing_layer + [outputdim], dropout = dropout)[:-1], nn.Sigmoid()) # Time might be informative : + 1

    def forward_batch(self, h, i, m, l):
        # Predict next step observation (shorten time)
        time = i.abs().min(dim = 2)[0].clone().detach().unsqueeze(-1)
        concat = torch.cat((h, time), 2)
        missing = self.missing(concat)
        return missing,

    def loss(self, h, i, m, l, batch = None, reduction = 'mean'):
        predictions, = self.forward(h[:, :-1, :], i[:, :-1, :], m, l, batch = batch)
        # Compare what is predicted for the next step to what is observed next
        submask = (m[:, 1:, :].sum(dim = 2) > 0).unsqueeze(2)  # Only try to predict missingness mask if there is something observed next
        loss = submask * nn.BCELoss(reduction = "none")(predictions, m[:, 1:, :].double())

        if reduction == 'mean':
            loss = torch.mean(loss.sum(axis = [1, 2]) / submask.sum(axis = [1, 2]))
            
        return loss