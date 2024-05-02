from .grud import GRUD
from .rnn_ode import ODE
from ..utils import *
import torch.nn as nn
import torch

class RNN(BatchForward):
    """
        Torch implementation of a recurrent joint model 
        With LSTM shared representation

        Factory like object which aggregate the multiple submodules
    """
    
    def __init__(self, inputdim, typ = 'LSTM', layers = 1, hidden = 10, recurrent_args = {}, dropout = 0.):
        """
        Args:
            inputdim (int): Input dimension
            typ (str, optional): Recurrent cell type. Defaults to 'LSTM'.
            layers (int, optional): Number of reccurent layers. Defaults to 1.
            hidden (int, optional): Dimension hidden state. Defaults to 10.
            recurrent_args (dict, optional): Arguments for the model. Defaults to {}.
        """
        super(RNN, self).__init__()

        self.inputdim = inputdim

        # Recurrent model
        self.typ = typ
        self.layers = layers
        self.hidden = hidden

        self.time = False

        if typ == 'LSTM':
            self.embedding = nn.LSTM(self.inputdim, self.hidden, self.layers,
                                   bias=True, batch_first=True, dropout = dropout, **recurrent_args)

        elif typ == 'RNN':
            self.embedding = nn.RNN(self.inputdim, self.hidden, self.layers,
                                  bias=True, batch_first=True,
                                  nonlinearity='relu', dropout = dropout, **recurrent_args)

        elif typ == 'GRU':
            self.embedding = nn.GRU(self.inputdim, self.hidden, self.layers,
                                  bias=True, batch_first=True, dropout = dropout, **recurrent_args)

        elif typ == 'GRUD':
            self.embedding = GRUD(self.inputdim, self.hidden, self.layers,
                                  bias=True, batch_first=True, dropout = dropout, **recurrent_args)
            self.time = True

        elif typ == "ODE":
            self.embedding = ODE(self.inputdim, self.hidden, self.layers,
                                  bias=True, batch_first=True, **recurrent_args)
            self.time = True
        else:
            raise NotImplementedError()

    def pickle(self):
        if self.typ == 'ODE':
            self.embedding.pickle()

    def unpickle(self):
        if self.typ == 'ODE':
            self.embedding.unpickle()


    def forward_batch(self, x, t, m, l, total_l = None):
        """
            Forward through RNN
        """
        # To handle different size time series
        if self.time:
            hidden, = self.embedding(x, t, m, l) 
        else:
            pack = torch.nn.utils.rnn.pack_padded_sequence(x,
                                           l.cpu(),
                                           enforce_sorted=False,
                                           batch_first=True)
            hidden, _ = self.embedding(pack)
            hidden = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first = True, total_length = total_l)[0]

        return hidden,
