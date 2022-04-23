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
    
    def __init__(self, inputdim, typ = 'LSTM', layers = 1, hidden = 10, recurrent_args = {}, cuda = False):
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
                                   bias=True, batch_first=True, **recurrent_args)

        elif typ == 'RNN':
            self.embedding = nn.RNN(self.inputdim, self.hidden, self.layers,
                                  bias=True, batch_first=True,
                                  nonlinearity='relu', **recurrent_args)

        elif typ == 'GRU':
            self.embedding = nn.GRU(self.inputdim, self.hidden, self.layers,
                                  bias=True, batch_first=True, **recurrent_args)

        elif typ == 'GRUD':
            self.embedding = GRUD(self.inputdim, self.hidden, self.layers,
                                  bias=True, batch_first=True, **recurrent_args)
            self.time = True

        elif typ == "ODE":
            self.embedding = ODE(self.inputdim, self.hidden, self.layers,
                                  bias=True, batch_first=True, **recurrent_args)
            self.time = True

        else:
            raise NotImplementedError()

        self.cuda = cuda


    def forward_batch(self, x, t, m, l):
        """
            Forward through RNN
        """
        # To handle different size time series
        if self.time:
            hidden, (hp, c) = self.embedding(x, t, m, l) 
            hp = get_last_observed(hidden, l - 1)
        else:
            pack = torch.nn.utils.rnn.pack_padded_sequence(x,
                                           l.cpu(),
                                           enforce_sorted=False,
                                           batch_first=True)
            if self.cuda:
                pack = pack.cuda()
            if self.typ == 'GRU':
                hidden, hp = self.embedding(pack)
            else:
                hidden, (hp, c) = self.embedding(pack)
            hp = hp[-1]
            hidden = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True, total_length = x.size(1))[0]
        return hp, hidden
