import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.modules.rnn import RNNCellBase

class GRUDCell(RNNCellBase):
    """
        Cell with decay between time points
    """

    def __init__(self, input_size, input_size_for_decay, hidden_size, bias = True, dropout = 0):
        super(GRUDCell, self).__init__(input_size, hidden_size, bias, num_chunks = 3)

        self.input_size_for_decay = input_size_for_decay
        self.decay = nn.Sequential(nn.Linear(input_size_for_decay, hidden_size))

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def gru_exp_decay_cell(self, x, time, hidden, w_ih, w_hh, b_ih, b_hh):
        """
        Args:
            x (Tensor): New data
            time (Tensor): Time difference since last observation
            hidden (Tensor): Hidden state from previous time
            w_ih ([type]): [description]
            w_hh ([type]): [description]
            b_ih ([type]): [description]
            b_hh ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Compute decay (limit values)
        decay = torch.exp( - torch.min(
                                torch.max(
                                    torch.zeros(size=(self.hidden_size,), device = x.get_device() if x.is_cuda else 'cpu'),
                                    self.decay(time)
                                ), 
                                torch.ones(size=(self.hidden_size,), device = x.get_device() if x.is_cuda else 'cpu') * 1000 
                            )
                         )

        # Decayed hidden state
        hidden = hidden * decay

        # Doors 
        gi = torch.mm(x, w_ih.t())
        gh = torch.mm(hidden, w_hh.t())
        
        if self.bias:
            gi += b_ih
            gh += b_hh
        
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy if self.dropout is None else self.dropout(hy)

    def forward(self, x, t, hx = None):
        if hx is None:
            hx = torch.zeros(x.size(0), self.hidden_size, 
                dtype=x.dtype, device = x.get_device() if x.is_cuda else 'cpu')
                
        return self.gru_exp_decay_cell(
            x, t, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh
        )

class GRUD(nn.Module):
    
    def __init__(self, inputdim, hidden, layers, bias=True, batch_first=True, imputation=False, dropout = 0):
        super(GRUD, self).__init__()
        self.cell = GRUDCell(inputdim, 1, hidden, bias, dropout) # Only time since last (put inputdim if all modelled)
        self.num_layers = layers
        self.hidden_size = hidden
        self.batch_first = batch_first
        self.imputation = imputation

    def forward(self, input, times, mask, length, hx = None):
        # Does not deal with Packed as it seems they have strange behavior
        orig_input = input
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        max_time = int(length.max().item())

        if self.imputation:
            # Impute missing data
            pass

        if hx is None: # Only one direction as time difference is given
            hx = torch.zeros(max_batch_size, self.hidden_size,
                             dtype=input.dtype, device = input.get_device() if input.is_cuda else 'cpu')

        outputs = None
        for i in range(max_time):           
            # TODO: adapt to do batch_first = False and to have same format than GRU (with packed)
            hx = self.cell.forward(torch.squeeze(input[:,i:i+1,:]),
                    times[:,i:i+1],
                    hx)

            if outputs is None:
                outputs = hx.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, hx.unsqueeze(1)), 1)

        empty = torch.zeros(max_batch_size, 1, self.hidden_size, dtype=input.dtype, device=input.get_device() if input.is_cuda else 'cpu')
        for i in range(max_time, input.size(1)): 
            outputs = torch.cat((outputs, empty), 1)
        
        # Remove last steps that are unecessary
        end = outputs[0, length[0].long() - 1, :].unsqueeze(0)
        for i in range(1, input.size()[0]):
            end = torch.cat((end, outputs[i, length[i].long() - 1, :].unsqueeze(0)), 0)

        return outputs, (end, None)