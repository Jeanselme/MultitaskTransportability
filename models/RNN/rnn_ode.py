import torch
import torch.nn as nn
from .grud import *
class ODESolver(nn.Module):
    """
        From https://github.com/YuliaRubanova/latent_ode
    """

    def __init__(self, method, input_dimension, ode_func = None, odeint_rtol = 1e-4,
        odeint_atol = 1e-5, step_size = 0.1):
        super(ODESolver, self).__init__()
        self.ode_func = ode_func
        if self.ode_func is None:
            self.ode_func = nn.Sequential(
                                nn.Linear(input_dimension, 50), nn.Tanh(),
                                nn.Linear(50, 50), nn.Tanh(),
                                nn.Linear(50, input_dimension), nn.Tanh())
            self.ode_func.forward_y = self.ode_func.forward
            self.ode_func.forward = lambda t, x: self.ode_func.forward_y(x)

        self.method = method
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

        self.step_size = step_size

    def forward(self, x, t):
        """
        # Decode the trajectory through ODE Solver
        """
        from torchdiffeq import odeint as odeint
        time_eval, indices = self.split_time(t)
        
        pred_y = odeint(self.ode_func, x, time_eval, 
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.method)
        pred_y = pred_y.permute(1,2,0)
        
        # Repeat to select on second dimension the right index of time
        indices = indices.unsqueeze(-1).unsqueeze(-1).repeat(1, x.shape[1], time_eval.shape[0])        
        return torch.gather(pred_y, 2, indices)[:, :, 0] # Select first as it is repeated 

    def split_time(self, t):
        # Compute where we need to estimate hidden state
        res, index = torch.unique(t.abs().min(dim = 1)[0], sorted=True, return_inverse=True) #  Compute the min for all covariates since last time evaluated
        max_time = max(res[-1], 1e-4)
        if self.step_size > max_time:
            self.step_size = max_time / 20
        
        # Compute linspace with given stepsize and add points of interest
        ls = torch.linspace(0, max_time, int(max_time / self.step_size)).to(t.device)
        times, indices = torch.unique(torch.cat([res, ls]), sorted=True, return_inverse=True)

        return times, indices[:res.size(0)][index]
        

class ODECell(RNNCellBase):
    """
        Cell with decay between time points
    """

    def __init__(self, input_size, hidden_size, bias=True, step_size = 1.0, dropout = 0):
        super(ODECell, self).__init__(input_size, hidden_size, bias, num_chunks = 3)
        self.cell = nn.GRUCell(input_size = input_size, hidden_size = hidden_size, bias = bias)
        self.ode = ODESolver("euler", hidden_size, 
                        odeint_rtol = 1e-3, odeint_atol = 1e-4, step_size = step_size)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, t, hx = None):
        hx = self.ode(hx, t) if t.max() > 0 else hx
        return self.cell(x, hx) if self.dropout is None else self.dropout(self.cell(x, hx))


class ODE(GRUD):
    
    def __init__(self, inputdim, hidden, layers, bias=True, batch_first=True):
        super(ODE, self).__init__(inputdim, hidden, layers, bias)
        self.cell = ODECell(inputdim, hidden, bias)

    def pickle(self):
        self.cell.ode.ode_func.forward = None

    def unpickle(self):
        self.cell.ode.ode_func.forward = lambda t, x: self.cell.ode.ode_func.forward_y(x)
