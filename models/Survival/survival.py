from ..utils import *
from scipy.interpolate import interp1d
import torch.nn as nn
import torch

class Survival():
    """
        Factory object
    """
    @staticmethod
    def create(survival, inputdim, outputdim, survival_args = {}, dropout = 0.): 
        if survival == 'deepsurv':
            return DeepSurv(inputdim, outputdim, survival_args, dropout)
        elif survival == 'nfg':
            return NFG(inputdim, outputdim, survival_args, dropout)
        else:
            raise NotImplementedError()


class DeepSurv(BatchForward):
    """
        DeepSurv
    """

    def __init__(self, inputdim, outputdim, survival_args = {}, dropout = 0.):
        """
        Args:
            inputdim (int): Input dimension (hidden state)
            outputdim (int): Output dimension (number competing risks)
            survival_args (dict, optional): Arguments for the model. Defaults to {}.
        """
        super(DeepSurv, self).__init__()

        self.inputdim = inputdim
        self.outputdim = outputdim

        survival_layer = survival_args.get("layers", [100])
        
        # One neural network for each outcome => More flexibility
        self.survival = nn.ModuleList([nn.Sequential(*create_nn(inputdim, survival_layer + [1], dropout = dropout)[:-1]) 
                                    for _ in range(outputdim)])

    def forward_batch(self, h):
        outcome = [o(h) for o in self.survival]
        outcome = torch.cat(outcome, -1)
        return outcome,

    def loss(self, h, e, t, batch = None, reduction = 'mean', weights = None):
        h, e, t = sort_given_t(h, e, t = t) # Sort data
        loss, e = 0., e.squeeze()
        predictions, = self.forward(h, batch = batch)

        # Weights the different event differently
        weights = torch.ones_like(e) if weights is None else weights

        ## Sum all previous event : **Require order by decreasing time**: prediciton at time t is the hazard
        p_cumsum = torch.logcumsumexp(predictions, 0)
        for ei in range(1, self.outputdim + 1):
            loss += torch.sum((p_cumsum[e == ei][:, ei - 1] - predictions[e == ei][:, ei - 1]) * weights[e == ei])

        if reduction == 'mean' and (e != 0).sum() > 0:
            loss /= weights[e != 0].sum()

        return loss

    def compute_baseline(self, h, e, t, batch = None):
        # Breslow estimator
        # At time of the event, the cumulative proba is one
        predictions = torch.exp(self.forward(h, batch = batch)[0])

        # Remove duplicates and order
        self.baselines = []
        self.times, indices = torch.unique(t.squeeze(), return_inverse = True, sorted = True) # Inverse is the position of the element i in new list times
        for risk in range(1, self.outputdim + 1):
            e_summed, p_summed = [], []
            for i, _ in enumerate(self.times):
                # Descending order
                e_summed.insert(0, (e[indices == i] == risk).sum())
                p_summed.insert(0, predictions[indices == i][:, risk - 1].sum()) # -1 because only one dimension for each risk (no 0 modelling)

            e_summed, p_summed = torch.DoubleTensor(e_summed), \
                                torch.DoubleTensor(p_summed)
            p_summed = torch.cumsum(p_summed, 0) # Number patients
            
            # Reverse order
            self.baselines.append(torch.cumsum((e_summed / p_summed)[torch.arange(len(self.times), 0, -1) - 1], 0).unsqueeze(0))
        self.baselines = torch.cat(self.baselines, 0)
        return self

    def predict_batch(self, h, horizon, risk = 1):
        forward, = self.forward_batch(h)
        cumulative_hazard = self.baselines[risk - 1].unsqueeze(0)
        if h.is_cuda:
            cumulative_hazard = cumulative_hazard.cuda()

        # exp(W X) * Cum_intensity = Cumulative hazard at time t
        # Survival = exp(-cum hazard) 
        predictions = torch.exp(- torch.matmul(torch.exp(forward[:, risk - 1].unsqueeze(1)), cumulative_hazard))

        if isinstance(horizon, list):
            # Interpolate to make the prediction at the point of interest
            result = []
            for h in horizon:
                if h > self.times[-1]:
                    closest = len(self.times)
                else:
                    _, closest = torch.min((self.times <= h), 0)
                closest -= 1
                if closest < 0:
                    result.append(torch.ones_like(predictions[:, 0]))
                else:
                    result.append(predictions[:, closest])
            predictions = torch.stack(result).T
        return predictions,
        

class NFG(BatchForward):
    def __init__(self, inputdim, outputdim, survival_args = {}, dropout = 0.):
        """
        Args:
            inputdim (int): Input dimension (hidden state)
            outputdim (int): Output dimension (number competing risks)
            survival_args (dict, optional): Arguments for the model. Defaults to {}.
        """
        from NeuralFineGray.nfg.nfg_torch import NeuralFineGrayTorch
        super(NFG, self).__init__()

        self.inputdim = inputdim
        self.outputdim = outputdim
        
        self.survival = NeuralFineGrayTorch(inputdim, risks = 2, dropout = dropout, **survival_args)

    def forward_batch(self, h, t, gradient = False):
        return self.survival(h, t, gradient)

    def loss(self, h, e, t, batch = None, reduction = 'mean', weights = None):
        e, t = e.squeeze(), t.squeeze()
        log_sr, log_hr, log_b = self.forward(h, t, gradient = True, batch = batch)

        log_balance_sr = log_b + log_sr
        log_balance_derivative = log_b + log_sr + log_hr

        weights = torch.ones_like(e) if weights is None else weights

        # Likelihood error
        if (e == 0).sum() > 0:
            error = - (torch.logsumexp(log_balance_sr[e == 0], dim = 1) * weights[e == 0]).sum() # Sum over the different risks and then across patient
        else:
            error = 0
        for k in range(self.survival.risks):
            error -= (log_balance_derivative[e == (k + 1)][:, k] * weights[e == (k + 1)]).sum() # Sum over patients with this risk

        if reduction == 'mean' and (e != 0).sum() > 0:
            error /= weights.sum()

        return error

    def compute_baseline(self, h, e, t, batch = None):
        return self # No baseline estimation
    
    def predict_batch(self, h, horizon, risk = 1):
        outcomes = torch.zeros((len(h), len(horizon)), device = h.get_device() if h.is_cuda else 'cpu')
        for i, time in enumerate(horizon):
            log_sr, _, log_beta  = self.forward_batch(h, torch.full((len(h),), time, device = h.get_device() if h.is_cuda else 'cpu'))
            outcomes[:, i] = (1 - log_beta.exp()  * (1 - torch.exp(log_sr)))[:, int(risk) - 1] # Exp diff => Ignore balance but just the risk of one disease
        return outcomes,
