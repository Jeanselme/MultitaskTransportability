from .utils import *
from .RNN.rnn import RNN
from .Survival.survival import Survival
from .observational import Observational

class RNNJointTorch(BatchForward):
    """
        Torch implementation of a recurrent joint model 
        With LSTM shared representation

        Factory like object which aggregate the multiple submodules
    """
    
    def __init__(self, inputdim, outputdim = 1, 
                typ = 'LSTM', layers = 1, hidden = 10, recurrent_args = {},
                survival = "deepsurv", survival_args = {}, weight = 0.,
                temporal = "None", temporal_args = {},
                longitudinal = "None", longitudinal_args = {}, 
                missing = "None", missing_args = {}, obs_mask = None, dropout = 0.):
        """
        Args:
            inputdim (int): Input dimension
            outputdim (int, optional): Number of potential risk(s). Defaults to 1.
            typ (str, optional): Recurrent cell type. Defaults to 'LSTM'.
            layers (int, optional): Number of reccurent layers. Defaults to 1.
            hidden (int, optional): Dimension hidden state. Defaults to 10.
            recurrent_args (dict, optional): Arguments for the model. Defaults to {}.

            survival (str, optional): Type of survival modelling.
                Possible choices: "deepsurv", "deephit", "full"
                Defaults to "deepsurv".
            survival_args (dict, optional): Arguments for the model. Defaults to {}.

            observational_components (int, optional): Number of components for mixture of 
                observational processes
                Defaults to 1.
            weight (float, opt): Weight to put on the observational process. Default to 0.

            temporal (str, optional): Type of temporal modelling. (When will be the next values ?)
                Possible choices: "point", "weibull" and "None".
                Defaults to "None".
            temporal_args (dict, optional): Arguments for the model. Defaults to {}.

            longitudinal (str, optional): Type of temporal modelling. (What will be the next values ?)
                Possible choices: "neural", "gaussian" and "None".
                Defaults to "None".
            longitudinal_args (dict, optional): Arguments for the model. Defaults to {}.

            missing (str, optional): Type of mask modelling (What will be missing ?).
                Possible choices: "neural", "bernoulli" and "None".
                Defaults to "None".
            missing_args (dict, optional): Arguments for the model. Defaults to {}.
        """
        super(RNNJointTorch, self).__init__()

        self.inputdim = inputdim
        self.outputdim = outputdim

        # Recurrent model
        self.embedding = RNN(self.inputdim, typ, layers, hidden, recurrent_args, dropout)

        # Survival model
        self.survival_model = Survival.create(survival, hidden, self.outputdim, survival_args, dropout)

        # Observational process
        self.weight = weight
        self.temporal, self.longitudinal, self.missing = (temporal != 'None'), (longitudinal != 'None'), (missing != 'None')
        self.observational = self.temporal or self.longitudinal or self.missing
        self.obs_mask = np.full(self.inputdim, True) if obs_mask is None else obs_mask
        if self.observational:
            self.observational_model = Observational(hidden, self.obs_mask.sum(),
                                                temporal, temporal_args, 
                                                longitudinal, longitudinal_args, 
                                                missing, missing_args, dropout)
    
    def pickle(self):
        self.embedding.pickle()

    def unpickle(self):
        self.embedding.unpickle()

    def compute_baseline(self, x, ie_to, ie_since, m, e, l, t, batch = None):
        hidden, = self.embedding.forward(x, ie_since, m, l, total_l = l.max(), batch = batch)
        hp = get_last_observed(hidden, l - 1)
        self.survival_model.compute_baseline(hp, e, t, batch = batch)
        return self
    
    def loss(self, x, ie_to, ie_since, m, e, l, t, batch = None, reduction = 'mean', survival = True, observational = True, weights = {}):
        """
            Compute loss model (need sorted if survival == True)
        """
        hidden, = self.embedding.forward(x, ie_since, m, l, total_l = l.max(), batch = batch)
        loss, losses = 0, {}
        
        if survival:
            hp = get_last_observed(hidden, l - 1)
            loss = losses['survival'] = self.survival_model.loss(hp, e, t, batch, reduction)

        if self.observational and observational:
            losses['observational'] = torch.stack(self.observational_model.loss(hidden, x[:, :, self.obs_mask], ie_to[:, :, self.obs_mask], m[:, :, self.obs_mask], l, batch, reduction))
            weight_surv, weight_obs = weights.get("survival", 1), weights.get("observational", 1)
            loss = (1 - self.weight) * (weight_surv * loss) + self.weight * (weight_obs * losses['observational']).sum()
            
        return loss, losses

    def predict_batch(self, x, ie_to, ie_since, m, l, horizon, risk = 1):
        hidden, = self.embedding.forward_batch(x, ie_since, m, l, total_l = l.max())
        hp = get_last_observed(hidden, l - 1)
        return self.survival_model.predict_batch(hp, horizon = horizon, risk = risk)[0],

    def observational_predict(self, x, ie_to, ie_since, m, l, batch = None):
        assert self.observational, "Do not model observational outcome"
        _, hidden = self.embedding.forward(x, ie_since, m, l, batch = batch)
        return self.observational_model.forward(hidden, ie_to[:, :, self.obs_mask], m[:, :, self.obs_mask], l, batch = batch)