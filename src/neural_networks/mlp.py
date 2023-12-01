from neural_networks.base import NNBase
from torch import nn
import torch
import torch.utils.data

class MLP(NNBase):
    """
    Multilayer perceptron with configurable hidden layers and activation functions.
    """
    def __init__(
        self,
        dim_in,
        hidden,
        dim_out,
        dropout=0.1,
        activation="tanh",
        cuda=False,
        seed=42):

        # init parent class
        super(MLP, self).__init__(cuda)

        # define parameters
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden = hidden
        self.activation_name = activation
        self.dropout_p = dropout
        
        # choose activation functions
        # https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions
        self.activation = getattr(nn.functional, activation)
        self.output_activation = nn.Identity()
    
        # construct layers
        self.layers = nn.ModuleList()
        in_size = self.dim_in
        for i, next_size in enumerate(hidden):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.layers.append(fc)
        self.fcout = nn.Linear(hidden[-1], dim_out)
        self.dropout = nn.Dropout(p=dropout)

        # loss function
        self._loss = nn.MSELoss()

        # set seed and cuda
        self.seed = seed
        torch.manual_seed(seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed(seed)
        self.to(self._device)
    
    def forward(self, x):
        for layer in enumerate(self.layers):
            x = layer[1](x)
            x = self.activation(x)
        preactivation = self.fcout(x)
        return self.output_activation(preactivation)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).detach()
        
    def save(self, file_path):
        kwargs = {  "dim_in": self.dim_in,
                    "hidden": self.hidden, 
                    "dim_out": self.dim_out,
                    "cuda": self.cuda_enabled, 
                    "seed": self.seed, 
                    "dropout": self.dropout_p, 
                    "activation": self.activation_name}
        state_dict = self.state_dict()
        torch.save({
            "kwargs":kwargs, "state_dict":state_dict}, file_path)