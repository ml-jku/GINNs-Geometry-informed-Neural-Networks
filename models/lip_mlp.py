from mimetypes import init
from typing import List
import torch
from torch import nn
import math

# taken from: https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/lipmlp.py

def init_lipmlp_linear(weight):
    stdv = 1. / math.sqrt(weight.size(1))
    weight.data.uniform_(-stdv, stdv)

class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, init_fn):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters(init_fn)

    def initialize_parameters(self, init_fn):
        init_fn(self.weight)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.bias.data.uniform_(-stdv, stdv)

        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max() # just a rough initialization

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)

class ConditionalLipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, cond_idcs, init_fn):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cond_idcs = cond_idcs
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters(init_fn)

    def initialize_parameters(self, init_fn):
        init_fn(self.weight)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.bias.data.uniform_(-stdv, stdv)
        
        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max() # just a rough initialization

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight[:, self.cond_idcs]).sum(1)
        scale = torch.clamp(scale, max=1.0)
        # here we have to rescale only the weights corresponding to the conditional input
        # weight = self.weight * scale.unsqueeze(1)
        weight = self.weight.clone()
        weight[:, self.cond_idcs] = weight[:, self.cond_idcs] * scale.unsqueeze(1)
        return torch.nn.functional.linear(input, weight, self.bias)


class lipmlp(torch.nn.Module):
    def __init__(self, dims):
        """
        dim[0]: input dim
        dim[1:-1]: hidden dims
        dim[-1]: out dim

        assume len(dims) >= 3
        """
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for ii in range(len(dims)-2):
            self.layers.append(LipschitzLinear(dims[ii], dims[ii+1]))

        self.layer_output = LipschitzLinear(dims[-2], dims[-1])
        self.relu = torch.nn.ReLU()

    def get_lipschitz_loss(self):
        loss_lipc = 1.0
        for ii in range(len(self.layers)):
            loss_lipc = loss_lipc * self.layers[ii].get_lipschitz_constant()
        loss_lipc = loss_lipc *  self.layer_output.get_lipschitz_constant()
        return loss_lipc

    def forward(self, x):
        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            x = self.relu(x)
        return self.layer_output(x)
    
    
class CondLipMLP(nn.Module):
    def __init__(self, 
                 layers: List[int],
                 bias: bool = True):
        """
        SIREN model from the paper [Implicit Neural Representations with
        Periodic Activation Functions](https://arxiv.org/abs/2006.09661).
        Implementation modified from : https://github.com/dalmia/siren/tree/master/siren

        :param layers: list of number of neurons in each layer, including the
            input and output layers; e.g. [2, 3, 1] means 2 input, 1 output
        :type layers: List[int]
        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        :param w0_initial: `w0` of first layer. defaults to 30 (as used in the
            paper)
        :type w0_initial: float, optional
        :param bias: whether to use bias or not. defaults to
            True
        :type bias: bool, optional
        :param initializer: specifies which initializer to use. defaults to
            'siren'
        :type initializer: str, optional
        :param c: value used to compute the bound in the siren intializer.
            defaults to 6
        :type c: float, optional

        # References:
            -   [Implicit Neural Representations with Periodic Activation
                 Functions](https://arxiv.org/abs/2006.09661)
        """
        super(CondLipMLP, self).__init__()
        self._check_params(layers)
        
        assert bias, 'LipschitzLinear requires bias to be True'
        
        # LipschitzLinear(dims[ii], dims[ii+1]
        self.layers = [LipschitzLinear(layers[0], layers[1]), nn.Softplus()]

        for index in range(1, len(layers) - 2):
            self.layers.extend([
                LipschitzLinear(layers[index], layers[index + 1]),
                nn.Softplus()
            ])

        self.layers.append(LipschitzLinear(layers[-2], layers[-1]))
        self.network = nn.Sequential(*self.layers)

    @staticmethod
    def _check_params(layers):
        assert isinstance(layers, list), 'layers should be a list of ints'
        assert len(layers) >= 1, 'layers should not be empty'

    def get_lipschitz_loss(self):
        loss_lipc = 1.0
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], LipschitzLinear):
                loss_lipc = loss_lipc * self.layers[i].get_lipschitz_constant()
        # loss_lipc = loss_lipc *  self.layer_output.get_lipschitz_constant()
        return loss_lipc

    def forward(self, x, z):
        xz = torch.cat([x, z], dim=-1)
        res = self.network(xz)
        return res