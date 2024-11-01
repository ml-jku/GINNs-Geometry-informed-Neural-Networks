from functools import partial
from typing import List
import torch
from torch import nn
import math

from models.lip_mlp import ConditionalLipschitzLinear, LipschitzLinear
from models.siren import Sine, siren_uniform_

init_lipsiren_linear = partial(siren_uniform_, mode='fan_in', c=6)
    
class CondLipSIREN(nn.Module):
    def __init__(self, 
                 layers: List[int],
                 nz: int,
                 w0: float = 1.0,
                 w0_initial: float = 30.0):
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
        super(CondLipSIREN, self).__init__()
        self._check_params(layers)
        
        # first layer is conditioned on last dimension of input
        condition_idcs = range(layers[0] - nz, layers[0])
        self.layers = [ConditionalLipschitzLinear(layers[0], layers[1], cond_idcs=condition_idcs, init_fn=init_lipsiren_linear), Sine(w0=w0_initial)]

        for index in range(1, len(layers) - 2):
            self.layers.extend([
                LipschitzLinear(layers[index], layers[index + 1], init_fn=init_lipsiren_linear),
                Sine(w0=w0)
            ])

        self.layers.append(LipschitzLinear(layers[-2], layers[-1], init_fn=init_lipsiren_linear))
        self.network = nn.Sequential(*self.layers)


    @staticmethod
    def _check_params(layers):
        assert isinstance(layers, list), 'layers should be a list of ints'
        assert len(layers) >= 1, 'layers should not be empty'

    def get_lipschitz_loss(self):
        loss_lipc = self.layers[0].get_lipschitz_constant()
        # for the first layer, only use the Lipschitz constant of the z
        for i in range(1, len(self.layers)):
            if isinstance(self.layers[i], LipschitzLinear):
                loss_lipc = loss_lipc * self.layers[i].get_lipschitz_constant()
        # loss_lipc = loss_lipc *  self.layer_output.get_lipschitz_constant()
        return loss_lipc

    def forward(self, x, z):
        xz = torch.cat([x, z], dim=-1)
        res = self.network(xz)
        return res