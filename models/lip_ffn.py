from typing import List
import torch
from torch import nn
from torch import sin
from torch.nn.functional import relu, celu, tanh

from models.lip_mlp import LipschitzLinear, ConditionalLipschitzLinear, init_lipmlp_linear

class LipschitzConditionalFFN(nn.Module):
    """ Wrapper around FFN (Fourier Feature Network)[Tancik] """
    def __init__(self, 
                layers: List[int],
                 nz,
                 n_ffeat, 
                 act='softplus', 
                 sigma=1,
                 ):
        super().__init__()
        
        if act == 'softplus':
            act = nn.Softplus()
        else:
            raise NotImplementedError(f"Activation {act} not implemented for FFN.")
        
        # first layer
        condition_idcs = range(layers[0] - nz, layers[0])
        self.layers = [ConditionalLipschitzLinear(2 * n_ffeat + nz, layers[1], cond_idcs=condition_idcs, init_fn=init_lipmlp_linear), act]
        
        # add other layers except the last one
        for index in range(1, len(layers) - 2):
            self.layers.extend([
                LipschitzLinear(layers[index], layers[index + 1], init_fn=init_lipmlp_linear),
                act,
            ])
            
        # add the last layer
        self.layers.append(LipschitzLinear(layers[-2], layers[-1], init_fn=init_lipmlp_linear))
        self.network = nn.Sequential(*self.layers)

        # initialize the Fourier features
        # make this deterministic even if global seed changes        
        rng = torch.Generator()
        rng.manual_seed(0)
        n_in = layers[0] - nz
        self.B = torch.normal(0, sigma**2, size=[n_ffeat, n_in], generator=rng)
        # make B a buffer so that it is loaded to the device and saved with the model but not trained
        # self.register_buffer('B', torch.normal(0, sigma**2, size=[n_ffeat, n_in], generator=rng))
    
    def get_lipschitz_loss(self):
        # for the first layer, only use the Lipschitz constant of the z
        loss_lipc = self.layers[0].get_lipschitz_constant()
        # iterate over rest of the layers
        for i in range(1, len(self.layers)):
            if isinstance(self.layers[i], LipschitzLinear):
                loss_lipc = loss_lipc * self.layers[i].get_lipschitz_constant()
        return loss_lipc
    
    def forward(self, x, z):
        # make sure B is on the same device as x
        self.B = self.B.to(x.device)
        
        # compute the Fourier features
        x_B = 2*torch.pi*x@self.B.T
        x_sin = torch.sin(x_B)
        x_cos = torch.cos(x_B)
        x_ff = torch.hstack([x_sin, x_cos])
        
        # forward pass
        xz = torch.cat([x_ff, z], dim=-1)
        res = self.network(xz)
        return res
