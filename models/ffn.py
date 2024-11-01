from typing import List
import torch
from torch import nn
from torch import sin
from torch.nn.functional import relu, celu, tanh


class ConditionalFFN(nn.Module):
    """ Wrapper around FFN (Fourier Feature Network)[Tancik] """
    def __init__(self, 
                layers: List[int],
                 nz,
                 n_ffeat, 
                 act='softplus', 
                 sigma=1, 
                 bias=True,
                 ):
        super().__init__()
        
        if act == 'softplus':
            act = nn.Softplus()
        else:
            raise NotImplementedError(f"Activation {act} not implemented for FFN.")
        
        # first layer
        self.layers = [nn.Linear(2 * n_ffeat + nz, layers[1], bias=bias), act]
        
        # add other layers except the last one
        for index in range(1, len(layers) - 2):
            self.layers.extend([
                nn.Linear(layers[index], layers[index + 1], bias=bias),
                act,
            ])
            
        # add the last layer
        self.layers.append(nn.Linear(layers[-2], layers[-1], bias=bias))
        self.network = nn.Sequential(*self.layers)

        # initialize the Fourier features
        # make this deterministic even if global seed changes        
        rng = torch.Generator()
        rng.manual_seed(0)
        n_in = layers[0] - nz
        self.B = torch.normal(0, sigma**2, size=[n_ffeat, n_in], generator=rng)
        # make B a buffer so that it is loaded to the device and saved with the model but not trained
        # self.register_buffer('B', torch.normal(0, sigma**2, size=[n_ffeat, n_in], generator=rng))
            
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
    