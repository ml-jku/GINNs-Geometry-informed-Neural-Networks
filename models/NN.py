import torch
from torch import nn
from torch import sin
from torch.nn.functional import relu, celu, tanh

class ConditionalGeneralResNet(nn.Module):
    def __init__(self, ks, act=celu):
        super(ConditionalGeneralResNet, self).__init__()
        self.ks = ks
        self.fcs = nn.ModuleList([nn.Linear(in_features, out_features)
            for in_features, out_features in zip(self.ks[:-1],self.ks[1:])])
        self.D = len(self.fcs)
        self.act = act

    def forward(self, x, z):
        '''
        First concatenates to tensor of dim [bz bx (nx+nz)]. Then it passes this tensor through the network.
        :param x: [bx nx]
        :param z: [1 nz]
        Will later be vectorized, to go for [bz bx (nx+nz)]
        :return:
        '''
        xz = torch.cat([x, z], dim=-1)
        x = self.act(self.fcs[0](xz))
        for i in range(2,self.D):
            # Note: as per 7.12.2023 it throws the following error if x += ... is used
            # Exception has occurred: RuntimeError       (note: full exception trace is shown but execution is paused at: _run_module_as_main)
            # one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [512, 2, 64]],
            x = x + self.act(self.fcs[i-1](x))
        x = self.fcs[self.D-1](x)
        return x

class ConditionalGeneralNet(nn.Module):
    def __init__(self, ks, act=celu):
        super(ConditionalGeneralNet, self).__init__()
        self.ks = ks
        self.fcs = nn.ModuleList([nn.Linear(in_features, out_features)
            for in_features, out_features in zip(self.ks[:-1],self.ks[1:])])
        self.D = len(self.fcs)
        self.act = act

    def forward(self, x, z):
        '''
        First concatenates to tensor of dim [bz bx (nx+nz)]. Then it passes this tensor through the network.
        :param x: [bx nx]
        :param z: [1 nz]
        Will later be vectorized, to go for [bz bx (nx+nz)]
        :return:
        '''
        xz = torch.cat([x, z], dim=-1)
        x = self.fcs[0](xz)
        for i in range(2,self.D):
            x = self.fcs[i-1](self.act(x))  # as per 
        x = self.fcs[self.D-1](self.act(x))
        return x


class GeneralNet(nn.Module):
    def __init__(self, ks, act=celu):
        super(GeneralNet, self).__init__()
        self.ks = ks
        self.fcs = nn.ModuleList([nn.Linear(in_features, out_features)
            for in_features, out_features in zip(self.ks[:-1],self.ks[1:])])
        self.D = len(self.fcs)
        self.act = act

    def forward(self, x):
        x = self.fcs[0](x)
        for i in range(2,self.D+1):
            x = self.fcs[i-1](self.act(x))
        return x
    
class GeneralResNet(nn.Module):
    def __init__(self, ks, act=celu):
        super(GeneralResNet, self).__init__()
        self.ks = ks
        self.fcs = nn.ModuleList([nn.Linear(in_features, out_features)
            for in_features, out_features in zip(self.ks[:-1],self.ks[1:])])
        self.D = len(self.fcs)
        self.act = act

    def forward(self, x):
        x = self.act(self.fcs[0](x))
        for i in range(2,self.D):
            # Note: as per 7.12.2023 it throws the following error if x += ... is used
            # Exception has occurred: RuntimeError       (note: full exception trace is shown but execution is paused at: _run_module_as_main)
            # one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [512, 2, 64]],
            x = x + self.act(self.fcs[i-1](x))
        x = self.fcs[self.D-1](x)
        return x

class GeneralNetBunny(GeneralNet):
    """Defines the architecture and loads the weights of a pretrained 2D bunny with a few different activation functions."""
    def __init__(self, act='sin', path_prefix="GINN"):
        assert act in ['sin','celu','relu'], "act must be sin, celu, relu"
        acts = {'sin':sin, 'celu':celu, 'relu':relu}
        GeneralNet.__init__(self, ks=[2,32,32,32,32,1], act=acts[act])
        self.load_state_dict(torch.load(f'{path_prefix}/models/2,32,32,32,32,1,{act},bunny'))


PI = torch.pi
def encode(x, N=5):
    '''
    Compute the positional encoding of inputs x.
    x: tensor of shape [B, D]
    N: number of frequencies
    Returns:
    enc: tensor of shape [B, 2*D*N]
    '''
    Ns = torch.logspace(start=1, end=N, steps=N, base=2, device=x.device)[:,None,None]
    batched = len(x.shape)==2
    if not batched:
        x = x.unsqueeze(0)
    block = PI*(x*Ns).permute(1,2,0)
    # print("block:", block.shape) # [B, nx, N]
    enc = torch.hstack([
            torch.sin(block),
            torch.cos(block),
        ]).reshape(len(x), -1)
    if not batched: enc = enc.squeeze(0)
    return enc


class GeneralNetPosEnc(GeneralNet):
    """ NOTE: need to compute proper ks yourself. For example
    enc_dim = 2*nx*config['N_posenc']
    model = GeneralNetPosEnc(ks=[2, enc_dim, 20, 20, 1])
    """
    def __init__(self, ks, act=celu):
        super(GeneralNetPosEnc, self).__init__(ks=ks, act=act)
        self.N_posenc = ks[1]//(2*self.ks[0])
    def forward(self, x):
        x = encode(x, N=self.N_posenc) # 0 -> 1
        x = self.fcs[1](x) # 1 -> 2
        for i in range(3, self.D+1): # 2 -> 3 --> D
            x = self.fcs[i-1](self.act(x))
        return x


## TODO: unify conditional and non-conditional (not just here) to not repeat code twice. 
class GeneralNetFFN(GeneralNet):
    """ Wrapper around FFN (Fourier Feature Network)[Tancik] """
    def __init__(self, ks, act=tanh, N_ffeat=0, sigma=1, nx=2):
        super(GeneralNetFFN, self).__init__(ks=ks, act=act)
        self.N_ffeat = N_ffeat
        if self.N_ffeat:
            self.B = torch.normal(0, sigma**2, size=[self.N_ffeat, nx])
    def forward(self, x):
        if self.N_ffeat:
            x_B = 2*torch.pi*x@self.B.T
            x_sin = torch.sin(x_B)
            x_cos = torch.cos(x_B)
            x = torch.hstack([x_sin, x_cos])
        x = self.fcs[0](x) # 0 -> 1
        for i in range(1, self.D): # 1 --> D
            x = self.fcs[i](self.act(x))
        return x


