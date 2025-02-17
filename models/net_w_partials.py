from torch import vmap
from torch._functorch.eager_transforms import jacfwd, jacrev
from torch._functorch.functional_call import functional_call

from turtle import st
from typing import Any

from cycler import V
import einops
import torch

from util.model_utils import tensor_product_xz


class NetWithPartials:
    '''
    The only stateful part of the model is the parameters.
    '''
    
    @staticmethod
    def create_from_model(model, nz, nx, group_size_fwd_no_grad=-1, **kwargs):
        ## Parameters for stateless model
        params = dict(model.named_parameters())
        ## Stateless model
        if nz == 0:
            assert False, 'change with power not implemented'
            def f(params, x):
                """
                Stateless call to the model. This works for
                1) single inputs:
                x: [nx]
                returns: [ny]
                -- and --
                2) batch inputs:
                x: [bx, nx]
                returns: [bx, ny]
                """
                return functional_call(model, params, x)
            ## Jacobian
            f_x = jacrev(f, argnums=1)  ## params, [nx] -> [nx, ny]
            vf_x = vmap(f_x, in_dims=(None, 0), out_dims=(0))  ## params, [bx, nx] -> [bx, ny, nx]
            ## Hessian
            f_xx = jacfwd(f_x, argnums=1)  ## params, [nx] -> [nx, ny, nx]
            vf_xx = vmap(f_xx, in_dims=(None, 0), out_dims=(0))  ## params, [bx, nx] -> [bx, ny, nx, nx]
            vf_z = None
        else:
            def f(params, x, z):
                return functional_call(model, params, (x, z))
            vf = vmap(f, in_dims=(None, 0, 0), out_dims=(0))  ## params, [bxz, nx], [bxz, nz] -> [bxz, ny]
            ## Note the difference: in the in_dims and out_dims we want to vectorize in the 0-th dimension
            ## Jacobian
            f_x = jacrev(f, argnums=1)  ## params, [nx], [nz] -> [nx, ny]
            vf_x = vmap(f_x, in_dims=(None, 0, 0), out_dims=(0))  ## params, [bxz, nx], [bxz, nz] -> [bxz, ny, nx]
            ## Hessian
            f_xx = jacfwd(f_x, argnums=1)  ## params, [nx], [nz] -> [nx, ny, nx]
            vf_xx = vmap(f_xx, in_dims=(None, 0, 0), out_dims=(0))  ## params, [bxz, nx], [bxz, nz] -> [bxz, ny, nx, nx]
            ## Jacobian wrt z
            f_z = jacrev(f, argnums=2)  ## params, [nx], [nz] -> [nz, ny]
            vf_z = vmap(f_z, in_dims=(None, 0, 0), out_dims=(0))  ## params, [bxz, nx], [bxz, nz] -> [bxz, ny, nz]
            
            return NetWithPartials(f=f, vf=vf, vf_x=vf_x, vf_xx=vf_xx, 
                               params=params, nz=nz, nx=nx, vf_z=vf_z, 
                               max_group_size=group_size_fwd_no_grad)
            
            # # using power
            # def f(params, x, z, p):
            #     res = functional_call(model, params, (x, z))
            #     return torch.pow(res, p)
            # vf = vmap(f, in_dims=(None, 0, 0, None), out_dims=(0))  ## params, [bxz, nx], [bxz, nz] -> [bxz, ny]
            # ## Note the difference: in the in_dims and out_dims we want to vectorize in the 0-th dimension
            # ## Jacobian
            # f_x = jacrev(f, argnums=1)
            # vf_x = vmap(f_x, in_dims=(None, 0, 0, None), out_dims=(0))
            # ## Hessian
            # f_xx = jacfwd(f_x, argnums=1)
            # vf_xx = vmap(f_xx, in_dims=(None, 0, 0, None), out_dims=(0))
            # ## Jacobian wrt z
            # f_z = jacrev(f, argnums=2)
            # vf_z = vmap(f_z, in_dims=(None, 0, 0, None), out_dims=(0))
    
        # return NetWithPartials(f=f, vf=vf, vf_x=vf_x, vf_xx=vf_xx, 
        #                        params=params, nz=nz, nx=nx, vf_z=vf_z, 
        #                        max_group_size=group_size_fwd_no_grad, p=p)
    
    def __init__(self, f, vf, vf_x, vf_xx, params, nz, nx, vf_z=None, max_group_size=None) -> None:
        self.f_ = f
        self.vf_ = vf
        self.vf_x_ = vf_x
        self.vf_xx_ = vf_xx
        self.vf_z_ = vf_z
        self.params = params
        self.nz = nz
        self.nx = nx
        self.max_group_size = max_group_size
    
    def __call__(self, *args: Any) -> Any:
        return self.vf(*args) # params and p are already included in the vf call
    
    def f(self, *args: Any) -> Any:
        return self.f_(self.params, *args)
    
    def vf(self, *args: Any) -> Any:
        return self.vf_(self.params, *args)
    
    def vf_x(self, *args: Any) -> Any:
        return self.vf_x_(self.params, *args)
    
    def vf_xx(self, *args: Any) -> Any:
        return self.vf_xx_(self.params, *args)
    
    def vf_z(self, *args: Any) -> Any:
        if self.vf_z_ is None:
            raise ValueError('vf_z not defined. Have you specified nz > 0?')
        return self.vf_z_(self.params, *args)
    
    def detach(self) -> None:
        ## Make the parameters not request gradient computation by creating a detached copy.
        ## The copy is shallow, meaning it shares memory, and is tied to parameter updates in the outer loop.
        new_params = {key: tensor.detach() for key, tensor in self.params.items()}
        new_net = NetWithPartials(f=self.f_, vf=self.vf_, vf_x=self.vf_x_, vf_xx=self.vf_xx_, 
                                  params=new_params, nz=self.nz, nx=self.nx, vf_z=self.vf_z_, 
                                  max_group_size=self.max_group_size)
        return new_net
    
    def grouped_no_grad_fwd(self, f_str, x, z) -> Any:
        '''Groups the forward pass into groups of size max_group_size. To reduce peak memory usage.
            This is useful for PH and Meshgrid forward passes.
        Args
            x: [b, nx]
            z: [b, nz]
        Returns
            [b, by]
        '''
        assert self.max_group_size is not None, 'max_group_size must be set in the constructor.'
        assert f_str in ['f', 'vf', 'vf_x', 'vf_xx', 'vf_z'], 'f_str must be one of the following: f, vf, f_x, vf_x, f_xx, vf_xx, vf_z'
        
        # choose function and normalize group size by the number of dimensions
        if f_str in ['f', 'vf']:
            func = self.vf
            group_size = self.max_group_size
        elif f_str in ['vf_x']:
            func = self.vf_x
            group_size = self.max_group_size // self.nx
        elif f_str in ['vf_xx']:
            func = self.vf_xx
            group_size = self.max_group_size // (self.nx * self.nx)
        elif f_str == 'vf_z':
            func = self.vf_z
            group_size = self.max_group_size // self.nz
        
        # if max_group_size is not set, or group is larger than the input, set group to the input size
        if self.max_group_size < 0 or group_size > x.shape[0]:
            group_size = x.shape[0]
            
        if x.shape[0] == 0:
            return 
        
        with torch.no_grad():
            res = []
            for i in range(0, z.shape[0], group_size):
                z_group = z[i:min(z.shape[0], i+group_size)]  # last group might be smaller
                x_group = x[i:min(z.shape[0], i+group_size)]  # last group might be smaller
                y = func(x_group, z_group)
                res.append(y)
                
        return torch.cat(res, dim=0)