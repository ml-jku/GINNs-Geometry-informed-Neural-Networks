import torch
import einops

from models.net_w_partials import NetWithPartials
from torch import vmap
from torch._functorch.eager_transforms import jacfwd, jacrev
from torch._functorch.functional_call import functional_call


def tensor_product_xz(x, z):
    """
    Generate correct inputs for bx different xs and bz different zs.
    For each z we want to evaluate it at all xs and vice versa, so we need a tensor product between the rows of the two vectors.
    x: [bx, nx]
    z: [bz, nz]
    returns: [bx*bz, nx+nz]
    """
    z_tp = z.repeat_interleave(len(x), 0)
    x_tp = x.repeat(len(z), 1)
    return x_tp, z_tp


if __name__ == '__main__':

   pass


def get_activation(act_str):
    if act_str == 'relu':
        activation = torch.relu
    elif act_str == 'softplus':
        activation = torch.nn.Softplus(beta=10)
    elif act_str == 'celu':
        activation = torch.celu
    elif act_str == 'sin':
        activation = torch.sin
    elif act_str == 'tanh':
        activation = torch.tanh
    else:
        activation = None
        # print(f'activation not set')

    return activation


def get_stateless_net_with_partials(model, nz=0):
    """
    Returns the stateless representation of a torch model,
    including the vectorized Jacobian and Hessian matrices.
    """

    ## Parameters for stateless model
    params = dict(model.named_parameters())

    ## Stateless model
    if nz == 0:
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

        # f_theta = jacrev(f, argnums=0)  ## params, [nx], [nz] -> [nx, n_params]
        # vf_theta = vmap(f_theta, in_dims=(None, 0, 0), out_dims=(0))  ## params, [bxz, nx], [bxz, nz] -> [bxz, nx, n_params]

    netp = NetWithPartials(f, f_x, vf_x, f_xx, vf_xx, params, vf_z)

    return netp