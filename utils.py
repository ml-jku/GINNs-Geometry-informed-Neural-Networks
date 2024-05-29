import math
import os
import random

import numpy as np
import torch
from torch import vmap
from torch._functorch.eager_transforms import jacrev, jacfwd
from torch._functorch.functional_call import functional_call
from models.net_w_partials import NetWithPartials
from models.siren import ConditionalSIREN

from models.NN import ConditionalGeneralNet, ConditionalGeneralResNet, GeneralNet, GeneralNetBunny, GeneralNetPosEnc, GeneralResNet


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def flip_arrow(xy, dxy):
    return xy + dxy, -1 * dxy

def set_and_true(key, config):
    return (key in config) and config[key]

def get_is_out_mask(x, bounds):
    out_mask = (x < bounds[:, 0]).any(1) | (x > bounds[:, 1]).any(1)
    return out_mask

def find_file_with_substring(run_id, root_dir, use_epoch=None):
    # search in all subdirectories
    candidate_files = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if run_id in file:
                candidate_files.append(os.path.join(subdir, file))
    
    if len(candidate_files) > 1:
        print(f"Found {len(candidate_files)} candidate files")
        for file in candidate_files:
            print(file)
    
    if use_epoch is None:
        # return file with shortest path
        return min(candidate_files, key=len)
    else:
        for file in candidate_files:
            if f'it_{use_epoch}' in file:
                return file
    
    raise ValueError(f"Could not find model with {run_id=} in {root_dir=}")

def get_model_path_via_wandb_id_from_fs(run_id, base_dir='./', use_epoch=None):
    dirs = ['/system/user/publicwork/radler/objgen/saved_models']
    if base_dir is not None: dirs.append(base_dir)
    for root_dir in dirs:
        try:
            file_path = find_file_with_substring(run_id, root_dir, use_epoch=use_epoch)
            print(f'Found model at {file_path}')
            return file_path
        except Exception as e:
            print(e)
    raise ValueError(f"Could not find model with {run_id=} anywhere")

def generate_color(i):
    r = (i & 1) * 255       # Red when i is odd
    g = ((i >> 1) & 1) * 255 # Green when the second bit of i is 1
    b = ((i >> 2) & 1) * 255 # Blue when the third bit of i is 1
    return r + (g << 8) + (b << 16)

def precompute_sample_grid(n_points, bounds):
        '''
        An equidistant grid of points is computed. These are later taken as starting points to discover critical points
        via gradient descent. The number of total points defined in config['n_points_find_cps'] is equally distributed
        among all dimensions.
        :return:
        xc_grid: the 2d or 3d grid of equidistant points over the domain
        xc_grid_dist: the distance as a 2d or 3d vector neighbors along the respective dimension
        '''
        nx = bounds.shape[0]
        
        n_points_root = int(math.floor(math.pow(n_points, 1 / nx)))
        dist_along_a_dim = 1 / (n_points_root + 1)
        xi_range = torch.arange(start=0, end=n_points_root, step=1) * dist_along_a_dim + dist_along_a_dim / 2
        if nx == 2:
            x1_grid, x2_grid = torch.meshgrid(xi_range, xi_range, indexing="ij")
            xc_grid = torch.stack((x1_grid.reshape(-1), x2_grid.reshape(-1)), dim=1)
        elif nx == 3:
            x1_grid, x2_grid, x3_grid = torch.meshgrid(xi_range, xi_range, xi_range, indexing="ij")
            xc_grid = torch.stack((x1_grid.reshape(-1), x2_grid.reshape(-1), x3_grid.reshape(-1)), dim=1)
        xc_grid = bounds[:, 0] + (bounds[:, -1] - bounds[:, 0]) * xc_grid
        xc_grid_dist = torch.tensor(dist_along_a_dim).repeat(nx) * (bounds[:, -1] - bounds[:, 0])
        return xc_grid, xc_grid_dist
    

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def get_stateless_net_with_partials(model, use_x_and_z_arg=False):
    """
    Returns the stateless representation of a torch model,
    including the vectorized Jacobian and Hessian matrices.
    """

    ## Parameters for stateless model
    params = dict(model.named_parameters())

    ## Stateless model
    if not use_x_and_z_arg:
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
        f_x = jacrev(f, argnums=1)  ## params, [nx] -> [ny, nx]
        vf_x = vmap(f_x, in_dims=(None, 0), out_dims=(0))  ## params, [bx, nx] -> [bx, ny, nx]
        ## Hessian
        f_xx = jacfwd(f_x, argnums=1)  ## params, [nx] -> [ny, nx, nx]
        vf_xx = vmap(f_xx, in_dims=(None, 0), out_dims=(0))  ## params, [bx, nx] -> [bx, ny, nx, nx]
    else:
        def f(params, x, z):
            return functional_call(model, params, (x, z))

        ## Note the difference: in the in_dims and out_dims we want to vectorize in the 0-th dimension
        ## Jacobian
        f_x = jacrev(f, argnums=1)  ## params, [nx] -> [ny, nx]
        vf_x = vmap(f_x, in_dims=(None, 0, 0), out_dims=(0))  ## params, [bx, nx] -> [bx, ny, nx]
        ## Hessian
        f_xx = jacfwd(f_x, argnums=1)  ## params, [nx] -> [ny, nx, nx]
        vf_xx = vmap(f_xx, in_dims=(None, 0, 0), out_dims=(0))  ## params, [bx, nx] -> [bx, ny, nx, nx]

    nep = NetWithPartials(f, vf_x, vf_xx, params)

    return nep


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

def do_plot(config, epoch, key=None):
        """
        Checks if the plot specified by the key should be produced.
        First checks if the output is set: if not, no need to produce the plot.
        Then, checks if the key is set and true.
        If the key is not specified, only checks the global plotting behaviour.
        """
        is_output_set = config['fig_show'] or config['fig_save'] or config['fig_wandb']

        # if no output is set, no need to produce the plot
        if is_output_set == False:
            return False
        
        # if no key is specified, only check the global plotting behaviour
        if key is None:
            return is_output_set
        
        # otherwise, check if the key is set
        if key not in config:
            return False
        
        val = config[key]
        if isinstance(val, bool):
            return val
        
        if isinstance(val, dict):
            # if the val is a tuple, it is (do_plot, plot_interval)
            if val['on'] == False:
                return False
            else:
                assert val['interval'] % config['plot_every_n_epochs'] == 0, f'plot_interval must be a multiple of plot_every_n_epochs'
                return (epoch % val['interval'] == 0) or (epoch == config['max_epochs'])
        
        raise ValueError(f'Unknown value for key {key}: {val}')


def get_model(config):
    
    model_str = config['model']
    activation = get_activation(config.get('activation', None))
    
    if model_str == 'siren':
        raise ValueError('SIREN is not supported yet; the layers and in_features are not well defined')
        # model = SIREN(layers=config['layers'], in_features=config['nx'], out_features=1, w0=config['w0'], w0_initial=config['w0_initial'])
    elif model_str == 'cond_siren':
        model = ConditionalSIREN(layers=config['layers'], w0=config['w0'], w0_initial=config['w0_initial'])
    elif model_str == 'general_net':
        model = GeneralNet(ks=config['layers'], act=activation)
    elif model_str == 'general_resnet':
        model = GeneralResNet(ks=config['layers'], act=activation)
    elif model_str == 'cond_general_net':
        model = ConditionalGeneralNet(ks=config['layers'], act=activation)
    elif model_str == 'cond_general_resnet':
        model = ConditionalGeneralResNet(ks=config['layers'], act=activation)
    elif model_str == 'general_net_posenc':
        enc_dim = 2*config['nx']*config['N_posenc']
        model = GeneralNetPosEnc(ks=[config['nx'], enc_dim, 20, 20, 1])
    elif model_str == 'bunny':
        model = GeneralNetBunny(act='sin')
    else:
        raise ValueError(f'model not specified properly in config: {config["model"]}')
    return model