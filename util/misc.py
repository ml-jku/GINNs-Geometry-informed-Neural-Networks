import os
import random

import numpy as np
import torch

from models.NN import ConditionalGeneralNet, ConditionalGeneralResNet, GeneralNet, GeneralNetBunny, GeneralNetPosEnc, GeneralResNet
from models.lip_ffn import LipschitzConditionalFFN
from models.lip_mlp import CondLipMLP
from models.lip_siren import CondLipSIREN
from models.net_w_partials import NetWithPartials
from models.siren import ConditionalSIREN, LatentModulatedSiren
from models.wire import ConditionalWIRE
from functools import cmp_to_key


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_is_out_mask(x, bounds):
    out_mask = (x < bounds[:, 0]).any(1) | (x > bounds[:, 1]).any(1)
    return out_mask

def is_every_n_epochs_fulfilled(epoch, config, key):
    if key not in config:
        return False
    is_fulfilled = (epoch % config[key] == 0) or (epoch == config['max_epochs'] - 1)
    return is_fulfilled

def do_plot(config, epoch, key=None):
        """
        Checks if the plot specified by the key should be produced.
        First checks if the output is set: if not, no need to produce the plot.
        Then, checks if the key is set and true.
        If the key is not specified, only checks the global plotting behaviour.
        """
        is_output_set = config['plot']['fig_wandb']

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

def t_(x):
    return torch.tensor(x, dtype=torch.float32)

def get_problem(problem_config, **kwargs):
    if problem_config['problem_str'] == 'obstacle':
        from GINN.problems.problem_obstacle import ProblemObstacle
        return ProblemObstacle(**problem_config, **kwargs)
    elif problem_config['problem_str'] == 'wheel':
        from GINN.problems.problem_wheel import ProblemWheel
        return ProblemWheel(**problem_config, **kwargs)
    elif problem_config['problem_str'] == 'pipes':
        from GINN.problems.problem_pipes import ProblemPipes
        return ProblemPipes(**problem_config, **kwargs)
    elif problem_config['problem_str'] == 'mbb_beam_2d':
        from GINN.problems.problem_mbb_beam_2d import ProblemMBBBeam2d
        return ProblemMBBBeam2d(**problem_config, **kwargs)
    elif problem_config['problem_str'] == 'cantilever_2d':
        from GINN.problems.problem_cantilever_2d import ProblemCantilever2d
        return ProblemCantilever2d(**problem_config, **kwargs)
    elif problem_config['problem_str'] == 'simjeb':
        from GINN.problems.problem_simjeb import ProblemSimjeb
        return ProblemSimjeb(**problem_config, **kwargs)
    else:
        raise ValueError(f'Unknown problem type: {problem_config["problem_str"]}')


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



def get_model(model_str, nx, nz, layers, ny=1, activation=None, w0=None, w0_initial=None, 
              wire_scale=None, n_ffeat=None, ffn_sigma=None, **kwargs):
    
    act = get_activation(activation)
    layers.insert(0, nx + nz)  ## input layer
    layers.append(ny)  ## output layer

    if model_str == 'siren':
        raise ValueError('SIREN is not supported yet; the layers and in_features are not well defined')
        # model = SIREN(layers=layers, in_features=config['nx'], out_features=1, w0=config['w0'], w0_initial=config['w0_initial'])
    elif model_str == 'cond_siren':
        model = ConditionalSIREN(layers=layers, w0=w0, w0_initial=w0_initial, **kwargs)
    elif model_str == 'lip_siren':
        model = CondLipSIREN(layers=layers, nz=nz, w0=w0, w0_initial=w0_initial)
    elif model_str == 'lip_mlp':
        model = CondLipMLP(layers=layers)
    elif model_str == 'comod_siren':
        model = LatentModulatedSiren(layers=layers, w0=w0, w0_initial=w0_initial, latent_dim=nz)
    elif model_str == 'cond_wire':
        model = ConditionalWIRE(layers=layers, first_omega_0=w0_initial, hidden_omega_0=w0, scale=wire_scale, **kwargs)  # kwargs to pass legacy arguments
    elif model_str == 'grid_mock':
        model = ConditionalGridMock(**kwargs)
    elif model_str == 'general_net':
        model = GeneralNet(ks=layers, act=act)
    elif model_str == 'general_resnet':
        model = GeneralResNet(ks=layers, act=act)
    elif model_str == 'cond_general_net':
        model = ConditionalGeneralNet(ks=layers, act=act)
    elif model_str == 'cond_general_resnet':
        model = ConditionalGeneralResNet(ks=layers, act=act)
    elif model_str == 'lip_ffn':
        model = LipschitzConditionalFFN(layers=layers, nz=nz, n_ffeat=n_ffeat, sigma=ffn_sigma)
    elif model_str == 'bunny':
        model = GeneralNetBunny(act='sin')
    else:
        raise ValueError(f'model not specified properly in config: {model}')
    return model



def geometric_mean(input_x, dim=0):
    '''
    Compute the geometric mean of the input tensor along the specified dimension.
    Is numerically stable for large values. For small values, taking the log first might be worse.
    '''
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))

def inverse_geometric_mean(input_x, dim=0):
    '''
    Compute the inverse geometric mean of the input tensor along the specified dimension.
    Is numerically stable for large values. For small values, taking the log first might be worse.
    '''
    log_x = torch.log(input_x)
    return torch.exp(-torch.mean(log_x, dim=dim))

def compute_grad_norm(parameters):
    """
    Computes the norm of the gradients of the parameters.
    Args:
        parameters (iterable): An iterable of torch.Tensor containing the parameters of the model.
    """        
    grads = [param.grad.detach().flatten() for param in parameters if param.grad is not None ]
    grad_norm = torch.cat(grads).norm()
    return grad_norm

def fuzzy_lexsort(keys, tol):
    """
    Perform a lexsort on the keys with a tolerance for equality.

    Parameters:
    keys (list of arrays): The keys to sort by.
    tol (float): The tolerance within which two numbers are considered equal.

    Returns:
    sorted_indices (array): The indices that would sort the keys lexicographically with the given tolerance.
    """
    # Ensure keys are numpy arrays
    keys = [np.asarray(key) for key in keys]

    # Create a custom comparison function
    def fuzzy_compare(a, b):
        if abs(a[1] - b[1]) <= tol:
            return 0
        elif a[1] < b[1]:
            return -1
        else:
            return 1

    # Create a sorted index array for each key
    sorted_indices = np.arange(len(keys[0]))
    for k in reversed(keys):
        sorted_indices_delta = [i for i, x in sorted(enumerate(k[sorted_indices]), key=cmp_to_key(fuzzy_compare))]
        # sorted_indices_delta = sorted(k, key=cmp_to_key(fuzzy_compare))
        sorted_indices = sorted_indices[sorted_indices_delta]
    return np.array(sorted_indices)

def combine_dicts(dicts):
    """
    Combines a list of dictionaries into a single dictionary.
    Args:
        dicts (list): A list of dictionaries.
    Returns:
        combined_dict (dict): The combined dictionary.
    """
    combined_dict = {}
    for d in dicts:
        combined_dict.update(d)
    return combined_dict

def idx_of_tensor_in_list(target_tensor, tensor_list):
    for i, tensor in enumerate(tensor_list):
        if torch.equal(tensor, target_tensor):
            return i
    return -1