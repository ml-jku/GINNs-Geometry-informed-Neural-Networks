from calendar import c
import os
import random

import numpy as np
import torch

from models.NN import ConditionalGeneralNet, ConditionalGeneralResNet, GeneralNet, GeneralNetBunny, GeneralNetPosEnc, GeneralResNet
from models.lip_ffn import LipschitzConditionalFFN
from models.lip_mlp import CondLipMLP
from models.lip_siren import CondLipSIREN
from models.siren import ConditionalSIREN, LatentModulatedSiren
from models.wire import ConditionalWIRE
from models.wire_density import ConditionalWIREDensity
from util.model_utils import get_activation


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_and_true(key, config):
    return (key in config) and config[key]

def set_else_default(key, config, val):
    """Returns the value in config if specified, otherwise returns a default value"""
    return config[key] if key in config else val

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

def t_(x):
    return torch.tensor(x, dtype=torch.float32)

def get_problem_sampler(config):
    if config['problem'] == 'obstacle':
        from GINN.problems.problem_obstacle import ProblemObstacle
        return ProblemObstacle(config)
    elif config['problem'] == 'simjeb':
        from GINN.problems.problem_simjeb import ProblemSimjeb
        return ProblemSimjeb(config)
    else:
        raise ValueError(f'Unknown problem type: {config["problem_type"]}')


def get_model(config):

    model_str = config['model']
    activation = get_activation(config.get('activation', None))
    layers = config['layers']
    layers.insert(0, config['nx'] + config['nz'])  ## input layer
    layers.append(1)  ## output layer

    if model_str == 'siren':
        raise ValueError('SIREN is not supported yet; the layers and in_features are not well defined')
        # model = SIREN(layers=layers, in_features=config['nx'], out_features=1, w0=config['w0'], w0_initial=config['w0_initial'])
    elif model_str == 'cond_siren':
        model = ConditionalSIREN(layers=layers, w0=config['w0'], w0_initial=config['w0_initial'])
    elif model_str == 'lip_siren':
        model = CondLipSIREN(layers=layers, nz=config['nz'], w0=config['w0'], w0_initial=config['w0_initial'])
    elif model_str == 'lip_mlp':
        model = CondLipMLP(layers=layers)
    elif model_str == 'comod_siren':
        model = LatentModulatedSiren(layers=layers, w0=config['w0'], w0_initial=config['w0_initial'], latent_dim=config['nz'])
    elif model_str == 'cond_wire':
        model = ConditionalWIRE(layers=layers, first_omega_0=config['w0_initial'], hidden_omega_0=config['w0'], scale=config['wire_scale'])
    elif model_str == 'cond_wire_density':
        model = ConditionalWIREDensity(layers=layers, first_omega_0=config['w0_initial'], hidden_omega_0=config['w0'], scale=config['wire_scale'], density_iso_level=config['density_iso_level'])
    elif model_str == 'general_net':
        model = GeneralNet(ks=config['layers'], act=activation)
    elif model_str == 'general_resnet':
        model = GeneralResNet(ks=config['layers'], act=activation)
    elif model_str == 'cond_general_net':
        model = ConditionalGeneralNet(ks=config['layers'], act=activation)
    elif model_str == 'cond_general_resnet':
        model = ConditionalGeneralResNet(ks=config['layers'], act=activation)
    elif model_str == 'cond_ffn':
        model = ConditionalFFN(layers=layers, nz=config['nz'], n_ffeat=config['n_ffeat'], sigma=config['ffn_sigma'])
    elif model_str == 'lip_ffn':
        model = LipschitzConditionalFFN(layers=layers, nz=config['nz'], n_ffeat=config['n_ffeat'], sigma=config['ffn_sigma'])
    elif model_str == 'general_net_posenc':
        enc_dim = 2*config['nx']*config['N_posenc']
        model = GeneralNetPosEnc(ks=[config['nx'], enc_dim, 20, 20, 1])
    elif model_str == 'bunny':
        model = GeneralNetBunny(act='sin')
    else:
        raise ValueError(f'model not specified properly in config: {config["model"]}')
    return model