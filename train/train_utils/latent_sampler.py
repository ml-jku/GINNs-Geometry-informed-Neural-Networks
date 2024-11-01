import math
import re
import torch
from einops import repeat

'''
Sampling protocol:
- sample_z are latents for GINN without data
    - excludes the corners of the interval, to avoid training on data AND GINN
- get_z_corners are the latents for the provided data

'''


def _linspace_excluding_ends(start, end, steps):
    # Generate a tensor with steps + 2 values to include both start and end
    full_tensor = torch.linspace(start, end, steps + 2)
    # Exclude the first and last elements
    return full_tensor[1:-1]

def sample_z(config, epoch, previous_z, is_validation=False):
    assert 'reset_zlatents_every_n_epochs' in config, 'reset_zlatents_every_n_epochs must be defined in config'
    interv = config['z_sample_interval']
    
    if is_validation:
        if config['nz'] == 1:
            z = torch.linspace(interv[0], interv[1], config['valid_plot_n_shapes'])[:, None]
        elif config['nz'] == 2:
            # plot as square grid
            # returns e.g.
            # [[[0.0000, 0.0000],
            #  [0.0000, 0.5000],
            #  [0.0000, 1.0000]],
            # [[0.5000, 0.0000],
            #  [0.5000, 0.5000],
            #  [0.5000, 1.0000]],
            # [[1.0000, 0.0000],
            #  [1.0000, 0.5000],
            #  [1.0000, 1.0000]]]
            n = int(math.sqrt(config['valid_plot_n_shapes']))
            assert n * n == config['valid_plot_n_shapes'], 'valid_plot_n_shapes must be a square number'
            z = torch.linspace(interv[0], interv[1], n)
            z = torch.cartesian_prod(z, z)
        else:
            z = torch.rand(size=(config['valid_plot_n_shapes'], config['nz'])) * (interv[1] - interv[0]) + interv[0]
        return z
   
    ## return previous z if not time to reset
    if previous_z is not None and epoch % config['reset_zlatents_every_n_epochs'] != 0:
        return previous_z
   
    ## get training z
    if config['z_sample_method'] == 'equidistant':
        '''
        for training, exclude the corners of the interval
        for validation, include them
        e.g. if nz=1, ginn_bsize=4, z_sample_interval=[-1, 1]
            training z is [-1, -0.33, 0.33, 1]
            validation z are the middle points [-0.66, 0, 0.66]
        '''
        # assert config['nz'] == 1, 'nz must be 1 for linspace z_sample_method'
        
        if config['nz'] == 1:
            z = _linspace_excluding_ends(config['z_sample_interval'][0], config['z_sample_interval'][1], config['ginn_bsize'])[:, None]
            return z
        elif config['nz'] == 2:
            n = int(math.sqrt(config['ginn_bsize']))
            assert n * n == config['ginn_bsize'], 'ginn_bsize must be a square number'
            z = _linspace_excluding_ends(config['z_sample_interval'][0], config['z_sample_interval'][1], n)
            z = torch.cartesian_prod(z, z)
            return z
        else:
            raise ValueError(f'z_sample_method equidistant not supported for nz > 2')
            

    if config['z_sample_method'] == 'uniform':
        z = torch.rand(size=(config['ginn_bsize'], config['nz'])) * (interv[1] - interv[0]) + interv[0]
        return z
        
    if config['z_sample_method'] == 'normal':
        z = torch.randn(size=(config['ginn_bsize'], config['nz']))
        mu, var = config['z_mean_var'][0], config['z_mean_var'][1]
        z = z * math.sqrt(var) + mu
        return z
    
    raise ValueError(f'z_sample_method {config["z_sample_method"]} not supported')

def get_z_corners(config):
    assert len(config['simjeb_ids']) <= 2, "simjeb_ids must be <= 2"
    
    z_corners = torch.tensor(config['z_sample_interval'])[:len(config['simjeb_ids'])]
    z_corners = repeat(z_corners, 'n -> n d', d=config['nz'])
    return z_corners