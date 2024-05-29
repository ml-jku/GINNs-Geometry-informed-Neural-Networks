import torch


def sample_new_z(config, is_init=False):
    assert config['nz'] == 1, 'Only nz=1 supported for init_z'
    if is_init:
        init_z = torch.linspace(config['z_sample_interval'][0], config['z_sample_interval'][1], config['batch_size'])[:, None]
        return init_z

    if config['z_sample_method'] == 'uniform':
        interval = config['z_sample_interval']
        z = torch.rand(size=(config['batch_size'], (config['nz'], 1))) * (interval[1] - interval[0]) + interval[0]
        return z[:, None]
    elif config['z_sample_method'] == 'normal':
        return torch.randn(size=(config['batch_size'], (config['nz'], 1)))
    elif config['z_sample_method'] == 'w0-informed':
        interval = [-0.5 / config['w0_initial'], 0.5 / config['w0_initial']]
        return torch.rand(size=(config['batch_size'], config['nz'])) * (interval[1] - interval[0]) + interval[0]
    elif config['z_sample_method'] == 'unit-vector':
        return torch.eye(config['nz'])[:config['batch_size']]
    raise ValueError(f'z_sample_method {config["z_sample_method"]} not supported')