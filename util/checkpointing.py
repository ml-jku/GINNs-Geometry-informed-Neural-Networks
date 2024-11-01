import datetime
import os

from util.const import MODELS_PARENT_DIR
from util.misc import is_every_n_epochs_fulfilled, set_and_true

import torch
import yaml
from pathlib import Path


def find_model_file(run_id, root_dir, use_epoch=None, get_file='model.pt'):
    # search in all subdirectories
    candidate_files = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if run_id in file and get_file in file:
                candidate_files.append(os.path.join(subdir, file))

    if len(candidate_files) > 1:
        print(f"Found {len(candidate_files)} candidate files")
        for file in candidate_files:
            print(file)

    if use_epoch is None:
        ## return file from latest epoch
        ## /MODELS_PARENT_DIR/cond_siren/2024_05_11__15_55_28-4lsf3jsk/2024_05_11__15_55_28-4lsf3jsk-it_01850-model.pt
        def get_epoch_from_filename(file):
            spl = file.split('-')
            for s in spl:
                if 'it_' in s:
                    return int(s.split('_')[-1])
        ## sort by epoch
        sorted_by_epoch = sorted(candidate_files, key=get_epoch_from_filename)
        return sorted_by_epoch[-1]

    else:
        for file in candidate_files:
            if f'it_{use_epoch:05}' in file:
                return file

    raise ValueError(f"Could not find model with {run_id=} in {root_dir=}")


def save_model_every_n_epochs(model, optim, sched, config, epoch):
    if not is_every_n_epochs_fulfilled(epoch, config, 'save_every_n_epochs'):
        return False, None

    assert 'model_save_path' in config, 'model_save_path must be specified in config'

    ## create directory to save models if it does not exist
    if 'save_model_dir' not in config:
        name_stem = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '-' + config['wandb_id']
        model_parent_path = os.path.join(config['model_save_path'], config['model'], name_stem)
        config['save_model_dir'] = model_parent_path

        ## create parent directory if it does not exist
        if not os.path.exists(model_parent_path):
            Path(model_parent_path).mkdir(parents=True, exist_ok=True)
    model_parent_path = config['save_model_dir']
    name_stem = model_parent_path.split('/')[-1]

    ## add epoch to filename if not overwriting
    if not config['overwrite_existing_saved_model']:
        name_stem += f'-it_{epoch:05}'

    ## save model
    model_filename = name_stem + '-model.pt'
    model_path = os.path.join(model_parent_path, model_filename)
    torch.save(model.state_dict(), model_path)

    # save config as yml
    config_filename = name_stem + '-config.yml'
    config_path = os.path.join(model_parent_path, config_filename)
    # remove all keys that are tensors; important for loading later
    config = {k: v for k, v in config.items() if not isinstance(v, torch.Tensor)}
    if not os.path.exists(config_path):
        with open(config_path, 'w') as file:
            yaml.dump(config, file)

    ## save optimizer
    if config.get('save_optimizer', False):
        optim_filename = name_stem + '-optim.pt'
        optim_path = os.path.join(model_parent_path, optim_filename)
        torch.save(optim.state_dict(), optim_path)

        if config.get('use_scheduler', False):
            ## save scheduler (if used)
            sched_filename = name_stem + '-sched.pt'
            sched_path = os.path.join(model_parent_path, sched_filename)
            torch.save(sched.state_dict(), sched_path)

    return True, model_path


def get_model_path_via_wandb_id_from_fs(run_id, base_dir='./', use_epoch=None, get_file='model.pt'):

    dirs = [MODELS_PARENT_DIR]
    if base_dir is not None: dirs.append(base_dir)
    for root_dir in dirs:
        try:
            file_path = find_model_file(run_id, root_dir, use_epoch=use_epoch, get_file=get_file)
            print(f'Found model at {file_path}')
            return file_path
        except Exception as e:
            print(e)
    raise ValueError(f"Could not find model with {run_id=} anywhere")


def load_model_optim_sched(config, model, optim, sched):
    ## Load model weights
    if not (set_and_true('load_model', config) or set_and_true('load_mos', config)):
        if 'model_load_path' in config or 'model_load_wandb_id' in config:
            print('WARNING: model_load_path or model_load_wandb_id specified but load_model is False. Ignoring.')
        return model, optim, sched

    if 'model_load_wandb_id' in config:
        assert 'model_load_path' not in config, 'model_load_path and model_load_wandb_id cannot be specified at the same time'
        model_load_path = get_model_path_via_wandb_id_from_fs(config['model_load_wandb_id'])
        print(f'Loading model from {model_load_path}...')
        model.load_state_dict(torch.load(model_load_path))

        ## Load optimizer and scheduler
        if config.get('load_optimizer', False) or config.get('load_mos', False):
            optim_load_path = model_load_path.replace('-model.pt', '-optim.pt')
            print(f'Loading optimizer from {optim_load_path}...')
            optim.load_state_dict(torch.load(optim_load_path))

        if config.get('use_scheduler', False) or config.get('load_mos', False):
            sched_load_path = model_load_path.replace('-model.pt', '-sched.pt')
            print(f'Loading scheduler from {sched_load_path}...')
            sched.load_state_dict(torch.load(sched_load_path))

    elif 'model_load_path' in config:
        assert config.get('load_optimizer', False) == False, 'load_optimizer is not supported with model_load_path'
        ## Load model from path
        model.load_state_dict(torch.load(config['model_load_path']))
    else:
        raise ValueError('model_load_path or model_load_wandb_id must be specified if load_model is True')


    return model, optim, sched


def load_yaml_and_drop_keys(file_path, keys_to_drop):
    def remove_key_from_yaml(yaml_text, key_to_remove):
        lines = yaml_text.split('\n')
        modified_lines = []
        skip = False

        for line in lines:
            stripped_line = line
            if stripped_line.startswith(key_to_remove + ':'):
                skip = True
            elif skip and stripped_line.startswith('-'):
                continue
            elif skip and stripped_line.startswith('  '):
                continue
            else:
                skip = False
                modified_lines.append(line)

        return '\n'.join(modified_lines)

    # Read the YAML file as text
    with open(file_path, 'r') as file:
        yaml_text = file.read()
    for key_to_remove in keys_to_drop:
        modified_yaml_text = remove_key_from_yaml(yaml_text, key_to_remove)
    config = yaml.safe_load(modified_yaml_text)

    return config