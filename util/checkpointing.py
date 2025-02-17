import datetime
import glob
import os
import time

import numpy as np

from models.model_factory import ModelFactory
from util.const import MODELS_PARENT_DIR
from util.misc import is_every_n_epochs_fulfilled

import torch
import yaml
from pathlib import Path


def find_model_file(run_id, root_dir, use_epoch=None, get_file='model.pt', suppress_print=False):
    # search in all subdirectories
    start_t = time.time()
    ## /MODELS_PARENT_DIR/cond_siren/2024_05_11__15_55_28-4lsf3jsk/2024_05_11__15_55_28-4lsf3jsk-it_01850-model.pt
    glob_pattern = os.path.join(root_dir, '*', f'*{run_id}', f'*{run_id}*{get_file}')
    candidate_files = glob.glob(glob_pattern, recursive=True)
    
    if len(candidate_files) == 0:
        print(f"No files found in fast search. Note that for this the pattern must be root_dir/SINGLE_DIR_INBETWEEN/*{run_id}/*{run_id}*{get_file}")
        print(f'e.g.: MODELS_PARENT_DIR/cond_siren/2024_05_11__15_55_28-4lsf3jsk/2024_05_11__15_55_28-4lsf3jsk-it_01850-model.pt')
        print(f'Starting slower search...')
        candidate_files = glob.glob(os.path.join(root_dir, '**' f'*{run_id}', f'*{run_id}*{get_file}'), recursive=True)
        print(f'Finished slower search')
        
    print(f'Time to search for candidates: {time.time() - start_t:.2f}')
    print(f"Found {len(candidate_files)} candidate files") 
       
    assert len(candidate_files) > 0, f"No files found for {run_id} in {root_dir}"
    
    if not suppress_print:
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



def save_model_every_n_epochs(model, optim, sched, config, epoch):
    if not is_every_n_epochs_fulfilled(epoch, config, 'save_every_n_epochs'):
        return False, None

    assert 'model_save_path' in config, 'model_save_path must be specified in config'

    ## create directory, save config only if it does not exist
    if 'save_model_dir' not in config:
        name_stem = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '-' + config['wandb_id']
        model_parent_path = os.path.join(config['model_save_path'], config['model']['model_str'], name_stem)
        config['save_model_dir'] = model_parent_path

        ## create parent directory if it does not exist
        if not os.path.exists(model_parent_path):
            Path(model_parent_path).mkdir(parents=True, exist_ok=True)
            
        # save config as yml
        config_filename = name_stem + '-config.yml'
        config_path = os.path.join(model_parent_path, config_filename)
        # remove all keys that are tensors; important for loading later
        config = {k: v for k, v in config.items() if not (isinstance(v, torch.Tensor) or isinstance(v, np.ndarray))}
        if not os.path.exists(config_path):
            with open(config_path, 'w') as file:
                yaml.dump(config, file)

    model_parent_path = config['save_model_dir']
    name_stem = model_parent_path.split('/')[-1]

    ## add epoch to filename if not overwriting
    if not config['overwrite_existing_saved_model']:
        name_stem += f'-it_{epoch:05}'

    ## save model
    model_filename = name_stem + '-model.pt'
    model_path = os.path.join(model_parent_path, model_filename)
    # torch.save(model.state_dict(), model_path)
    ModelFactory.save(model, model_path)

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


def get_model_path_via_wandb_id_from_fs(run_id, root_dir, use_epoch=None, get_file='model.pt', suppress_print=False):

    try:
        start_t = time.time()
        file_path = find_model_file(run_id, root_dir, use_epoch=use_epoch, get_file=get_file, suppress_print=suppress_print)
        print(f'Found file in {time.time() - start_t:.2f}s at {file_path}')
        return file_path
    except Exception as e:
        print(e)
    raise ValueError(f"Could not find model with {run_id=} anywhere")


def load_model_optim_sched(config, model, optim, sched, device='cuda'):
    ## Load model weights
    if not (config.get('load_model', False) or config.get('load_mos', False)):
        if 'model_load_path' in config or 'model_load_wandb_id' in config:
            print('WARNING: model_load_path or model_load_wandb_id specified but load_model is False. Ignoring.')
        return model, optim, sched

    if 'model_load_wandb_id' in config:
        assert 'model_load_path' not in config, 'model_load_path and model_load_wandb_id cannot be specified at the same time'
        model_load_path = get_model_path_via_wandb_id_from_fs(config['model_load_wandb_id'], root_dir=MODELS_PARENT_DIR)
        print(f'Loading model from {model_load_path}...')
        model.load_state_dict(torch.load(model_load_path, map_location=device)['state_dict'])

        ## Load optimizer and scheduler
        if config.get('load_optimizer', False) or config.get('load_mos', False):
            optim_load_path = model_load_path.replace('-model.pt', '-optim.pt')
            print(f'Loading optimizer from {optim_load_path}...')
            optim.load_state_dict(torch.load(optim_load_path, map_location=device))

        if config.get('use_scheduler', False) or config.get('load_mos', False):
            sched_load_path = model_load_path.replace('-model.pt', '-sched.pt')
            print(f'Loading scheduler from {sched_load_path}...')
            sched.load_state_dict(torch.load(sched_load_path, map_location=device))

    elif 'model_load_path' in config:
        assert config.get('load_optimizer', False) == False, 'load_optimizer is not supported with model_load_path'
        ## Load model from path
        model.load_state_dict(torch.load(config['model_load_path'], map_location=device))
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
        yaml_text = remove_key_from_yaml(yaml_text, key_to_remove)
    config = yaml.safe_load(yaml_text)

    return config