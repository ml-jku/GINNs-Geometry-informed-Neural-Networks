"""
Entry point for starting the training.
Checks and performs global config, including the device and the model.
"""
import datetime
import os
from pathlib import Path

import torch
import wandb

from configs.get_config import compute_config_entries_set_envs_seeds_etc, read_cli_args_and_get_config
from GINN.helpers.mp_manager import MPManager
from train.ginn_trainer import Trainer
from utils import get_model, get_model_path_via_wandb_id_from_fs, set_and_true
import logging


def main():

    # read CLI args
    config = read_cli_args_and_get_config()
    compute_config_entries_set_envs_seeds_etc(config)

    # FAIL FAST
    # deprecated config entries
    assert 'source_points_from' not in config, 'FAIL FAST: source_points_from is not supported anymore'
    # consistency
    plot_show = config.get('fig_show', False)
    fig_save = config.get('fig_save', False)
    wandb_plot = config.get('fig_wandb', False)
    assert sum([plot_show, fig_save, wandb_plot]) <= 1, "Only one of fig_show, fig_save, or fig_wandb can be True"


    # create process pool before CUDA is initialized
    mp_manager = MPManager(config)

    # set default device after cuda visibility has been configured in compute_config
    if torch.cuda.is_available():
        # print('set cuda as default device')
        print('Visible CUDA devices:')
        for i_cuda in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i_cuda).name)
        device = 'cuda'
        torch.set_default_device(device)
        config['device'] = device
    else:
        print('CUDA not available - proceeding on CPU')

    ## Create model
    model = get_model(config)
    
    ## Load model weights
    if set_and_true('load_model', config):
        assert 'model_load_path' in config or 'model_load_wandb_id' in config, 'model_load_path or model_load_wandb_id must be specified if load_model is True'
        
        if 'model_load_wandb_id' in config:
            assert 'model_load_path' not in config, 'model_load_path and model_load_wandb_id cannot be specified at the same time'
            model_load_path = get_model_path_via_wandb_id_from_fs(config['model_load_wandb_id'])
            logging.info(f'Loading model from {model_load_path}')
            model.load_state_dict(torch.load(model_load_path))
        else:
            ## Load model from path
            model.load_state_dict(torch.load(config['model_load_path']))
    if not set_and_true('load_model', config) and ('model_load_path' in config or 'model_load_wandb_id' in config):
        logging.warning('model_load_path or model_load_wandb_id specified but load_model is False. Ignoring.')

    # NOTE: to disable wandb set the ENV
    # "WANDB_MODE": "disabled"
    print(f"WANDB_MODE: {os.getenv('WANDB_MODE')}")
    wandb_save_dir = os.getenv('WANDB_SAVE_DIR', 'wandb')
    print(f'wandb_save_dir: {wandb_save_dir}')
    wandb.init(entity=config['wandb_entity_name'],
               project=config['wandb_project_name'],
               name=config["wandb_experiment_name"],
               dir=wandb_save_dir,
               config=config)
    
    wandb_id = 'no_wandb_id'
    if os.getenv('WANDB_MODE') != 'disabled':
        wandb_id = wandb.run.id
        
    model_save_path = config.get('model_save_path', '_saved_models')   
    model_parent_path = os.path.join(model_save_path, config['model'])
    if not os.path.exists(model_parent_path):
        # os.mkdir(model_parent_path)
        Path(model_parent_path).mkdir(parents=True, exist_ok=True)
    model_filename = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + '-' + wandb_id + '.pth'
    config['model_path'] = os.path.join(model_parent_path, model_filename)

    trainer = Trainer(config, model, mp_manager)
    trainer.train()
    # trainer.test_plotting()

if __name__ == '__main__':
    main()