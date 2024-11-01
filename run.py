

"""
Entry point for starting the training.
Checks and performs global config, including the device and the model.
"""
import os
import wandb

from configs.get_config import compute_config_entries_set_envs_seeds_etc, read_cli_args_and_get_config
from GINN.speed.mp_manager import MPManager
import logging

import torch.multiprocessing as multiprocessing
import torch
from train.ginn_trainer import Trainer
from util.misc import get_model
from util.misc import set_all_seeds

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

    # set up logging
    log_level = config.get('log_level', 'INFO')
    logging.basicConfig(level=log_level.upper())  # e.g. debug, becomes DEBUG
    logging.info(f'Log level set to {log_level}')
    # explicitly set the log level for other modules
    logging.getLogger('PIL').setLevel(logging.INFO)

    # create process pool before CUDA is initialized
    mp_manager = MPManager(config)

    set_all_seeds(config['seed'])


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
    config['wandb_id'] = wandb_id

    trainer = Trainer(config, model, mp_manager)
    trainer.train()
    # trainer.test_plotting()

if __name__ == '__main__':
    main()