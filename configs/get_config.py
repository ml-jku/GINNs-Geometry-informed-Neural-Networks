import argparse
import os
import random
import kappaconfig as kc
import wandb as wandb
from utils import set_all_seeds


def read_cli_args_and_get_config():
    ## TODO: clean up unused arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml", help="The yml config to load", required=True)
    parser.add_argument("--gpu_list", help="Comma-separated list of CUDA devices to use", default=None)
    parser.add_argument("--seed", help="Seed for RNG", type=float, default=None)
    parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=None)
    parser.add_argument("--gradient_clip_val", help="Value for gradient clipping", type=float, default=None)
    parser.add_argument("--max_epochs", help="total number of epochs to train", type=int, default=None)
    parser.add_argument("--warmup_steps", help="number of warmup steps", type=int, default=None)
    parser.add_argument("--decay_steps", help="steps over to decay learning rate", type=int, default=None)
    parser.add_argument("--batch_size", help="Overall batch_size over the configured GPUs", type=int, default=None)
    parser.add_argument("--use_compile", help="Use torch.compile", type=bool, default=False)
    parser.add_argument("--hp_dict", help="Hyperparameters in the form of 'name:value;...;name:value", default=None)
    args = parser.parse_args()
    assert args.yml, "Must provide a yml config file"

    # get config
    overwrite_hps = {
        'gpu_list': [int(vv) for vv in args.gpu_list.split(',')] if args.gpu_list is not None else None,
        'lr': args.lr if (args.lr is not None and args.lr > 0) else None,
        'seed': int(args.seed) if (args.seed is not None and args.seed > 0) else None,
        'batch_size': args.batch_size if (args.batch_size is not None and args.batch_size > 0) else None,
        'use_compile': args.use_compile,
        'gradient_clip_val': args.gradient_clip_val if (args.gradient_clip_val is not None and args.gradient_clip_val > 0) else None,
        'max_epochs': args.max_epochs if (args.max_epochs is not None and args.max_epochs > 0) else None,
        'warmup_steps': args.warmup_steps if (args.warmup_steps is not None and args.warmup_steps > 0) else None,
        'decay_steps': args.decay_steps if (args.decay_steps is not None and args.decay_steps > 0) else None
        }
    # add HPs of hp_dict CLI arg
    if args.hp_dict:
        hp_dict_arg_dict = parse_argument_string(args.hp_dict)
        config = update_dict(overwrite_hps, hp_dict_arg_dict, print_overwrites=False)
    
    base_yml_path = "configs/base_config.yml"
    yml_path = f'configs/{args.yml}'
    config = get_config_from_yml(yml_path, base_yml_path)
    
    # overwrites from direct CLI args
    config = update_dict(config, overwrite_hps, print_overwrites=True)
            
    return config

def get_config_from_yml(yml_path, base_yml_path=None):   
    resolver = kc.DefaultResolver()
    hp = resolver.resolve(kc.from_file_uri(yml_path))
    if base_yml_path is not None:
        base_hp = resolver.resolve(kc.from_file_uri(base_yml_path))
        hp = update_dict(base_hp, hp)
    return hp

def compute_config_entries_set_envs_seeds_etc(config):
    print(f'==============================')
    
    
    # set seed
    if 'seed' not in config.keys():
        config.seed = random.randint(20, 1000000)
    # print(f'configure run with seed {config.seed}')
    set_all_seeds(config.seed)
    
    # set group
    if 'wandb_run_group' not in config.keys():
        config.wandb_run_group = "g-" + wandb.util.generate_id()
    os.environ["WANDB_RUN_GROUP"] = config.wandb_run_group
    
    # computed
    if 'gpu_list' not in config:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        config.n_gpus = len(config.gpu_list)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in config.gpu_list])

    return config

    
def update_dict(hp, kc_hp, print_overwrites=False):
    # update dict recursively
    for k, v in kc_hp.items():
        if v is None:
            # print(f'Not overwriting {k} with value None')
            continue
        if isinstance(v, dict):
            raise NotImplementedError('nested dicts not supported; probably results in error of shallow copy above')
            # if print_overwrites:
            #     print(f'Overwriting {k} with {v}')
            # hp[k] = update_dict(hp.get(k, {}), v)
        else:
            if print_overwrites:
                print(f'Overwriting {k} with {v}')
            hp[k] = v
    return hp


def parse_argument_string(arg_string):
    # Split the string into key-value pairs
    pairs = arg_string.split(';')

    # Initialize an empty dictionary
    arg_dict = {}

    # Iterate over each pair
    for pair in pairs:
        # Split the pair into key and value
        key, value = pair.split(':', 1)

        # Handling for list-like and range values
        if value.startswith('[') and value.endswith(']'):
            # Convert string representation of list to actual list
            value = eval(value)
        elif value.startswith('(') and value.endswith(')'):
            # Convert string representation of tuple to actual tuple
            value = eval(value)
        elif value.lower() == 'true':
            # Convert string "True" to boolean True
            value = True
        elif value.lower() == 'false':
            # Convert string "False" to boolean False
            value = False
        elif value.replace('.', '', 1).isdigit():
            # Convert numeric strings to floats or integers
            value = float(value) if '.' in value else int(value)

        # Add to dictionary
        arg_dict[key] = value

    return arg_dict

if __name__ == "__main__":
    hp_dict_str = 'lr:0.1;batch_size:32;seed:42;gpu_list:[0,1,2,3];is_on:True;range:[-0.5,0.5];tuple:(1,2,3);float:0.5;int:1;bool:False'
    # hp_dict = get_dict_from_hp_dict_argument(hp_dict_str)
    hp_dict = parse_argument_string(hp_dict_str)
    print(hp_dict)