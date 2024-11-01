import argparse
import os
import random
import sys
import kappaconfig as kc
import wandb as wandb
import ast

def read_cli_args_and_get_config():
    parser = argparse.ArgumentParser(description="Parse command-line arguments")
    
    parser.add_argument('kv', nargs='*', help='Key-value pairs in the form key=value')
    args = parser.parse_args()
    kv_pairs = {}
    for kv in args.kv:
        key, value = kv.split('=', 1)
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            pass  # Keep value as a string if it can't be evaluated
        kv_pairs[key] = value
    args = kv_pairs
    
    assert 'yml' in args.keys(), f'fail fast: yml is required to be set via CLI'
    assert 'gpu_list' in args.keys(), f'fail fast: gpu_list is required to be set via CLI'
    assert 'ginn_on' not in args.keys(), f'fail fast: ginn_on is not allowed to be set via CLI as other parts of the config depend on it'
        
    base_yml_path = os.path.join("configs", "base_config.yml")
    yml_path = os.path.join("configs", args['yml'] if args['yml'].endswith('.yml') else args['yml'] + '.yml')
    config = get_config_from_yml(yml_path, base_yml_path)
    
    # overwrites from direct CLI args
    config = update_dict(config, args, print_overwrites=True)
                
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
        config['seed'] = random.randint(20, 1000000)
    # print(f'configure run with seed {config.seed}')
    
    # set group
    if 'wandb_run_group' not in config.keys():
        config['wandb_run_group'] = "g-" + wandb.util.generate_id()
    os.environ["WANDB_RUN_GROUP"] = config['wandb_run_group']
    
    # computed
    if 'gpu_list' not in config:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print(f'No GPUs specified')
    else:
        # config['n_gpus'] = len(config['gpu_list']) if isinstance(config['gpu_list'], list) else 1
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in config['gpu_list']]) if isinstance(config['gpu_list'], list) else str(config['gpu_list'])
        print(f'set CUDA_VISIBLE_DEVICES to {os.environ["CUDA_VISIBLE_DEVICES"]}')

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



if __name__ == "__main__":
    pass