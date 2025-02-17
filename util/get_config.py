import argparse
import os
import numpy as np
import random
import kappaconfig as kc
import wandb as wandb
import ast

def read_cli_args_and_get_config():
    parser = argparse.ArgumentParser(description="Parse command-line arguments")
    
    parser.add_argument('kv', nargs='*', help='Key-value pairs in the form key=value')
    cli_args = parser.parse_args()
    # arg_dict = cli_args_2_nested_dict(cli_args.kv)
    arg_dict = cli_args_2_flat_dict(cli_args.kv)
    
    assert 'yml' in arg_dict.keys(), f'fail fast: yml is required to be set via CLI'
    assert 'gpu_list' in arg_dict.keys(), f'fail fast: gpu_list is required to be set via CLI'
        
    base_yml_path = os.path.join("configs", "base_config.yml")
    yml_path = os.path.join("configs", arg_dict['yml'] if arg_dict['yml'].endswith('.yml') else arg_dict['yml'] + '.yml')
    arg_dict.pop('yml')
    config = get_config_from_ymls_with_native_overwrite(yml_path, base_yml_path, arg_dict)
    # config = get_config_from_yml(yml_path, base_yml_path)
    # # overwrites from direct CLI args
    # config = deep_update(config, arg_dict, print_overwrites=True)
    
    arg_dict.pop('gpu_list')
    config['args_str'] = ' '.join(f'{key}={value}' for key, value in arg_dict.items())

    return config

def get_config_from_yml(yml_path, base_yml_path=None):   
    resolver = kc.DefaultResolver()
    # TODO: could do kc.from_string here to allow to overwrite the logic in the yml file
    hp = resolver.resolve(kc.from_file_uri(yml_path))
    if base_yml_path is not None:
        base_hp = resolver.resolve(kc.from_file_uri(base_yml_path))
        hp = deep_update(base_hp, hp)
    return hp

def deep_update(original, updates, print_overwrites=True):
    for key, value in updates.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            deep_update(original[key], value)
        else:
            if print_overwrites:
                if key in original:
                    print(f'Overwriting {key} in base_config with {value} (was {original[key]})')
                else:
                    pass
                    # print(f'Adding {key} with {value}')
            original[key] = value
    return original

def cli_args_2_flat_dict(args):
    
    res = {}

    for kv in args:
        keys, value = kv.split('=', 1)
        res[keys] = value    
    return res

def cli_args_2_nested_dict(args):
    
    res = {}

    for kv in args:
        keys, value = kv.split('=', 1)
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            pass  # Keep value as a string if it can't be evaluated

        keys = keys.split('.')
        cur_dict = res
        for key in keys[:-1]:
            subdict = cur_dict.get(key, {})
            cur_dict[key] = subdict
            cur_dict = subdict
        cur_dict[keys[-1]] = value
            
    return res

def get_config_from_ymls_with_native_overwrite(yml_path, base_yml_path=None, overwrite_flat_kvs=None):   
    resolver = kc.DefaultResolver()
    
    # read string from yml file
    with open(yml_path, 'r') as file:
        yml_str = file.read()
        
    with open(base_yml_path, 'r') as file:
        base_yml_str = file.read()
    
    yml_str, overwrite_flat_kvs = update_yml_str(yml_str, overwrite_flat_kvs)
    base_yml_str, overwrite_flat_kvs = update_yml_str(base_yml_str, overwrite_flat_kvs)
    assert len(overwrite_flat_kvs) == 0, f'Not all keys were found in the yml files: {overwrite_flat_kvs}'
    
    hp = resolver.resolve(kc.from_string(yml_str))
    if base_yml_path is not None:
        base_hp = resolver.resolve(kc.from_string(base_yml_str))
        hp = deep_update(base_hp, hp)
    return hp


def update_yml_str(yml_str, kvs, indent_spaces=2):
    kvs = kvs.copy()
    lines = yml_str.split('\n')
    new_lines = []
    parent_key_stack = []
    for i in range(len(lines)):
        line = lines[i]
        
        # remove comments
        line = line.split('#')[0]
        
        # check if line is empty or comment
        if not ':' in line:
            new_lines.append(line)
            continue
        
        n_indents = len(line) - len(line.lstrip())
        n_parents = n_indents // indent_spaces
        parent_key_stack = parent_key_stack[:n_parents]
        
        # check if line is a parent, i.e. it does not have a value
        if not line.split(':')[-1].strip():
            parent_key_stack.append(line.split(':')[0].strip())
            new_lines.append(line)
            continue
        
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        serialized_nested_keys = '.'.join(parent_key_stack + [key])
        if serialized_nested_keys in kvs:
            new_line = ' ' * n_indents + key + ': ' + kvs[serialized_nested_keys]
            print(f'replacing line {line} with {new_line}')
            line = new_line
            kvs.pop(serialized_nested_keys)
        
        new_lines.append(line)
    
    new_yml_str = '\n'.join(new_lines)
    return new_yml_str, kvs


def set_cuda_devices(gpu_list):
    if gpu_list is None or gpu_list == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print(f'WARNING: No GPUs specified, running on CPU')
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in gpu_list]) if isinstance(gpu_list, list) else str(gpu_list)
    print(f'set CUDA_VISIBLE_DEVICES to {os.environ["CUDA_VISIBLE_DEVICES"]}')






if __name__ == "__main__":
#     kvs = {'bla2.ginn_on': '0', 'ginn_on': '2'}
#     yml_str = \
# '''
# ginn_on: 1
# bla: 3
# bla2:
#   ginn_on: 1    
# '''
#     new_yml_str = update_yml_str(yml_str, kvs)
#     prin(new_yml_str)
    
    kvs = {'ginn_on': '0', 'latent_sampling.ginn_bsize': '-10'}
    config = get_config_from_ymls_with_native_overwrite('configs/curv_obj_simjeb_wire_multishape.yml', 'configs/base_config.yml', kvs)
    print(config)