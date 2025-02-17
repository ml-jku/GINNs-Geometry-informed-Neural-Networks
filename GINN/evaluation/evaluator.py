import os
import datetime

from train.train_utils.latent_sampler import sample_z
from util.checkpointing import get_model_path_via_wandb_id_from_fs, load_yaml_and_drop_keys
from util.const import MODELS_PARENT_DIR, MESHES_PARENT_DIR

from jeb_meter import JebMeter
from util.formatting import format_significant_digits
from util.misc import get_model
from util.visualization.utils_mesh import get_watertight_mesh_for_latent
import torch
import numpy as np
import trimesh
from tqdm import tqdm
import pandas as pd

from simple_obst_meter import SimpleObstacleMeter
from util.misc import get_stateless_net_with_partials
from util.visualization.utils_mesh import get_2d_contour_for_latent

SAVE_PLOTS = False
save_plots_dir = 'plots'
mc_resolution = 256
device = 'cpu'
# keys_to_keep = ['description', 'nz', 'z_sample_interval', 'z_sample_method', 'n_shapes', 'ginn_bsize', 'reset_zlatents_every_n_epochs']

class Evaluator:

    def __init__(self) -> None:
        self.common_config = {
            'simjeb_root_dir': 'GINN/simJEB/data',
            'output_dir': 'evaluation/metrics',
            'device': device,
            'mc_resolution': mc_resolution,
            'metrics_diversity_inner_agg_fns': ['mean'],
            'metrics_diversity_outer_agg_fns': ['mean'],
            'metrics_diversity_ps': [0.5],
            'metrics_chamfer_orders': [2],
            }
        
        torch.set_default_device(self.common_config['device'])
        
        self.jeb_config = self.common_config.copy()
        self.jeb_config.update({
            'bounds': torch.tensor(np.load(os.path.join(self.common_config['simjeb_root_dir'], 'bounds.npy'))),
            'metrics_diversity_n_samples': 10000,
            'surf_pts_nof_points': 10000,
            'nx': 3,
            'surf_pts_nof_points': 32768, # 32768  ## nof points for initializing the flow to surface
            })
        self.jeb_meter = JebMeter(self.jeb_config)
        
    def run_architecture(self, config):
        device = self.common_config['device']

        run_id = config['run_id']
        n_shapes = config['n_shapes']
        use_validation_z = n_shapes > 1  # if there is only one shape, use the training z; otherwise do validation interpolation
        
        # Parameters
        model_path = get_model_path_via_wandb_id_from_fs(run_id, MODELS_PARENT_DIR)

        # overwrite some parameters
        config['valid_plot_n_shapes'] = n_shapes  ## number of shapes to generate
        config['z_sample_method'] = 'equidistant'  ## equidistant sampling for plots
        config['layers'] = config['layers'][1:-1] ## remove input and output layer; this is wrongly saved in the config

        # overwrite some parameters in jeb_meter
        self.jeb_meter.config['ginn_bsize'] = n_shapes  ## number of shapes to generate

        ## MODEL
        model = get_model(config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        netp = get_stateless_net_with_partials(model, config['nz'])
        params, f, vf_x, vf_xx = netp.params, netp.f_, netp.vf_x_, netp.vf_xx_
        z_latents = sample_z(config, epoch=0, previous_z=None, is_validation=use_validation_z)

        print(f'z_latents: {z_latents}')

        mesh_list = []
        for z in tqdm(z_latents): ## do marching cubes for every z
            verts, faces = get_watertight_mesh_for_latent(netp.f_, netp.params, z, self.jeb_config['bounds'], self.jeb_config['mc_resolution'], device, chunks=1)
            print(f'Found a mesh with {len(verts)} vertices and {len(faces)} faces for latent z={z}')
            mesh_list.append((verts, faces))
        
        metrics_dict = self.jeb_meter.get_average_metrics_as_dict(mesh_list, netp, z_latents)
        
        print(f'metrics_dict: {metrics_dict}')
        return metrics_dict

    def run_all(self):
        
        models = [
            # GINN     
            dict(run_id='cevzjdar', description='no eikonal', n_shapes=1),
            dict(run_id='21ocqlyh', description='no connectedness', n_shapes=1),
            dict(run_id='z79aqfo8', description='no smoothness', n_shapes=1),
            dict(run_id='32sr1pvk', description='w/ log curvature', n_shapes=1),
            dict(run_id='3xq707nt', description='base (single)', n_shapes=1),
            
            # # dict(run_id='mlzgb41l', description='wire default params', n_shapes=1),
            # # gGINN
            dict(run_id='zp99scot', description='no diversity', n_shapes=9),
            dict(run_id='tr8kyxcw', description='z equidistant', n_shapes=9),
            dict(run_id='hcw86uql', description='base (multi)', n_shapes=9),
        ]

        list_of_metrics_dicts = []
        
        for config in models:
            print(f'Running model with key {config["run_id"]}')
            
            config_path = get_model_path_via_wandb_id_from_fs(config['run_id'], MODELS_PARENT_DIR, get_file='config.yml')
            config_from_file = load_yaml_and_drop_keys(config_path, keys_to_drop=['bounds'])
            config.update(config_from_file)
            # try:
            avg_metrics_dict = self.run_architecture(config)
            # except Exception as e:
            #     print(f'Failed to run model with key {config["run_id"]} with error: {e}')
            #     continue
            
            ## for each key that ends with _avg, find the _std and merge them to a +/- key
            ## if std is 0, there is no need to show it as there was just one latent
            for key in list(avg_metrics_dict.keys()):
                if key.endswith('_avg'):
                    base_key = key.replace('_avg', '')
                    std_key = key.replace('_avg', '_std')
                    ## only show the std if there was more than one latent
                    has_std = config['n_shapes'] > 1
                    entry_str = combine_avg_and_std(avg_metrics_dict[key], avg_metrics_dict[std_key], has_std=has_std)
                    ## remove _avg and _std keys
                    avg_metrics_dict.pop(key)
                    avg_metrics_dict.pop(std_key)
                    ## add the new entry
                    avg_metrics_dict[base_key] = entry_str
                    
                if key.startswith('diversity_chamfer'):
                    ## round to 2 significant digits
                    avg_metrics_dict[key] = format_significant_digits(avg_metrics_dict[key], 2)

            # for k in keys_to_keep:
            #     avg_metrics_dict[k] = config[k]
                                
            avg_metrics_dict.update(config)
                                
            list_of_metrics_dicts.append(avg_metrics_dict)
            
        
        # visualize as table
        df = pd.DataFrame(list_of_metrics_dicts)
        ## order columns alphabetically by name
        df = df.reindex(sorted(df.columns), axis=1)
        # export as excel
        date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if not os.path.exists(self.common_config['output_dir']):
            os.makedirs(self.common_config['output_dir'], exist_ok=True)
        df.to_excel(os.path.join(self.common_config['output_dir'], f'{date_str}_metrics.xlsx'), index=False)
        df.to_csv(os.path.join(self.common_config['output_dir'], f'{date_str}_metrics.csv'), index=False)

def combine_avg_and_std(avg, std, has_std=True):
    '''
    :param avg: average value
    :param std: standard deviation
    :param has_std: whether the standard deviation is available; is false if there was only a single run
    '''
    
    print(f'avg: {avg}, std: {std}')
    
    ## if avg is approximately an integer, treat it as an integer
    avg_str = format_significant_digits(avg, 2)
        
    std_str = ''
    if has_std:
        std_str = f' +/- {format_significant_digits(std, 2)}'
        
    ## build the new entry
    entry_str = f'{avg_str}{std_str}'
    return entry_str
    
        
        
if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_all()