import datetime

from jeb_meter import JebMeter
from notebooks.notebook_utils import get_mesh_for_latent
import torch
import numpy as np
import trimesh
from tqdm import tqdm
import pandas as pd

from simple_obst_meter import SimpleObstacleMeter
from utils import get_activation, get_model, get_model_path_via_wandb_id_from_fs, get_stateless_net_with_partials
from visualization.utils_mesh import get_2d_contour_for_latent

class Evaluator:

    def __init__(self) -> None:
        self.jeb_meter = JebMeter('../GINN/simJEB')
        self.sim_obst_meter = SimpleObstacleMeter()
        
    def run_architecture(self, config):

        ## Parameters        
        mc_resolution = 256
        device = 'cpu'
        torch.set_default_device(device)

        model_path = get_model_path_via_wandb_id_from_fs(config['key'], use_epoch=config.get('iteration', None))
        z_latents = config['z_latents']

        ## MODEL
        if config['problem'] == 'jeb':
            bounds = torch.from_numpy(np.load('../GINN/simJEB/derived/bounds.npy')).float()
            activation = get_activation(config.get('activation', None))
            model = get_model(config, activation, 'cond_siren').to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
        elif config['problem'] == 'sim_obst':
            bounds =  torch.tensor([[-1, 1],[-0.5, 0.5]])
            activation = get_activation('softplus')
            model = get_model(config, activation, 'cond_general_resnet').to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
        params, f, vf_x, vf_xx = get_stateless_net_with_partials(model, use_x_and_z_arg=True)

        print(f'Retrieved model with key {config["key"]} and z_latents {z_latents}')

        shape_dicts = []
        mesh_list = []
        for z in tqdm(z_latents): ## do marching cubes for every z
            if config['problem'] == 'jeb':
                verts, faces = get_mesh_for_latent(f, params, z, bounds, mc_resolution, device, chunks=1, watertight=True)
                print(f'Found a mesh with {len(verts)} vertices and {len(faces)} faces for latent z={z}')
                cur_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                mesh_list.append(cur_mesh)
                metrics_dict = self.jeb_meter.get_all_metrics_as_dict(cur_mesh)
            elif config['problem'] == 'sim_obst':
                contour_model = get_2d_contour_for_latent(f, params, z, bounds, mc_resolution, device, chunks=1, watertight=False)
                mesh_list.append(contour_model)
                metrics_dict = self.sim_obst_meter.get_all_metrics_as_dict(contour_model)
            # metrics_dict = {}
            print(f'metrics_dict: {metrics_dict}')
            mesh_dict = config.copy()
            mesh_dict.update(metrics_dict)
            mesh_dict['z_latent'] = z
            shape_dicts.append(mesh_dict)
            print(f'Added metrics for latent z={z}')
        
        if len(mesh_list) > 1:
            if config['problem'] == 'jeb':
                diversity_chamfer = self.jeb_meter.diversity_chamfer(mesh_list, n_samples=20000)
            elif config['problem'] == 'sim_obst':
                diversity_chamfer = self.sim_obst_meter.diversity_chamfer(mesh_list)
            else:
                raise ValueError(f'Unknown problem: {config["problem"]}')
            for shape_dict in shape_dicts:
                shape_dict['diversity_chamfer'] = diversity_chamfer        
        
        return shape_dicts

    def run_all(self):
                
        models = [
            # {'key': 'crn6sqnh', 'problem': 'jeb', 'iteration': 1500, 'z_latents': torch.linspace(-0.1, 0.1, 4)[:,None], 'comment': 'GOOD MODEL', 'activation': 'sin', 'layers': [4, 256, 256, 256, 256, 256, 1], 'w0': 1, 'w0_initial': 8.0},
            # {'key': 'ibeyd2qk', 'problem': 'jeb', 'z_latents': torch.tensor([[-0.1]]), 'comment': 'GOOD without diversity; single', 'activation': 'sin', 'layers': [4, 256, 256, 256, 256, 256, 1], 'w0': 1, 'w0_initial': 8.0},
            # {'key': 'np5nr2ye', 'problem': 'jeb', 'z_latents': torch.eye(16)[0][None,:], 'layers': [19, 256, 256, 256, 256, 256, 1], 'w0': 1.0, 'w0_initial': 6.5, 'comment': 'FAIL: "w0_initial too low (w0_initial=6.5)"'},
            # {'key': '6tuzjybp', 'problem': 'jeb', 'z_latents': torch.tensor([[-0.1]]) , 'comment': 'FAIL: "SCC not enabled"', 'activation': 'sin', 'layers': [4, 256, 256, 256, 256, 256, 1], 'w0': 1, 'w0_initial': 8.0},
            # {'key': 'yzlpb8jy', 'problem': 'jeb', 'z_latents': torch.linspace(-0.1, 0.1, 5)[:,None], 'comment': 'FAIL: "Interpolation does not work"', 'activation': 'sin', 'layers': [4, 256, 256, 256, 256, 256, 1], 'w0': 1, 'w0_initial': 8.0},
            
            {'key': 'pjdljhfa', 'problem': 'jeb', 'z_latents': torch.linspace(-0.1, -0.0333, 5)[:,None], 'comment': 'GOOD MODEL', 'activation': 'sin', 'layers': [4, 256, 256, 256, 256, 256, 1], 'w0': 1, 'w0_initial': 8.0},
            
            # {'key': 'am8w2d8s', 'problem': 'sim_obst', 'z_latents': torch.linspace(-1, 1, 15)[:,None], 'comment': '16 shapes WITH diversity that separates the shapes above and below the obstacle', 'layers': [3, 512, 512, 512, 512, 1], 'activation': 'softplus'},
            # {'key': 'xpjzunbz', 'problem': 'sim_obst', 'z_latents': torch.linspace(-1, 1, 15)[:,None], 'comment': '16 shapes WITHOUT diversity (same HPs as am8w2d8s)', 'layers': [3, 512, 512, 512, 512, 1], 'activation': 'softplus'},
            # {'key': 'flyufwb8', 'problem': 'sim_obst', 'z_latents': torch.tensor([[-1]]), 'comment': 'Single shape, WITH SCC', 'layers': [3, 512, 512, 512, 512, 1], 'activation': 'softplus'},
            # {'key': 'mz5y1x3w', 'problem': 'sim_obst', 'z_latents': torch.tensor([[-1]]), 'comment': 'Single shape, WITHOUT SCC', 'layers': [3, 512, 512, 512, 512, 1], 'activation': 'softplus'},
        ]
        
        list_of_mesh_dicts = []
        
        for arch_dict in models:
            mesh_dict_list_for_arch = self.run_architecture(arch_dict)
            list_of_mesh_dicts.extend(mesh_dict_list_for_arch)
        
        # visualize as table
        df = pd.DataFrame(list_of_mesh_dicts)
        # export as excel
        date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_excel(f'GINN/simJEB/{date_str}_metrics.xlsx', index=False)

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_all()