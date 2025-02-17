import argparse
import os
import time
from matplotlib.pylab import f
import numpy as np
import torch
import trimesh
import igl
from GINN.evaluation.metrics_utils import two_sided_chamfer_divergence, compute_integral_curvature, compute_pairwise_chamfer_divergence
from util.const import DISTS_PARENT_DIR, MESHES_PARENT_DIR, SIMJEB_MESHES_DIR
from util.simjeb_ids import filtered_ids

class SimJEBMeter:
    
    def __init__(self, simjeb_root_dir) -> None:
        self.simjeb_root_dir = simjeb_root_dir
        env_path = os.path.join(simjeb_root_dir, '411_for_envelope.obj')
        interface_path = os.path.join(simjeb_root_dir, 'interfaces.stl')
        center_for_translation_path = os.path.join(simjeb_root_dir, 'center_for_translation.npy')
        scale_factor_path = os.path.join(simjeb_root_dir, 'scale_factor.npy')
        bounds_path = os.path.join(simjeb_root_dir, 'bounds.npy')
        
        self.center_for_translation = np.load(center_for_translation_path)
        self.scale_factor = np.load(scale_factor_path)
        # self.bounds = np.load(bounds_path)
        # self.mesh_design_region = self._load_mesh_and_scale_and_center(env_path, self.center_for_translation, self.scale_factor)
        # self.mesh_interface = self._load_mesh_and_scale_and_center(interface_path, self.center_for_translation, self.scale_factor)
    
    def load_mesh_files(self, folder_path, ids, n=None, file_extension='obj', do_center_scale=True):
        meshes = []
        selected_ids = []
        for i, id in enumerate(ids):
            if n is not None and i >= n:
                break
            mesh_path = os.path.join(folder_path, f'{id}.{file_extension}')
            mesh = self._load_mesh_and_scale_and_center(mesh_path, self.center_for_translation, self.scale_factor, do_center_scale)
            meshes.append(mesh)
            selected_ids.append(id)
        return meshes, selected_ids
    
    def _load_mesh_and_scale_and_center(self, mesh_path, center_for_translation, scale_factor, do_center_scale=True):
        mesh = trimesh.load(mesh_path)
        if do_center_scale:
            mesh.apply_translation(-center_for_translation)
            mesh.apply_scale(1/scale_factor)
        return mesh
    
    
if __name__ == '__main__':
    
    n_shapes_default = 14**2
    default_device = 'cuda:0'
    # read CLI args
    args = argparse.ArgumentParser()
    args.add_argument('--mesh_source', type=str)
    args.add_argument('--operation', type=str)
    args.add_argument('--n_shapes', type=int, default=n_shapes_default)
    args.add_argument('--device', type=str, default=default_device)
    args = args.parse_args()
    
    mesh_source = args.mesh_source
    n_shapes = args.n_shapes
    operation = args.operation.split(',')
    device = args.device
    assert mesh_source in ['simjeb', 'ginn']
    assert all([op in ['mmd', 'curvature', 'bounds'] for op in operation])
    
    simjeb_meter = SimJEBMeter(simjeb_root_dir='GINN/simJEB/data')
    # mesh = trimesh.load(os.path.join(SIMJEB_MESHES_DIR, "0.obj"))
    n_shapes = 14**2
    # operation = ['integral_curvature']
    # mesh_source = 'ginn'
    # device = 'cuda:0'
    
    if mesh_source == 'simjeb':
        print(f'Using simjeb meshes')
        meshes, ids = simjeb_meter.load_mesh_files(SIMJEB_MESHES_DIR, filtered_ids, n=n_shapes, file_extension='obj')
    elif mesh_source == 'ginn':
        print(f'Using GINN meshes')
        run_id = 'hcw86uql'
        meshes_dir = os.path.join(MESHES_PARENT_DIR, f'{run_id}_{n_shapes}')
        meshes, ids = simjeb_meter.load_mesh_files(meshes_dir, range(n_shapes), n=n_shapes, file_extension='stl', do_center_scale=False)
    print(len(meshes))
    
    if 'bounds' in operation:
        print(f'Computing bounds')
        bounds = []
        for id, mesh in zip(ids, meshes):
            start_t = time.time()
            min_bounds, max_bounds = mesh.bounds
            bounds.append((min_bounds, max_bounds))
            print(f'Bounds: {min_bounds}, {max_bounds}')
            print(f'Time for mesh {id}: {time.time() - start_t:.2f}')
        bounds = np.array(bounds)
        print(f'Min bounds: {np.min(bounds, axis=0)}')
        print(f'Max bounds: {np.max(bounds, axis=0)}')
    
    if 'mmd' in operation:
        print(f'Computing MMD')
        start_t = time.time()
        dists = compute_pairwise_chamfer_divergence(meshes, 20000, device=device)
        # save to file
        dists_file_path = os.path.join(DISTS_PARENT_DIR, f'dists_{mesh_source}_{n_shapes}.pt')
        print(f'saving to {dists_file_path}')
        if not os.path.exists(DISTS_PARENT_DIR):
            os.makedirs(DISTS_PARENT_DIR)
        torch.save(dists, dists_file_path)
        print(f'Time for dists: {time.time() - start_t:.2f}')
    
    if 'curvature' in operation:
        print(f'Computing integral curvature')
        curvatures = []
        total_areas = []
        curv_per_area = []
        mean_integral_curves = []
        for id, mesh in zip(ids, meshes):
            start_t = time.time()
            integr_curv, total_area, mean_integral_curv = compute_integral_curvature(mesh)    
            curvatures.append(integr_curv)
            total_areas.append(total_area)
            curv_per_area.append(integr_curv / total_area)
            mean_integral_curves.append(mean_integral_curv)
            print(f'Time for mesh {id}: {time.time() - start_t:.2f}')
        print(f'Integral curvature: {np.mean(curvatures):.3f}')
        print(f'Total area: {np.mean(total_areas):.3f}')
        print(f'Curvature per area: {np.mean(curv_per_area):.3f}')
        print(f'Mean mean integral curvature: {np.mean(mean_integral_curves):.3f}')
    
    