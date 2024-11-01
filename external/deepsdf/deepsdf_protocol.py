import time
import argparse
import trimesh
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import external.mesh_to_sdf as mesh_to_sdf

class DeepSDFProtocol:
    def __init__(self, n_points_surface=250000, n_points_sphere_target=25000) -> None:
        self.objdir = 'TODO_path'
        self.target_dir = 'TODO_path'
       
        ## print current directory
        print(f'os.getcwd()', os.getcwd())
       
        self.center_for_translation = np.load('GINN/simJEB/data/center_for_translation.npy')
        self.scale_factor = np.load('GINN/simJEB/data/scale_factor_deepSDF.npy')
        self.var1 = 0.0025  # according to DeepSDF
        self.var2 = 0.00025  # according to DeepSDF
        self.n_points_surface = n_points_surface
        self.n_points_sphere_target = n_points_sphere_target
        self.n_points = n_points_surface * 2 + n_points_sphere_target
        self.n_points_sphere = n_points_sphere_target * 2  # 2 ~= 6/pi is the ratio of volume of unit sphere to volume of unit cube
        self.n_total_points = self.n_points_surface * 2 + self.n_points_sphere_target

    def deepsdf_protocol_single_mesh(self, file, do_save=False):
        start_t = time.time()
        print(f'working in file {file} ...')
        
        mesh_path = os.path.join(self.objdir, file)
        pts_fp = os.path.join(self.target_dir, file + '_' + str(self.n_points) + '_points.npy')
        sdf_fp = os.path.join(self.target_dir, file + '_' + str(self.n_points) + '_sdf.npy')
        
        if do_save:
            if os.path.exists(pts_fp):
                print('file already exists', pts_fp)
                return False, file
        
        mesh = trimesh.load(mesh_path)
        # center and scale
        mesh.apply_translation(-self.center_for_translation)
        mesh.apply_scale(1/self.scale_factor)
        
        sign_method = 'depth'
        all_points, sdf_values = mesh_to_sdf.sample_sdf_near_surface(mesh, number_of_points=self.n_total_points, sign_method=sign_method, surface_point_method='sample', surface_pts_ratio=4/5)
        print(f'time needed to get points for {file} is {time.time() - start_t:0.1f} seconds')
        
        # # sample surface points from mesh and perturb
        # points_1 = trimesh.sample.sample_surface(mesh, self.n_points_surface)[0] + np.random.normal(size=(self.n_points_surface, 3), scale=np.sqrt(self.var1))
        # points_2 = trimesh.sample.sample_surface(mesh, self.n_points_surface)[0] + np.random.normal(size=(self.n_points_surface, 3), scale=np.sqrt(self.var2))
        
        # sphere_points = np.random.uniform(-1, 1, size=(self.n_points_sphere, 3))
        # sphere_points = sphere_points[np.linalg.norm(sphere_points, axis=1) < 1]
        
        # all_points = np.concatenate([points_1, points_2, sphere_points], axis=0)
        
        # print(f'time needed to get points for {file} is {time.time() - start_t:0.1f} seconds')
        
        # start_t = time.time()
        # # get sdf values
        # sdf_values = (-1) * mesh.nearest.signed_distance(all_points)  ## negative because k3d returns negative values if outside the mesh
        # print(f'time needed for sdf extraction for {file} is {time.time() - start_t:0.1f} seconds')
        
        if do_save:
            np.save(pts_fp, all_points)
            np.save(sdf_fp, sdf_values)
        
        return True, file

    def extract_from_all_meshes(self, n_threads=1, do_first_n=-1, do_save=False, id_list=None):        
        
        obj_files = [f for f in os.listdir(self.objdir) if f.endswith('.obj')]
        obj_files = sorted(obj_files)
        if do_first_n > 0:
            obj_files = obj_files[:do_first_n]
        
        # filter by id_list
        if id_list is not None:
            # e.g. '0.obj', '10.obj', '101.obj',
            obj_files = [f for f in obj_files if int(f.split('.')[0]) in id_list]
            print(f'id_list: {id_list}')
        
        if n_threads <= 1:
            if len(obj_files) >= 10:
                print('WARNING: n_threads <= 1 but len(obj_files) >= 10')
            for file in obj_files:
                print('handling file', file)
                self.deepsdf_protocol_single_mesh(file, do_save)
        
        
        with ThreadPoolExecutor(max_workers=int(n_threads)) as executor:
            results = []
            for file in obj_files:
                results.append(executor.submit(self.deepsdf_protocol_single_mesh, file, do_save))
            print('jobs submitted')
            for future in as_completed(results):
                print(future.result())
        print('done')
            
if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_surf', type=int, default=250000)
    parser.add_argument('--n_sphere', type=int, default=25000)
    parser.add_argument('--n_threads', type=int, default=1)
    parser.add_argument('--do_first_n', type=int, default=-1)
    parser.add_argument('--id_list', type=str, default=None)
    
    args = parser.parse_args()    
    
    id_list = None
    if args.id_list is not None:
        # remove [ and ] from the string
        id_list = args.id_list.split(',')
        id_list = [int(x) for x in id_list]
    
    deepsdf_protocol = DeepSDFProtocol(n_points_surface=args.n_surf, n_points_sphere_target=args.n_sphere)
    deepsdf_protocol.extract_from_all_meshes(n_threads=args.n_threads, do_first_n=args.do_first_n, do_save=True, id_list=id_list)
