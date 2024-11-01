import os
import numpy as np
import trimesh

from util.const import SIMJEB_MESHES_DIR


def handle_single_shape(mesh, name):
    
    # compute sdf, sample points and save them
    
    # compute sdf
    pass


def get_obj_files_in_dir(directory):
    files = os.listdir(directory)
    obj_files = [f for f in files if f.endswith('.obj')]
    return obj_files

def get_obj_id(file_path):
    return file_path.split('/')[-1].split('.')[0]

def preprocess_meshes(directory):
    
    # scale and center
    scale_factor = np.load('derived/scale_factor.npy')
    center_for_translation = np.load('derived/center_for_translation.npy')
    
    obj_files = get_obj_files_in_dir(directory)
    for obj_file in obj_files:
        mesh = trimesh.load(obj_file)        
        # center and scale the mesh
        mesh.apply_translation(-center_for_translation)
        mesh.apply_scale(1/scale_factor)
        
        handle_single_shape(mesh, get_obj_id(obj_file))
        
        
if __name__ == '__main__':
    preprocess_meshes(SIMJEB_MESHES_DIR)