import os
import k3d
import numpy as np
import k3d.platonic as platonic

# used for plotting in notebooks with real-sized shapes
default_camera = [41.189719283747884,
    45.70126861241561,
    72.880863507058,
    0.03530847628622623,
    0.136595584342187,
    -0.1740857143117238,
    -0.47532418411386657,
    -0.33378244998736883,
    0.8140369746374706]

default_cameras = dict(

    simjeb = 
    # [138.40081766594227,
    # 29.526415795126038,
    # 106.33893060120508,
    # -0.0001493990421295166,
    # -0.0016707777976989746,
    # 0,
    # -0.5375222895751952,
    # -0.14198113653204303,
    # 0.8312106502439082],
    [0.4257934369809876,
    2.439514758309903,
    2.044894892361516,
    -0.09748320306136546,
    -0.023071586381566395,
    0.18371110525556372,
    -0.2199130736715865,
    -0.5109269672367394,
    0.8310185763137804],

)

def set_camera(fig, camera):
    '''
    Sets up the camera.
    fig: k3d.plot
    camera: list of 9 values describing the position and angle of camera
    '''
    fig.camera_auto_fit = False
    fig.grid_auto_fit = False
    fig.grid_visible = False
    fig.axes_helper = 0 ## Hide axis thingy
    fig.camera_fov = 1
    fig.camera = camera
    
def set_cam_for_wandb(fig, camera):
    fig.camera_auto_fit = False
    fig.camera = camera
    
def make_camera(d=125, alpha=-30, beta=15, rotation=[0,-1,0]):
    '''
    Assumes ground is x-z plane and y is downward like in dualsdf.
    TODO: might want to add axis permutations/flips.
    '''
    alpha = np.deg2rad(alpha)
    beta  = np.deg2rad(beta)
    x =  d*np.cos(beta)*np.cos(alpha)
    y = -d*np.sin(beta)
    z =  d*np.cos(beta)*np.sin(alpha)
    camera = [
            x, y, z,  ## Camera position
            0, 0, 0,  ## Object center
            rotation[0], rotation[1], rotation[2]  ## Rotation
            ]
    return camera


def dummy_fig(height=1000, meshcolor=0xdddcdc, dummy_mesh=True):
    '''Dummy figure with dummy mesh.'''
    fig = k3d.plot(height=height, axes_helper=0)
    if dummy_mesh:
        cube = platonic.Cube()
        faces = cube.indices + list(reversed(cube.indices))
        fig.mesh = k3d.mesh(cube.vertices, faces, color=meshcolor, flat_shading=False)
        fig += fig.mesh
    return fig
