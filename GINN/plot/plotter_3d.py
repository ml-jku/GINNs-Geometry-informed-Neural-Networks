import logging
import math
import datetime
import os.path
from pathlib import Path

import einops
import fast_simplification
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from k3d.colormaps import matplotlib_color_maps
import wandb
from PIL import Image
from io import BytesIO
from matplotlib.colors import LogNorm
import warnings

from models.point_wrapper import PointWrapper
from util.vis_utils import generate_color, n_rows_cols
from util.k3d_plot_utils import default_cameras, set_cam_for_wandb
import k3d

# Filter Warning - TODO: maybe fix this and other lines more properly later
# fig += k3d.mesh(verts, faces, group=group, color=MESH_COLOR, side='double', opacity=MESH_OPACITY, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row]) 
warnings.filterwarnings("ignore", message="Given trait value dtype.*")

logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Points
DEFAULT_POINT_SIZE = 0.01
DEFAULT_POINT_OPACITY = 1.0
DEFAULT_POINT_SHADER = 'flat'
DEFAULT_PTS_COLOR = 0x808080
TRAJ_WIDTH = 0.005
## COLORS
MESH_COLOR = 0xe31b23
MESH_OPACITY = 1.0


class Plotter3d():

    def __init__(self, bounds, val_plot_grid, problem_str, **kwargs):
        self.bounds = bounds.cpu().numpy()
        self.val_plot_grid = val_plot_grid
        self.problem_str = problem_str
        self.logger = logging.getLogger('plot_helper')
        self.epoch = 0
        self.meshes = None

    
    def reset_output(self, verts_faces_list, epoch=None):
        '''
        Set the field to plot.
        INFO: The output is not recomputed here, as this plotter must be kept torch-free. 
              Otherwise multiprocessing needs torch context and occupies GPU memory.
        ''' 
        self.meshes = verts_faces_list
        self.epoch = epoch
        
        
        

    def plot_function_and_cps(self, pc: PointWrapper, fig_label, xc_attributes=None, p_backdrop:PointWrapper=None):
        """
        Plot the function and its critical points.
        xc: [N, 2], critical points
        fig_label: string labeling the figure for the output
        xc_attribute: [N], values to use for color-coding the points, e.g. gradient magnitudes
        x_backdrop: [N, 2], points. These are plotted lightly, e.g. to see the points before the operation
        """
        shape_grid = 1.5*(self.bounds[:,1] - self.bounds[:,0]) ## distance between the shape grid for plotting

        if xc_attributes is not None:
            xc_attributes = np.clip(xc_attributes, a_min=1.0e-15, a_max=None, out=xc_attributes)
            cmap = cm.get_cmap('inferno')
            normalizer = LogNorm(vmin=xc_attributes.min(), vmax=xc_attributes.max())
            colors = cmap(normalizer(xc_attributes))
            colors_rgb = (colors[:, :3] * 255).astype(np.uint32)
            colors_rgb_int = colors_rgb[:, 0] + (colors_rgb[:, 1] << 8) + (colors_rgb[:, 2] << 16)
        
        n_rows, n_cols = n_rows_cols(len(self.meshes))
        
        fig = k3d.plot(height=800)#, grid_visible=False, camera_fov=1.0)
        for i_shape in range(len(self.meshes)):
            i_col = (i_shape  % n_cols)
            i_row = (i_shape // n_cols)
            verts, faces = self.meshes[i_shape]
            group = f'Shape {i_shape}'
            shape_idcs = pc.get_idcs(i_shape)
            fig += k3d.mesh(verts, faces, group=group, color=MESH_COLOR, side='double', opacity=MESH_OPACITY, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            
            xc = pc.pts_of_shape(i_shape)
            
            if p_backdrop is not None:
                fig += k3d.points(p_backdrop.data, group=group, point_size=DEFAULT_POINT_SIZE, color=DEFAULT_PTS_COLOR, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            if xc_attributes is not None:                
                fig += k3d.points(xc, group=group, point_size=DEFAULT_POINT_SIZE, colors=colors_rgb_int[shape_idcs], opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            else:
                fig += k3d.points(xc, group=group, point_size=DEFAULT_POINT_SIZE, color=DEFAULT_PTS_COLOR, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
        
        return self._handle_plot_show(fig, fig_label=fig_label)

    def plot_grad_field(self, grad_fields, constraint_pts_dict=None, fig_label="Grad field", attributes=None):
        shape_grid = 1.5*(self.bounds[:,1] - self.bounds[:,0]) ## distance between the shape grid for plotting

        n_rows, n_cols = n_rows_cols(len(self.meshes))

        # take log of the gradient field
        grad_fields = np.abs(grad_fields)
        # take log but preserve the sign
        # grad_fields = np.log10(np.abs(grad_fields)) * np.sign(grad_fields)

        fig = k3d.plot(height=800)#, grid_visible=False, camera_fov=1.0)
        for i_shape in range(len(self.meshes)):
            # fill it up horizontally first, then go to the next row
            i_col = (i_shape  % n_cols)
            i_row = (i_shape // n_cols)
            group = f'Shape {i_shape}'
            fig += k3d.volume(grad_fields[i_shape], group=group, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row], 
                              color_map=k3d.basic_color_maps.Jet, 
                              bounds=self.bounds.flatten(), 
                              alpha_coef=5.0)
            if constraint_pts_dict is not None:
                for i, constraint_name in enumerate(constraint_pts_dict.keys()):
                    constraint_pts = constraint_pts_dict[constraint_name]
                    color = generate_color(i)
                    # in the first epoch, only plot the interface
                    cur_plot= k3d.points(constraint_pts, group=group, point_size=DEFAULT_POINT_SIZE, name=constraint_name, color=color, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, 
                                      translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
                    cur_plot.visible = (self.epoch > 0) or (constraint_name == 'interface')
                    fig += cur_plot
        
        return self._handle_plot_show(fig, fig_label=fig_label)
        

    def plot_shape(self, fig_label='Boundary', constraint_pts_dict={}, is_validation=False):
        shape_grid = 1.5*(self.bounds[:,1] - self.bounds[:,0]) ## distance between the shape grid for plotting

        if is_validation:
            n_rows, n_cols = self.val_plot_grid
            assert len(self.meshes) == n_rows * n_cols, f'Number of shapes {len(self.meshes)} must match the grid size {n_rows}x{n_cols} = {n_rows*n_cols}'
        else:
            n_rows, n_cols = n_rows_cols(len(self.meshes))

        fig = k3d.plot(height=800)#, grid_visible=False, camera_fov=1.0)
        for i_shape in range(len(self.meshes)):
            # fill it up horizontally first, then go to the next row
            i_col = (i_shape  % n_cols)
            i_row = (i_shape // n_cols)
            verts, faces = self.meshes[i_shape]
            group = f'Shape {i_shape}'
            fig += k3d.mesh(verts, faces, group=group, color=MESH_COLOR, side='double', opacity=MESH_OPACITY, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            
            if len(constraint_pts_dict) > 0:
                for i, constraint_name in enumerate(constraint_pts_dict.keys()):
                    constraint_pts = constraint_pts_dict[constraint_name]
                    color = generate_color(i)
                    # in the first epoch, only plot the interface
                    cur_plot= k3d.points(constraint_pts, group=group, point_size=DEFAULT_POINT_SIZE, name=constraint_name, color=color, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, 
                                      translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
                    # cur_plot.visible = (self.epoch > 0) or (constraint_name == 'interface')
                    fig += cur_plot
        
        return self._handle_plot_show(fig, fig_label=fig_label)
        
    def plot_shape_and_points(self, pc: PointWrapper, fig_label, point_attribute=None):
        """TODO: integrate this into plot_shape"""
        shape_grid = 1.5*(self.bounds[:,1] - self.bounds[:,0]) ## distance between the shape grid for plotting

        n_rows, n_cols = n_rows_cols(len(self.meshes))

        fig = k3d.plot(height=800)#, grid_visible=False, camera_fov=1.0)
        for i_shape in range(len(self.meshes)):
            i_col = (i_shape  % n_cols)
            i_row = (i_shape // n_cols)
            verts, faces = self.meshes[i_shape]
            group = f'Shape {i_shape}'
            x = pc.pts_of_shape(i_shape)

            fig += k3d.mesh(verts, faces, group=group, color=MESH_COLOR, side='double', opacity=MESH_OPACITY, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            if point_attribute is not None:
                plot_attributes = point_attribute[pc.get_idcs(i_shape)]
                # color_range=[0.,5.],
                fig += k3d.points(x, group=group, point_size=DEFAULT_POINT_SIZE, attribute=plot_attributes, color_map=matplotlib_color_maps.winter, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            else:
                fig += k3d.points(x, group=group, point_size=DEFAULT_POINT_SIZE, color=0x000000, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
        return self._handle_plot_show(fig, fig_label=fig_label)
        
        
    def _handle_plot_show(self, fig, fig_label, force_save=False, dim=3):
        """ Handle piping the plot to different outputs. """
        assert fig_label is not None, f"fig_label unspecified for wandb"
        # set_cam_for_wandb(fig, default_cameras[self.problem_str])
        
        if dim==3:
            formatted_fig = wandb.Html(fig.get_snapshot(), inject=False)
            return fig_label, formatted_fig
        
        elif dim==2:
            kwargs = {'transparent':False, 'facecolor':'w', 'dpi':100} 
            if not self.config.get('show_colorbar', False): kwargs |= {'bbox_inches':'tight', 'pad_inches':0}

            ## Let wandb handle the saving (this has a lot of whitespace)
            ## Remove the whitespace
            with BytesIO() as buf:
                fig.savefig(buf, **kwargs)
                buf.seek(0)
                wandb_img = wandb.Image(Image.open(buf))
            
            plt.close(fig) ## close figures, otherwise they just stack up
            
            return fig_label, wandb_img