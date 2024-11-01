import logging
import math
import datetime
import os.path
from pathlib import Path

import einops
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from k3d.colormaps import matplotlib_color_maps
import wandb
from matplotlib.collections import LineCollection
from PIL import Image
from io import BytesIO
from matplotlib.colors import LogNorm
import warnings

from models.point_wrapper import PointWrapper
from util.vis_utils import flip_arrow, generate_color, n_rows_cols
from util.misc import do_plot, set_and_true
import k3d
# Filter Warning - TODO: maybe fix this and other lines more properly later
# fig += k3d.mesh(verts, faces, group=group, color=MESH_COLOR, side='double', opacity=MESH_OPACITY, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row]) 
warnings.filterwarnings("ignore", message="Given trait value dtype.*")

logging.getLogger('matplotlib').setLevel(logging.WARNING)

TRAJ_WIDTH = 0.005
## COLORS
MESH_COLOR = 0xe31b23
MESH_OPACITY = 1.0

# Points
DEFAULT_POINT_SIZE = 0.01
DEFAULT_POINT_OPACITY = 1.0
DEFAULT_POINT_SHADER = '3dSpecular'
DEFAULT_PTS_COLOR = 0x808080
MINIMA_COLOR = 0x0000ff
MAXIMA_COLOR = 0xff0000
SADDLE_COLOR = 0x00ff00
# purple
ENDPOINT_COLOR = 0x800080

# Paths
DEFAULT_PATH_WIDTH = 0.005
DEFAULT_PATH_OPACITY = 1.0
PATH_SHADER = 'simple'
DEFAULT_PATH_COLOR = 0x606060
ASC_PATH_COLOR = 0xFFB3B3
DESC_PATH_COLOR = 0xB3B3FF


# STANDARD_COLORS
BLACK = 0x000000
GRAY = 0x808080
PASTEL_RED = 0xFFB3B3
PASTEL_GREEN = 0xB3FFB3
PASTEL_BLUE = 0xB3B3FF
PASTEL_YELLOW = 0xFFFFB3
PASTEL_PURPLE = 0xFFB3FF
PASTEL_CYAN = 0xB3FFFF
PASTEL_PEACH = 0xFFE5B4
PASTEL_LAVENDER = 0xE6B3FF
PASTEL_MINT = 0xB3FFC9
PASTEL_ROSE = 0xFFB3DF
PASTEL_SKY_BLUE = 0xB3E0FF
PASTEL_LEMON = 0xFFFFB3
PASTEL_CORAL = 0xFFC1B3
PASTEL_TEAL = 0xB3FFF2
PASTEL_VIOLET = 0xCAB3FF


class Plotter3d():

    def __init__(self, config, fig_size="?"):
        self.config = config
        self.logger = logging.getLogger('plot_helper')
        
        self.bounds = config['bounds'].cpu().numpy()
        # self.plot_scale = np.sqrt(np.prod(self.bounds[:, 1] - self.bounds[:, 0]))  # scale for plotting

        # assert config['train_plot_n_shapes'] <= config['ginn_bsize'], 'Can only plot up to ginn_bsize shapes'
        # assert config['plot_secondary_plots_every_n_epochs'] % config['plot_every_n_epochs'] == 0, 'plot_secondary_every_n_epochs must be a multiple of plot_every_n_epochs'

        # self.X0, self.X1, _ = get_meshgrid_in_domain(self.bounds)
        # self.y = None
        # self.Y = None
        self.epoch = 0

        self.fig_size = self.config['fig_size'] if 'fig_size' in config else fig_size

        if self.config['fig_save']:
            assert self.config["fig_path"] is not None, f"file_path must not be None if config['fig_save'] = True"
            cur_time_stamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            self.fig_parent_path = os.path.join(self.config['fig_path'], cur_time_stamp)
            Path(self.fig_parent_path).mkdir(parents=True, exist_ok=True)

    def do_plot(self, key=None):
        return do_plot(self.config, self.epoch, key=key)
    

    def do_plot_only_every_n_epochs(self):
        if 'plot_every_n_epochs' not in self.config or self.config['plot_every_n_epochs'] is None:
            return True
        return (self.epoch % self.config['plot_every_n_epochs'] == 0) or (self.epoch == self.config['max_epochs'])

    def reset_output(self, output_container, epoch=None):
        """Compute the function on the grid.
        output_container: a list of meshes.
        epoch: will be used to identify figures for wandb or saving
        """
        self.epoch = epoch
        if not self.do_plot_only_every_n_epochs():
            return
        self.meshes = output_container
        

    def plot_function_and_cps(self, pc: PointWrapper, fig_label, xc_attributes=None, p_backdrop:PointWrapper=None):
        """
        Plot the function and its critical points.
        xc: [N, 2], critical points
        fig_label: string labeling the figure for the output
        xc_attribute: [N], values to use for color-coding the points, e.g. gradient magnitudes
        x_backdrop: [N, 2], points. These are plotted lightly, e.g. to see the points before the operation
        """
        if not self.do_plot_only_every_n_epochs():
            return

        shape_grid = 1.5*(self.bounds[:,1] - self.bounds[:,0]) ## distance between the shape grid for plotting

        if xc_attributes is not None:
            xc_attributes = np.clip(xc_attributes, a_min=1.0e-15, a_max=None, out=xc_attributes)
            cmap = cm.get_cmap('inferno')
            normalizer = LogNorm(vmin=xc_attributes.min(), vmax=xc_attributes.max())
            colors = cmap(normalizer(xc_attributes))
            colors_rgb = (colors[:, :3] * 255).astype(np.uint32)
            colors_rgb_int = colors_rgb[:, 0] + (colors_rgb[:, 1] << 8) + (colors_rgb[:, 2] << 16)
        
        n_rows, n_cols = n_rows_cols(len(self.meshes), self.config.get('flatten_plots', False))
        
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


    def plot_shape(self, constraint_pts_dict, fig_label='Boundary'):
        shape_grid = 1.5*(self.bounds[:,1] - self.bounds[:,0]) ## distance between the shape grid for plotting

        n_rows, n_cols = n_rows_cols(len(self.meshes), self.config.get('flatten_plots', False))

        fig = k3d.plot(height=800)#, grid_visible=False, camera_fov=1.0)
        for i_shape in range(len(self.meshes)):
            i_col = (i_shape  % n_cols)
            i_row = (i_shape // n_cols)
            verts, faces = self.meshes[i_shape]
            group = f'Shape {i_shape}'
            fig += k3d.mesh(verts, faces, group=group, color=MESH_COLOR, side='double', opacity=MESH_OPACITY, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            
            if self.epoch == 0 and len(constraint_pts_dict) > 0:
                for i, constraint_name in enumerate(constraint_pts_dict.keys()):
                    constraint_pts = constraint_pts_dict[constraint_name]
                    color = generate_color(i)
                    visibility = constraint_name == 'interface'
                    cur_plot= k3d.points(constraint_pts, group=group, point_size=DEFAULT_POINT_SIZE, name=constraint_name, color=color, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, 
                                      translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
                    cur_plot.visible = visibility
                    fig += cur_plot
            
        return self._handle_plot_show(fig, fig_label=fig_label)
        
    def plot_shape_and_points(self, pc: PointWrapper, fig_label, point_attribute=None):
        """TODO: integrate this into plot_shape"""
        if not self.do_plot_only_every_n_epochs():
            return
    
        shape_grid = 1.5*(self.bounds[:,1] - self.bounds[:,0]) ## distance between the shape grid for plotting

        n_rows, n_cols = n_rows_cols(len(self.meshes), self.config.get('flatten_plots', False))

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
                fig += k3d.points(x, group=group, point_size=DEFAULT_POINT_SIZE, attribute=plot_attributes, color_map=matplotlib_color_maps.winter, color_range=[0.,5.], opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            else:
                fig += k3d.points(x, group=group, point_size=DEFAULT_POINT_SIZE, color=0x000000, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
        return self._handle_plot_show(fig, fig_label=fig_label)
        
        
    def _handle_plot_show(self, fig, fig_label=None, force_save=False, dim=3):
        """ Handle piping the plot to different outputs. """
        if dim==3:
            formatted_fig = None
            if self.config['fig_wandb']:
                assert fig_label is not None, f"fig_label unspecified for wandb"
                formatted_fig = wandb.Html(fig.get_snapshot(), inject=False)
            elif self.config['fig_save']:
                assert fig_label is not None, f"fig_label unspecified for saving the figure"
                fig_path = f'{self.fig_parent_path}/{fig_label}'
                Path(fig_path).mkdir(parents=True, exist_ok=True)
                file_path = f'{fig_path}/{self.epoch}.html'
                with open(file_path, 'w') as f:
                    f.write(fig.get_snapshot())
                raise NotImplementedError
            elif self.config['fig_pynb']:
                formatted_fig = fig
            else:
                assert False, "unknown plot output"
            
            return fig_label, formatted_fig
        
        elif dim==2:
            kwargs = {'transparent':False, 'facecolor':'w', 'dpi':100} 
            if not set_and_true('show_colorbar', self.config): kwargs |= {'bbox_inches':'tight', 'pad_inches':0}

            wandb_img = None
            if self.config['fig_show']:
                plt.tight_layout()
                plt.show()
            elif self.config['fig_save']:
                assert fig_label is not None, f"fig_label unspecified for saving the figure"
                fig_path = f'{self.fig_parent_path}/{fig_label}'
                Path(fig_path).mkdir(parents=True, exist_ok=True)
                file_path = f'{fig_path}/{self.epoch}.png'
                fig.savefig(file_path, **kwargs)
            elif self.config['fig_wandb']:
                assert fig_label is not None, f"fig_label unspecified for wandb"
                ## Let wandb handle the saving (this has a lot of whitespace)
                ## Remove the whitespace
                with BytesIO() as buf:
                    fig.savefig(buf, **kwargs)
                    buf.seek(0)
                    wandb_img = wandb.Image(Image.open(buf))
            
            if force_save: ## TODO: quickfix for animations
                file_path = f'{self.config["fig_path"]}/{fig_label}/{self.epoch}.png'
                fig.savefig(file_path, **kwargs)
            
            plt.close(fig) ## close figures, otherwise they just stack up
            
            return fig_label, wandb_img

    def _handle_plot_show_matplotlib(self, fig, fig_label=None, force_save=False):
        """ Handle piping the plot to different outputs. """

        kwargs = {'transparent':False, 'facecolor':'w', 'dpi':100} 
        if not set_and_true('show_colorbar', self.config): kwargs |= {'bbox_inches':'tight', 'pad_inches':0}

        wandb_img = None
        if self.config['fig_show']:
            # plt.suptitle(fig_label)   
            plt.tight_layout()
            plt.show()
        elif self.config['fig_save']:
            assert fig_label is not None, f"fig_label unspecified for saving the figure"
            fig_path = f'{self.fig_parent_path}/{fig_label}'
            Path(fig_path).mkdir(parents=True, exist_ok=True)
            file_path = f'{fig_path}/{self.epoch}.png'
            fig.savefig(file_path, **kwargs)
        elif self.config['fig_wandb']:
            assert fig_label is not None, f"fig_label unspecified for wandb"
            ## Let wandb handle the saving (this has a lot of whitespace)
            ## Remove the whitespace
            with BytesIO() as buf:
                fig.savefig(buf, **kwargs)
                buf.seek(0)
                wandb_img = wandb.Image(Image.open(buf))
            # self.log[fig_label] = wandb_img
            # self.log[fig_label] = wandb.Image(fig)
            # self.log[fig_label] = fig
        
        if force_save: ## TODO: quickfix for animations
            file_path = f'{self.config["fig_path"]}/{fig_label}/{self.epoch}.png'
            fig.savefig(file_path, **kwargs)
        
        plt.close(fig) ## close figures, otherwise they just stack up
        
        return fig_label, wandb_img
    
    