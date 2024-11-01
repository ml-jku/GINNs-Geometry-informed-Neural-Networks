import math
import datetime
import os.path
from pathlib import Path
import time

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import wandb
from matplotlib.collections import LineCollection
from PIL import Image
from io import BytesIO
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

from models.point_wrapper import PointWrapper
from util.vis_utils import flip_arrow, n_rows_cols
from util.misc import do_plot, set_and_true
from util.visualization.utils_mesh import get_meshgrid_in_domain
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class Plotter2d():

    def __init__(self, config,  fig_size=(12, 12)):
        self.config = config
        self.logger = logging.getLogger('plot_helper')
        
        self.bounds = config['bounds'].cpu().numpy()
        self.plot_scale = np.sqrt(np.prod(self.bounds[:, 1] - self.bounds[:, 0]))  # scale for plotting
        
        self.X0, self.X1, _ = get_meshgrid_in_domain(self.bounds)
        self.y = None
        self.Y = None
        self.epoch = 0

        self.fig_size = self.config['fig_size'] if 'fig_size' in config else fig_size

        if self.config['fig_save']:
            assert self.config["fig_path"] is not None, f"file_path must not be None if config['fig_save'] = True"
            cur_time_stamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            self.fig_parent_path = os.path.join(self.config['fig_path'], cur_time_stamp)
            Path(self.fig_parent_path).mkdir(parents=True, exist_ok=True)

    def do_plot_only_every_n_epochs(self):
        if 'plot_every_n_epochs' not in self.config or self.config['plot_every_n_epochs'] is None:
            return True
        return (self.epoch % self.config['plot_every_n_epochs'] == 0) or (self.epoch == self.config['max_epochs'])

    def reset_output(self, output_container, epoch=None):
        """Compute the function on the grid.
        output_container: a tuple of outputs as batched vector (y) or reshaped in a (batch of) 2D grids (Y).
        epoch: will be used to identify figures for wandb or saving
        """
        self.epoch = epoch
        if not self.do_plot_only_every_n_epochs():
            return
        self.y, self.Y = output_container

    def _draw_base(self, ax, i_shape, constraint_pts_dict=[], show_colorbar=None, levels=50):
        Y_i = self.Y[i_shape]
        im = ax.imshow(Y_i, origin='lower', extent=self.bounds.flatten())
        ax.contour(self.X0, self.X1, Y_i, levels=levels, colors='gray', linewidths=1.0)
        ax.contour(self.X0, self.X1, Y_i, levels=[self.config['level_set']], colors='r')
        
        if len(constraint_pts_dict) > 0:
            for i, constraint_name in enumerate(constraint_pts_dict.keys()):
                if constraint_name == 'domain':
                    continue
                color = 'k' if i==0 else plt.cm.tab10(i)
                ax.scatter(*constraint_pts_dict[constraint_name], color=color, marker='.', zorder=10)

        if show_colorbar is None:
            show_colorbar = set_and_true('show_colorbar', self.config)
        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust the size and pad as needed
            plt.colorbar(im, cax=cax)

  
    def plot_shape(self, constraint_pts_list, fig_label="Boundary"):
        
        n_rows, n_cols = n_rows_cols(len(self.Y), self.config.get('flatten_plots', False))
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=self.fig_size)
        axs = axs.ravel() if len(self.Y) > 1 else [axs]
        for i_shape in range(len(self.Y)):
            ax = axs[i_shape]
            self._draw_base(ax, i_shape, constraint_pts_dict=constraint_pts_list)
            ax.axis('scaled')
            # ax.axis('off')
        return self._handle_plot_show(fig, fig_label=fig_label, force_save=self.config['force_saving_shape'])
        
    def plot_shape_and_points(self, pc: PointWrapper, fig_label, **kwargs):
        if not self.do_plot_only_every_n_epochs():
            return
    
        n_rows, n_cols = n_rows_cols(len(pc.get_map()), self.config.get('flatten_plots', False))
    
        fig, axs = plt.subplots(n_rows, n_cols, figsize=self.fig_size)
        axs = axs.ravel() if len(pc.get_map()) > 1 else [axs]
        for i_shape in range(len(pc.get_map())):
            ax = axs[i_shape]
            x = pc.pts_of_shape(i_shape)
            self._draw_base(ax, i_shape)
            ax.scatter(*x.T, color='k', marker='.', zorder=10)
            ax.axis('scaled')
            # ax.axis('off')
        return self._handle_plot_show(fig, fig_label=fig_label, force_save=self.config['force_saving_shape'])

    def _handle_plot_show(self, fig, fig_label=None, force_save=False):
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

    def do_plot(self, key=None):
        return do_plot(self.config, self.epoch, key=key)
        # """
        # Checks if the plot specified by the key should be produced.
        # First checks if the output is set: if not, no need to produce the plot.
        # Then, checks if the key is set and true.
        # If the key is not specified, only checks the global plotting behaviour.
        # """
        # is_output_set = self.config['fig_show'] or self.config['fig_save'] or self.config['fig_wandb']

        # # if no output is set, no need to produce the plot
        # if is_output_set == False:
        #     return False
        
        # # if no key is specified, only check the global plotting behaviour
        # if key is None:
        #     return is_output_set
        
        # # otherwise, check if the key is set
        # if key not in self.config:
        #     return False
        
        # val = self.config[key]
        # if isinstance(val, bool):
        #     return val
        
        # if isinstance(val, dict):
        #     # if the val is a tuple, it is (do_plot, plot_interval)
        #     if val['on'] == False:
        #         return False
        #     else:
        #         assert val['interval'] % self.config['plot_every_n_epochs'] == 0, f'plot_interval must be a multiple of plot_every_n_epochs'
        #         return (self.epoch % val['interval'] == 0) or (self.epoch == self.config['max_epochs'])
        
        # raise ValueError(f'Unknown value for key {key}: {val}')
    