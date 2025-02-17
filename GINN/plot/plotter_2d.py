import math
import datetime
import os.path
from pathlib import Path
import time

import einops
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

from models.net_w_partials import NetWithPartials
from models.point_wrapper import PointWrapper
from util.model_utils import tensor_product_xz
from util.vis_utils import flip_arrow, n_rows_cols
from util.visualization.utils_mesh import get_meshgrid_in_domain
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class Plotter2d():

    def __init__(self, envelope, bounds, level_set, val_plot_grid, plot_2d_resolution, show_colorbar=True, fig_size=(12, 12), **kwargs):
        
        self.envelope = envelope
        self.bounds = bounds.cpu().numpy()
        self.level_set = level_set
        self.val_plot_grid = val_plot_grid
        self.show_colorbar = show_colorbar
        self.fig_size = fig_size
        self.logger = logging.getLogger('plot_helper')
        self.plot_scale = np.sqrt(np.prod(self.bounds[:, 1] - self.bounds[:, 0]))  # scale for plotting
        self.X0, self.X1, self.xs = get_meshgrid_in_domain(self.bounds, plot_2d_resolution, plot_2d_resolution)
        # self.xs = torch.tensor(self.xs, dtype=torch.float32)
        self.y = None
        self.Y = None
        self.epoch = 0
    
    def reset_output(self, Y, epoch=None, X0=None, X1=None):
        '''
        Set the field to plot.
        INFO: The output is not recomputed here, as this plotter must be kept torch-free. 
              Otherwise multiprocessing needs torch context and occupies GPU memory.
        '''
        assert isinstance(Y, np.ndarray), 'Y must be a numpy array'
        self.Y = Y
        self.epoch = epoch
        if X0 is not None:
            self.X0 = X0
        if X1 is not None:
            self.X1 = X1
    
    def _draw_base(self, ax, i_shape, constraint_pts_dict=[], show_colorbar=None, levels=20):
        Y_i = self.Y[i_shape]
        im = ax.imshow(Y_i, origin='lower', extent=self.bounds.flatten())
        ax.contour(self.X0, self.X1, Y_i)
        # ax.contour(self.X0, self.X1, Y_i, levels=levels, colors='gray', linewidths=0.5)
        ax.contour(self.X0, self.X1, Y_i, levels=[self.level_set], colors='r')
        
        if len(constraint_pts_dict) > 0 and self.epoch == 0:
            for i, constraint_name in enumerate(constraint_pts_dict.keys()):
                if constraint_name == 'inside_envelope':
                    continue
                color = 'k' if i==0 else plt.cm.tab10(i)
                ax.scatter(*constraint_pts_dict[constraint_name], color=color, marker='.', zorder=10, alpha=0.2)
                
        if show_colorbar is None:
            show_colorbar = self.show_colorbar
        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust the size and pad as needed
            plt.colorbar(im, cax=cax)

    def plot_grad_field(self, grad_fields, constraint_pts_dict=None, fig_label="Grad field", attributes=None):
        '''Plot the gradient field(s)'''
        n_rows, n_cols = n_rows_cols(len(grad_fields))
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=self.fig_size)
        axs = axs.ravel() if len(grad_fields) > 1 else [axs]
        for i_shape in range(len(grad_fields)):
            ax = axs[i_shape]
            gf_i = grad_fields[i_shape]
            im = ax.imshow(gf_i, origin='lower', extent=self.envelope.flatten())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust the size and pad as needed
            plt.colorbar(im, cax=cax)
            ax.axis('scaled')
            if attributes is not None:
                ax.set_title(f'{attributes[i_shape].item():0.3f}')
            
        return self._handle_plot_show(fig, fig_label=fig_label)
  
    def plot_shape(self, fig_label="Boundary", constraint_pts_dict={}, is_validation=False):
        
        if is_validation:
            n_rows, n_cols = self.val_plot_grid
            assert len(self.Y) == n_rows * n_cols, 'Number of shapes must match the grid size'
        else:    
            n_rows, n_cols = n_rows_cols(len(self.Y))
            
        fig, axs = plt.subplots(n_rows, n_cols, figsize=self.fig_size)
        axs = axs.ravel() if len(self.Y) > 1 else [axs]
        for i_shape in range(len(self.Y)):
            ax = axs[i_shape]
            self._draw_base(ax, i_shape, constraint_pts_dict=constraint_pts_dict)
            ax.axis('scaled')
            # ax.axis('off')
        return self._handle_plot_show(fig, fig_label=fig_label)
        
    def plot_shape_and_points(self, pc: PointWrapper, fig_label, point_attribute, is_validation=False, **kwargs):
    
        if is_validation:
            n_rows, n_cols = self.val_plot_grid
            assert len(self.Y) == n_rows * n_cols, 'Number of shapes must match the grid size'
        else:
            n_rows, n_cols = n_rows_cols(len(self.Y))
            
        fig, axs = plt.subplots(n_rows, n_cols, figsize=self.fig_size)
        axs = axs.ravel() if len(pc.get_map()) > 1 else [axs]
        for i_shape in range(len(pc.get_map())):
            ax = axs[i_shape]
            x = pc.pts_of_shape(i_shape)
            self._draw_base(ax, i_shape)
            ax.scatter(*x.T, marker='.', zorder=10, c=point_attribute[pc.get_idcs(i_shape)])
            ax.axis('scaled')
            # ax.axis('off')
        return self._handle_plot_show(fig, fig_label=fig_label)

    def _handle_plot_show(self, fig, fig_label=None):
        """ Handle piping the plot to different outputs. """

        kwargs = {'transparent':False, 'facecolor':'w', 'dpi':100} 
        if not self.show_colorbar: 
            kwargs |= {'bbox_inches':'tight', 'pad_inches':0}

        wandb_img = None
        assert fig_label is not None, f"fig_label unspecified for wandb"
        ## Let wandb handle the saving (this has a lot of whitespace)
        ## Remove the whitespace
        with BytesIO() as buf:
            fig.savefig(buf, **kwargs)
            buf.seek(0)
            wandb_img = wandb.Image(Image.open(buf))
                
        plt.close(fig) ## close figures, otherwise they just stack up
        
        return fig_label, wandb_img

    