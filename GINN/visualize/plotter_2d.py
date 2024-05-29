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
from utils import do_plot, set_and_true, flip_arrow
from visualization.utils_mesh import get_meshgrid_in_domain
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class Plotter2d():

    def __init__(self, config,  fig_size=(12, 12)):
        self.config = config
        self.logger = logging.getLogger('plot_helper')
        
        self.bounds = config['bounds'].cpu().numpy()
        self.plot_scale = np.sqrt(np.prod(self.bounds[:, 1] - self.bounds[:, 0]))  # scale for plotting
        self.plot_n_shapes = self.config['plot_n_shapes']
        self.n_rows, self.n_cols = self.n_rows_cols()

        assert config['plot_n_shapes'] <= config['batch_size'], 'Can only plot up to batch_size shapes'

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
        
        if hasattr(self, 'cur_fig'):
            plt.close(self.cur_fig)
        fig, axs = plt.subplots(self.n_rows, self.n_cols, figsize=self.fig_size)
        if self.plot_n_shapes == 1:
            axs = [axs]
        else:
            axs = axs.ravel() if self.plot_n_shapes > 1 else [axs]
        for i_shape in range(self.plot_n_shapes):
            ax = axs[i_shape]
            self._draw_base(ax, i_shape, levels=20)
        self.cur_fig, self.cur_axs = fig, axs

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

        fig, axs = plt.subplots(self.n_rows, self.n_cols, figsize=self.fig_size)
        axs = axs.ravel() if self.plot_n_shapes > 1 else [axs]
        
        if xc_attributes is not None:
            # clip to 1.0e-15 to avoid 0-error when using LogNorm
            np.clip(xc_attributes, a_min=1.0e-15, a_max=None, out=xc_attributes)
            # set up shared colormap
            cmap=cm.get_cmap('inferno')
            normalizer=LogNorm()
            im=cm.ScalarMappable(norm=normalizer, cmap=cmap)
            im.set_array(xc_attributes)
            im.set_clim(xc_attributes.min(), xc_attributes.max())
            
        for i_shape in range(self.plot_n_shapes):
            ax = axs[i_shape]
            shape_idcs = pc.get_idcs(i_shape)
            xc = pc.pts_of_shape(i_shape)

            self._draw_base(ax, i_shape, show_colorbar=False)
            if p_backdrop is not None:
                ax.scatter(*p_backdrop.data[shape_idcs].T, c='gray', marker='o', alpha=.5)
            if xc_attributes is not None:
                sc = ax.scatter(*xc.T, c=xc_attributes[shape_idcs], marker='o', cmap=cmap, edgecolors='k', norm=normalizer)
                # TODO: fix for multiple axes
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust the size and pad as needed
                # plt.colorbar(sc, cax=cax)
            else:
                ax.scatter(*xc.T, c='k', marker='o')
            ax.axis('scaled')
            ax.axis('off')
        if xc_attributes is not None:
            pass
            # cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.9)
            # cbar.set_ticks(cbar.get_ticks().tolist()+[vmin, vmax])
        return self.handle_plot_show(fig, fig_label=fig_label)

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

    def plot_characterized_cps(self, pc, eigvecs_all, eigvals_all):
        """
        Plot the principal curvatures (eigenvectors of the Hessian) at the critical points.
        xc: [N, 2], critical points
        eigvecs: [N, 2, 2], eigenvectors. eigvecs[i, j, k] gives the j-th dimension of the k-th vector at the i-th critical point
        eigvals: [N, 1], eigenvalues
        """
        if not self.do_plot_only_every_n_epochs():
            return
        
        fig_label = "Principal curvatures"
        fig, axs = plt.subplots(self.n_rows, self.n_cols, figsize=self.fig_size)
        axs = axs.ravel() if self.plot_n_shapes > 1 else [axs]
        for i_shape in range(self.plot_n_shapes):
            ax = axs[i_shape]
            shape_idcs = pc.get_idcs(i_shape)
            xc = pc.pts_of_shape(i_shape)
            eigvecs = eigvecs_all[shape_idcs]
            eigvals = eigvals_all[shape_idcs]
            self._draw_base(ax, i_shape)

            for i in range(len(xc)):
                for j in [0,1]:
                    xy_pos = xc[i]
                    xy_neg = xc[i]
                    dxy_pos = eigvecs[i, :, j] * .03 * self.plot_scale
                    dxy_neg = eigvecs[i, :, j] * -.03 * self.plot_scale
                    if eigvals[i, j]<0:
                        xy_pos, dxy_pos = flip_arrow(xy_pos, dxy_pos)
                        xy_neg, dxy_neg = flip_arrow(xy_neg, dxy_neg)
                    ax.arrow(*xy_pos.T, *dxy_pos.T, length_includes_head=True, head_width=.007*self.plot_scale, overhang=.5, color='k')
                    ax.arrow(*xy_neg.T, *dxy_neg.T, length_includes_head=True, head_width=.007*self.plot_scale, overhang=.5, color='k')
            # ax.scatter(*xc_u.T, c='k', marker='.')
            ax.axis('scaled')
            ax.axis('off')

        return self.handle_plot_show(fig, fig_label=fig_label)

    def plot_perturbed_points_from_saddle_points(self, pc, indexes_all, p_path0_all):
        """NOTE: remove?"""
        if not self.do_plot_only_every_n_epochs():
            return
        
        fig_label = "Perturbations of CPs"
        fig, axs = plt.subplots(self.n_rows, self.n_cols, figsize=self.fig_size)
        axs = axs.ravel() if self.plot_n_shapes > 1 else [axs]
        for i_shape in range(self.plot_n_shapes):
            ax = axs[i_shape]
            shape_idcs = pc.get_idcs(i_shape)
            xc = pc.pts_of_shape(i_shape)
            x_path0 = p_path0_all.pts_of_shape(i_shape)
            indexes = indexes_all[shape_idcs]

            nof_half_paths = len(x_path0)//2 # index for selecting min/max paths; number of paths to min/max if half of all paths

            ## Plot the perturbations
            self._draw_base(ax, i_shape)
            ax.scatter(*xc[indexes == 1].T, c='k', marker='.')
            ax.scatter(*x_path0[:nof_half_paths].T, c='orange', marker='.')
            ax.scatter(*x_path0[nof_half_paths:].T, c='blue', marker='.')
            ax.axis('scaled')
            # ax.axis('off')
        return self.handle_plot_show(fig, fig_label=fig_label)

    def plot_saddle_trajectories(self, pc, p_path_list, x_path_over_iters_all, indexes_all, multipliers_to_descend_all):
        """
        xc: [N, 2] critical points
        x_path_over_iters: [N_iter, N_paths, 2] trajectories of 4*N_saddles = N_paths. First half are to minima, second half are to maxima.
        """
        if not self.do_plot_only_every_n_epochs():
            return
        
        fig_label = "Surfnet graph edges"
        x_path_over_iters_all = np.copy(x_path_over_iters_all)  ## make a copy to avoid messing with the original data

        offset_over_4_directions = np.arange(2 * self.config['nx']) * x_path_over_iters_all.shape[1]//4

        fig, axs = plt.subplots(self.n_rows, self.n_cols, figsize=self.fig_size)
        axs = axs.ravel() if self.plot_n_shapes > 1 else [axs]
        for i_shape in range(self.plot_n_shapes):
            ax = axs[i_shape]
            xc = pc.pts_of_shape(i_shape)
            indexes = indexes_all[pc.get_idcs(i_shape)]
            path_idcs_single = p_path_list[0].get_idcs(i_shape)
            path_idcs = np.concatenate([path_idcs_single+off for off in offset_over_4_directions], axis=0)
            x_path_over_iters = x_path_over_iters_all[:, path_idcs]
            multipliers_to_descend = multipliers_to_descend_all[path_idcs]
            is_desc = multipliers_to_descend > 0
            
            # nof_half_paths = x_path_over_iters.shape[1]//2 # index for selecting min/max paths; number of paths to min/max if half of all paths

            ## Mask the points outside the domain, because this messes with plotting
            out_mask = (x_path_over_iters < self.bounds[:,0]).any(2) | (x_path_over_iters > self.bounds[:,1]).any(2)
            out_mask = out_mask
            x_path_over_iters[out_mask] = np.nan

            ## Plot the trajectories
            self._draw_base(ax, i_shape)
            ax.plot(x_path_over_iters[:,is_desc,0], x_path_over_iters[:,is_desc,1], color='blue', linewidth=1.0, alpha=0.95)
            ax.plot(x_path_over_iters[:,~is_desc,0], x_path_over_iters[:,~is_desc,1], color='orange', linewidth=1.0, alpha=0.95)
            # ax.plot(x_path_over_iters[:,:nof_half_paths,0], x_path_over_iters[:,:nof_half_paths,1] , c='orange')
            # ax.plot(x_path_over_iters[:,nof_half_paths:,0], x_path_over_iters[:,nof_half_paths:,1] , c='blue')
            if indexes is None:
                ax.scatter(*xc.T, c='k', marker='o')
            else:
                ax.scatter(*xc[indexes == 0].T, c='b', s=120, marker='o', edgecolors='k', linewidths=2)
                ax.scatter(*xc[indexes == 1].T, c='g', s=120, marker='o', edgecolors='k', linewidths=2)
                ax.scatter(*xc[indexes == 2].T, c='r', s=120, marker='o', edgecolors='k', linewidths=2)
            ax.axis('scaled')
            # ax.axis('off')
        return self.handle_plot_show(fig, fig_label=fig_label)
        
    def plot_descent_trajectories(self, pc: PointWrapper, x_path_over_iters_all, fig_label="Trajectories of finding CPs", plot_anyway=False):
        """
        xc: [N, 2] starting points
        x_path_over_iters: [N_iter, N_paths, 2] trajectories of 4*N_saddles = N_paths. First half are to minima, second half are to maxima.
        """
        if not plot_anyway and not self.do_plot_only_every_n_epochs():
            return

        # start_t1 = time.time()
        x_path_over_iters_all = np.copy(x_path_over_iters_all)  ## make a copy to avoid messing with the original data

        fig, axs = plt.subplots(self.n_rows, self.n_cols, figsize=self.fig_size)
        axs = axs.ravel() if self.plot_n_shapes > 1 else [axs]
        for i_shape in range(self.plot_n_shapes):
            ax = axs[i_shape]
            shape_idcs = pc.get_idcs(i_shape)
            x_path_over_iters = x_path_over_iters_all[:, shape_idcs]

            ## Mask the points outside the domain, because this messes with plotting
            out_mask = (x_path_over_iters < self.bounds[:,0]).any(2) | (x_path_over_iters > self.bounds[:,1]).any(2)
            out_mask = out_mask
            x_path_over_iters[out_mask] = np.nan

            ## Plot the trajectories
            # start_t = time.time()
            # self.draw_base(ax, i_shape, levels=50)
            # self.logger.debug(f'needed {time.time() - start_t:0.2f} secs for draw_base')
            
            k=20
            kth_idcs = np.arange(start=0, stop=len(x_path_over_iters), step=k)
            if set_and_true('trajectories_use_colors', self.config):
                start_t = time.time()
                cmap = mpl.colormaps.get_cmap('tab10').resampled(x_path_over_iters.shape[1])
                colors = cmap(np.arange(0, cmap.N))
                np.random.shuffle(colors)
                # self.logger.debug(f'needed {time.time() - start_t1:0.2f} secs for colors')

                for i, color in enumerate(colors):
                    ax.plot(x_path_over_iters[:,i,0], x_path_over_iters[:,i,1], color=color, linewidth=1.0, alpha=0.95)
                    ax.scatter(x_path_over_iters[kth_idcs, i, 0].flatten(), x_path_over_iters[kth_idcs, i, 1].flatten(), marker='x', color=color, s=50, linewidth=1.2, alpha=0.95)
            else:
                ax.plot(x_path_over_iters[:,:,0], x_path_over_iters[:,:,1], linewidth=1.0, alpha=0.8, linestyle='-.')
                ax.scatter(x_path_over_iters[kth_idcs, :, 0].flatten(), x_path_over_iters[kth_idcs, :, 1].flatten(), marker='x', color='k', s=50, linewidth=1.2, alpha=0.8)
            
            # self.logger.debug(f'needed {time.time() - start_t1:0.2f} until for plotting descent')                

            ax.scatter(x_path_over_iters[0,:,0], x_path_over_iters[0,:,1], marker='.', c='gray', zorder=10) # start points
            # ax.scatter(x_path_over_iters[-1,:,0], x_path_over_iters[-1,:,1], marker='.', c='k', zorder=10) # end points

            ax.axis('scaled')
            # ax.axis('off')
        start_t = time.time()
        res = self.handle_plot_show(fig, fig_label=fig_label)
        # self.logger.debug(f'needed {time.time() - start_t:0.2f} secs for handle_plot_show')
        # self.logger.debug(f'needed {time.time() - start_t1:0.2f} secs for all')

        return res

    # TODO
    # def plot_edges_from_augm_surfnet_graph(pc_cpu, indexes_cpu, w_graph_list, fig_label, show_v_id, edge_weights, pts_to_penalize_list, penalties_list):
        
    #     edge_weights = [data['weight'] for _, _, data in w_graph_list.edges_z_list.data()]
    #     torch.tensor([list(t) for t in w_graph_list.edges_z_list()], device='cpu')
    #     self.plot_edges(pc_cpu, indexes_cpu, torch.tensor([list(t) for t in w_graph_list.edges_z_list()], device='cpu'),
    #                            "Weighted graph edges", show_v_id=False, edge_weights=edge_weights,
    #                                pts_to_penalize=pts_to_penalize_list, penalties=penalties_list)

    def plot_edges(self, pc, indexes_all, edges_list, fig_label, show_v_id=True, **kwargs):
        if not self.do_plot_only_every_n_epochs():
            return

        fig, axs = plt.subplots(self.n_rows, self.n_cols, figsize=self.fig_size)
        axs = axs.ravel() if self.plot_n_shapes > 1 else [axs]
        for i_shape in range(self.plot_n_shapes):
            ax = axs[i_shape]
            xc = pc.pts_of_shape(i_shape)
            indexes = indexes_all[i_shape]
            edges = edges_list[i_shape]
            

            self._draw_base(ax, i_shape)
            ax.scatter(*xc[indexes==0].T, c='b', s=120, marker='o', edgecolors='k', linewidths=2)
            ax.scatter(*xc[indexes==1].T, c='g', s=120, marker='o', edgecolors='k', linewidths=2)
            ax.scatter(*xc[indexes==2].T, c='r', s=120, marker='o', edgecolors='k', linewidths=2)

            if show_v_id:
                for i, xc_ in enumerate(xc):
                    ax.annotate(str(i), xc_, fontsize=20, color='w', ha='center')
    
            if ('penalties' in kwargs) and (kwargs['penalties'][i_shape] is not None) and (len(kwargs['penalties'][i_shape]) > 0):
                assert 'pts_to_penalize' in kwargs, 'pts_to_penalize not set'
                for plty, pt in zip(kwargs['penalties'][i_shape], kwargs['pts_to_penalize'][i_shape]):
                    ax.annotate(f"{plty:0.2f}", pt, fontsize=20, color='w', ha='center')

            if edges is None or edges.shape[0] == 0:
                self.logger.debug(f'no edges to plot')
                continue

            lc = LineCollection(xc[edges], color='k', linewidths=2)
            ax.add_collection(lc)
            if 'edge_weights' in kwargs and len(kwargs['edge_weights']) > 0:
                edge_centers = xc[edges].mean(dim=1)
                for weight, center in zip(kwargs['edge_weights'][i_shape], edge_centers):
                    ax.annotate(f'{weight:0.1f}', center, fontsize=20, color='w', ha='center')

            ax.axis('scaled')
            # ax.axis('off')
        return self.handle_plot_show(fig, fig_label=fig_label)

    def plot_hist(self, data, fig_label, log=True):
        if not self.do_plot_only_every_n_epochs():
            return
        
        fig, ax = plt.subplots(figsize=self.fig_size)

        if log:
            data = np.clip(data, a_min=1.0e-15, a_max=None)
            data = np.log10(data)

        ax.hist(data, bins=20, edgecolor='black')
        # ax.set_xscale('log')
        return self.handle_plot_show(fig, fig_label=fig_label)

    def plot_CCs_of_surfnet_graph(self, surfnet_graph_sub_CCs_list, pc, pyc_u_all, edges_list):
        """surfnet graph"""
        if not self.do_plot_only_every_n_epochs():
            return
        
        fig_label = "surfnet graph"
        fig, axs = plt.subplots(self.n_rows, self.n_cols, figsize=self.fig_size)
        axs = axs.ravel() if self.plot_n_shapes > 1 else [axs]
        for i_shape in range(self.plot_n_shapes):
            ax = axs[i_shape]
            xc = pc.pts_of_shape(i_shape)
            if len(xc) == 0:
                continue
            surfnet_graph_sub_CCs = surfnet_graph_sub_CCs_list[i_shape]
            yc_u = pyc_u_all.pts_of_shape(i_shape)
            edges = edges_list[i_shape]
            
            
            ## First coordinate could be some projection of 2D x space onto 1D space
            ## Second coordinate is the level value
            pts = np.vstack([xc[:, 0], yc_u]).T

            cmap = plt.get_cmap("tab10")

            ax.scatter(*pts.T, c='k', marker='o')
            for i, pt in enumerate(pts):
                ax.annotate(str(i), pt, fontsize=20, color='k')
            lc = LineCollection(pts[edges], color='k', linewidths=1, linestyle=':')
            ax.add_collection(lc)

            if surfnet_graph_sub_CCs is None:
                continue

            for i, CC in enumerate(surfnet_graph_sub_CCs):
                nodes_CC = np.array(CC.nodes(), dtype=int)
                edges_CC = np.array(CC.edges(), dtype=int)
                lc = LineCollection(pts[edges_CC], linewidths=2, color=cmap(i))
                ax.add_collection(lc)
                ax.scatter(*pts[nodes_CC].T, color=cmap(i))

            ax.axhline(self.config['level_set'], color='r')
        return self.handle_plot_show(fig, fig_label=fig_label)

    def plot_shape(self, constraint_pts_list):
        fig_label = "Boundary"
        fig, axs = plt.subplots(self.n_rows, self.n_cols, figsize=self.fig_size)
        axs = axs.ravel() if self.plot_n_shapes > 1 else [axs]
        for i_shape in range(self.plot_n_shapes):
            ax = axs[i_shape]
            self._draw_base(ax, i_shape, constraint_pts_dict=constraint_pts_list)
            ax.axis('scaled')
            # ax.axis('off')
        return self.handle_plot_show(fig, fig_label=fig_label, force_save=self.config['force_saving_shape'])
        
    def plot_shape_and_points(self, pc: PointWrapper, fig_label):
        if not self.do_plot_only_every_n_epochs():
            return
    
        fig, axs = plt.subplots(self.n_rows, self.n_cols, figsize=self.fig_size)
        axs = axs.ravel() if self.plot_n_shapes > 1 else [axs]
        for i_shape in range(self.plot_n_shapes):
            ax = axs[i_shape]
            x = pc.pts_of_shape(i_shape)
            self._draw_base(ax, i_shape)
            ax.scatter(*x.T, color='k', marker='.', zorder=10)
            ax.axis('scaled')
            # ax.axis('off')
        return self.handle_plot_show(fig, fig_label=fig_label, force_save=self.config['force_saving_shape'])

    def handle_plot_show(self, fig, fig_label=None, force_save=False):
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
        
    def n_rows_cols(self):
        ''' Returns the number of n_rows and columns. Favor more n_rows over n_cols to better show in wandb side-by-side '''
        n_cols = int(math.sqrt(self.plot_n_shapes))
        n_rows = int(math.ceil(self.plot_n_shapes / n_cols))
        return n_rows, n_cols
    
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
    