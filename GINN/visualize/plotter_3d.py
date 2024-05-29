import logging
import math
import datetime
import os.path
from pathlib import Path

import einops
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import wandb
from matplotlib.collections import LineCollection
from PIL import Image
from io import BytesIO
from matplotlib.colors import LogNorm
import warnings

from models.point_wrapper import PointWrapper
from utils import do_plot, generate_color, set_and_true, flip_arrow
import k3d
# Filter Warning - TODO: maybe fix this and other lines more properly later
# fig += k3d.mesh(verts, faces, group=group, color=MESH_COLOR, side='double', opacity=MESH_OPACITY, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row]) 
warnings.filterwarnings("ignore", message="Given trait value dtype.*")

logging.getLogger('matplotlib').setLevel(logging.WARNING)

TRAJ_WIDTH = 0.005
## COLORS
MESH_COLOR = 0xe31b23
MESH_OPACITY = 0.8

# Points
DEFAULT_POINT_SIZE = 0.01
DEFAULT_POINT_OPACITY = 1.0
DEFAULT_POINT_SHADER = 'flat'
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
        self.plot_n_shapes = self.config['plot_n_shapes']
        self.n_rows, self.n_cols = self.n_rows_cols()

        assert config['plot_n_shapes'] <= config['batch_size'], 'Can only plot up to batch_size shapes'
        assert config['plot_secondary_plots_every_n_epochs'] % config['plot_every_n_epochs'] == 0, 'plot_secondary_every_n_epochs must be a multiple of plot_every_n_epochs'

        # self.X0, self.X1, _ = get_meshgrid_in_domain(self.bounds)
        # self.y = None
        # self.Y = None
        self.verts = None
        self.faces = None
        self.epoch = 0

        self.fig_size = self.config['fig_size'] if 'fig_size' in config else fig_size

        if self.config['fig_save']:
            assert self.config["fig_path"] is not None, f"file_path must not be None if config['fig_save'] = True"
            cur_time_stamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            self.fig_parent_path = os.path.join(self.config['fig_path'], cur_time_stamp)
            Path(self.fig_parent_path).mkdir(parents=True, exist_ok=True)

    def do_plot(self, key=None):
        return do_plot(self.config, self.epoch, key=key)

    # def do_plot(self, key=None):
    #     """
    #     Checks if the plot specified by the key should be produced.
    #     First checks if the output is set: if not, no need to produce the plot.
    #     Then, checks if the key is set and true.
    #     If the key is not specified, only checks the global plotting behaviour.
    #     """
    #     is_output_set = self.config['fig_show'] or self.config['fig_save'] or self.config['fig_wandb']

    #     # if no output is set, no need to produce the plot
    #     if is_output_set == False:
    #         return False
        
    #     # if no key is specified, only check the global plotting behaviour
    #     if key is None:
    #         return is_output_set
        
    #     # otherwise, check if the key is set
    #     if key not in self.config:
    #         return False
        
    #     val = self.config[key]
    #     if isinstance(val, bool):
    #         return val
        
    #     if isinstance(val, dict):
    #         # if the val is a tuple, it is (do_plot, plot_interval)
    #         if val['on'] == False:
    #             return False
    #         else:
    #             assert val['interval'] % self.config['plot_every_n_epochs'] == 0, f'plot_interval must be a multiple of plot_every_n_epochs'
    #             return (self.epoch % val['interval'] == 0) or (self.epoch == self.config['max_epochs'])
        
    #     raise ValueError(f'Unknown value for key {key}: {val}')
    

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
        
        # if hasattr(self, 'cur_fig'):
        #     plt.close(self.cur_fig)
        # fig, axs = plt.subplots(self.n_rows, self.n_cols, figsize=self.fig_size)
        # axs = axs.ravel()
        # for i_shape in range(self.plot_n_shapes):
        #     ax = axs[i_shape]
        #     self._draw_base(ax, i_shape, levels=20)
        # self.cur_fig, self.cur_axs = fig, axs

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
            
        fig = k3d.plot(height=800)#, grid_visible=False, camera_fov=1.0)
        for i_shape in range(self.plot_n_shapes):
            i_col = (i_shape  % self.n_cols)
            i_row = (i_shape // self.n_cols)
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
                
        return self.handle_plot_show(fig, fig_label=fig_label)

    def plot_characterized_cps(self, pc, eigvecs_all, eigvals_all):
        """
        Plot the principal curvatures (eigenvectors of the Hessian) at the critical points.
        xc: [N, 2], critical points
        eigvecs: [N, 2, 2], eigenvectors. eigvecs[i, j, k] gives the j-th dimension of the k-th vector at the i-th critical point
        eigvals: [N, 1], eigenvalues
        """
        if not self.do_plot_only_every_n_epochs():
            return
        
        shape_grid = 1.5*(self.bounds[:,1] - self.bounds[:,0]) ## distance between the shape grid for plotting

        fig = k3d.plot(height=800)#, grid_visible=False, camera_fov=1.0)
        for i_shape in range(self.plot_n_shapes):
            i_col = (i_shape  % self.n_cols)
            i_row = (i_shape // self.n_cols)
            verts, faces = self.meshes[i_shape]
            group = f'Shape {i_shape}'
    
            xc = pc.pts_of_shape(i_shape)
            eigvecs = eigvecs_all[pc.get_idcs(i_shape)]
            eigvals = eigvals_all[pc.get_idcs(i_shape)]
            fig += k3d.mesh(verts, faces, group=group, color=MESH_COLOR, side='double', opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])

            # plot minima in blue, maxima in red and saddle points in green
            min_mask = (eigvals < 0).sum(1) == 0
            max_mask = (eigvals < 0).sum(1) == 3
            saddle_mask = ~(min_mask | max_mask)
            
            fig += k3d.points(xc[min_mask], group=group, point_size=DEFAULT_POINT_SIZE, color=MAXIMA_COLOR, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            fig += k3d.points(xc[max_mask], group=group, point_size=DEFAULT_POINT_SIZE, color=MINIMA_COLOR, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            fig += k3d.points(xc[saddle_mask], group=group, point_size=DEFAULT_POINT_SIZE, color=SADDLE_COLOR, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])

        return self.handle_plot_show(fig, fig_label='Characterized critical points')
                
    def plot_saddle_trajectories(self, pc, p_path_list, x_path_over_iters_all, indexes_all, multipliers_to_descend_all):
        """
        xc: [N, nx] critical points
        x_path_over_iters: [N_iter, N_paths, 3] trajectories of 4*N_saddles = N_paths. First half are to minima, second half are to maxima.
        """
        if not self.do_plot_only_every_n_epochs():
            return
        
        fig_label = "urfnet graph edges"
        x_path_over_iters_all = np.copy(x_path_over_iters_all)  ## make a copy to avoid messing with the original data

        offset_over_6_directions = np.arange(2 * self.config['nx']) * x_path_over_iters_all.shape[1]// (2 * self.config['nx'])

        shape_grid = 1.5*(self.bounds[:,1] - self.bounds[:,0]) ## distance between the shape grid for plotting
        fig = k3d.plot(height=800)#, grid_visible=False, camera_fov=1.0)
        for i_shape in range(self.plot_n_shapes):
            i_col = (i_shape  % self.n_cols)
            i_row = (i_shape // self.n_cols)
            xc = pc.pts_of_shape(i_shape)
            indexes = indexes_all[pc.get_idcs(i_shape)]
            path_idcs_single = p_path_list[0].get_idcs(i_shape)
            path_idcs = np.concatenate([path_idcs_single+off for off in offset_over_6_directions], axis=0)
            x_path_over_iters = x_path_over_iters_all[:, path_idcs]
            multipliers_to_descend = multipliers_to_descend_all[path_idcs]
            is_desc = multipliers_to_descend > 0
            
            verts, faces = self.meshes[i_shape]
            group = f'Shape {i_shape}'
            fig += k3d.mesh(verts, faces, group=group, color=MESH_COLOR, side='double', opacity=MESH_OPACITY, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            
            out_mask = (x_path_over_iters < self.bounds[:,0]).any(2) | (x_path_over_iters > self.bounds[:,1]).any(2)
            x_path_over_iters[out_mask] = np.nan
            
            # paths descending
            x_path_desc = x_path_over_iters[:,is_desc,:] # [N_iter, N_paths, 3]
            n_iter, n_paths_desc, _ = x_path_desc.shape
            x_path_flat = einops.rearrange(x_path_desc, 'n i j -> (i n) j') # [N_paths*N_iter, 3]
            idcs_desc = self.get_path_idcs(n_iter, n_paths_desc)
            fig += k3d.lines(x_path_flat, indices=idcs_desc, indices_type='segment', color=DESC_PATH_COLOR, group=group, width=TRAJ_WIDTH, shader=PATH_SHADER, opacity=DEFAULT_PATH_OPACITY,
                                translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            # paths ascending
            x_path_asc = x_path_over_iters[:,~is_desc,:] # [N_iter, N_paths, 3]
            n_iter, n_paths_asc, _ = x_path_asc.shape
            x_path_flat = einops.rearrange(x_path_asc, 'n i j -> (i n) j') # [N_paths*N_iter, 3]
            idcs_asc = self.get_path_idcs(n_iter, n_paths_asc)
            fig += k3d.lines(x_path_flat, indices=idcs_asc, indices_type='segment', color=ASC_PATH_COLOR, group=group, width=TRAJ_WIDTH, shader=PATH_SHADER, opacity=DEFAULT_PATH_OPACITY,
                                translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            
            if indexes is None:
                fig += k3d.points(xc, group=group, point_size=DEFAULT_POINT_SIZE, color=0x000000, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            else:
                fig += k3d.points(xc[indexes == 0], group=group, point_size=DEFAULT_POINT_SIZE, color=0x0000ff, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
                fig += k3d.points(xc[(indexes == 1) | (indexes == 2)], group=group, point_size=DEFAULT_POINT_SIZE, color=0x00ff00, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
                fig += k3d.points(xc[indexes == 3], group=group, point_size=DEFAULT_POINT_SIZE, color=0xff0000, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            
        return self.handle_plot_show(fig, fig_label=fig_label)

    def get_path_idcs(self, n_iter, n_paths_desc):
        idcs_desc = []
        for i in range(n_paths_desc):
            idcs = np.arange(i * n_iter, (i + 1) * n_iter - 1) # Index for each point in the path
            idcs_desc.extend(np.stack((idcs, idcs + 1), axis=1)) # Stack the indices to form pairs
        idcs_desc = np.vstack(idcs_desc)
        return idcs_desc
        
    def plot_descent_trajectories(self, pc: PointWrapper, x_path_over_iters_all, fig_label="Trajectories of finding CPs", plot_anyway=False):
        """
        xc: [N, 2] starting points
        x_path_over_iters: [N_iter, N_paths, 2] trajectories of 4*N_saddles = N_paths. First half are to minima, second half are to maxima.
        """
        if not plot_anyway and not self.do_plot_only_every_n_epochs():
            return

        x_path_over_iters_all = np.copy(x_path_over_iters_all)  ## make a copy to avoid messing with the original data

        shape_grid = 1.5*(self.bounds[:,1] - self.bounds[:,0]) ## distance between the shape grid for plotting
        fig = k3d.plot(height=800)#, grid_visible=False, camera_fov=1.0)
        for i_shape in range(self.plot_n_shapes):
            i_col = (i_shape  % self.n_cols)
            i_row = (i_shape // self.n_cols)
            shape_idcs = pc.get_idcs(i_shape)
            xc = pc.pts_of_shape(i_shape)
            x_path_over_iters = x_path_over_iters_all[:, shape_idcs]
            out_mask = (x_path_over_iters < self.bounds[:,0]).any(2) | (x_path_over_iters > self.bounds[:,1]).any(2)
            x_path_over_iters[out_mask] = np.nan
            
            verts, faces = self.meshes[i_shape]
            group = f'Shape {i_shape}'
            fig += k3d.mesh(verts, faces, group=group, color=MESH_COLOR, side='double', opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            fig += k3d.points(xc, group=group, name='End points', point_size=DEFAULT_POINT_SIZE, color=ENDPOINT_COLOR, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])

            # plot starting points
            fig += k3d.points(x_path_over_iters[0, :, :], name='Starting points', group=group, point_size=DEFAULT_POINT_SIZE, color=0x000000, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])

            n_iter, n_paths, _ = x_path_over_iters.shape
            x_path_flat = einops.rearrange(x_path_over_iters, 'i n j -> (n i) j') # [N_paths*N_iter, 3]
            idcs = self.get_path_idcs(n_iter, n_paths)
            fig += k3d.lines(x_path_flat, indices=idcs, name='Trajectories', indices_type='segment', color=DEFAULT_PATH_COLOR, group=group, width=TRAJ_WIDTH, shader=PATH_SHADER, opacity=DEFAULT_PATH_OPACITY,
                                translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            # fig += k3d.points(x_path_over_iters[kth_idcs, :, :], group=group, point_size=DEFAULT_POINT_SIZE, color=DEFAULT_PATH_COLOR, opacity=0.8, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])    
            
        return self.handle_plot_show(fig, fig_label=fig_label)

        
    def plot_edges(self, pc, indexes_all, edges_list, fig_label, show_v_id=True, **kwargs):
        if not self.do_plot_only_every_n_epochs():
            return

        shape_grid = 1.5*(self.bounds[:,1] - self.bounds[:,0]) ## distance between the shape grid for plotting
        fig = k3d.plot(height=800)#, grid_visible=False, camera_fov=1.0)
        for i_shape in range(self.plot_n_shapes):
            i_col = (i_shape  % self.n_cols)
            i_row = (i_shape // self.n_cols)
            xc = pc.pts_of_shape(i_shape)
            indexes = indexes_all[i_shape]
            edges = edges_list[i_shape]        

            verts, faces = self.meshes[i_shape]
            group = f'Shape {i_shape}'
            fig += k3d.mesh(verts, faces, group=group, color=MESH_COLOR, side='double', opacity=MESH_OPACITY, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])

            fig += k3d.points(xc[indexes==0], group=group, point_size=DEFAULT_POINT_SIZE, color=MAXIMA_COLOR, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            fig += k3d.points(xc[indexes==3], group=group, point_size=DEFAULT_POINT_SIZE, color=MINIMA_COLOR, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            fig += k3d.points(xc[(indexes==1) | (indexes==2)], group=group, point_size=DEFAULT_POINT_SIZE, color=SADDLE_COLOR, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            
            if show_v_id:
                for i, xc_ in enumerate(xc):
                    fig += k3d.text(str(i), xc_, color=BLACK, size=0.1, reference_point='cc', translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
                    
            if ('penalties' in kwargs) and (kwargs['penalties'][i_shape] is not None) and (len(kwargs['penalties'][i_shape]) > 0):
                assert 'pts_to_penalize' in kwargs, 'pts_to_penalize not set'
                for plty, pt in zip(kwargs['penalties'][i_shape], kwargs['pts_to_penalize'][i_shape]):
                    fig += k3d.text(f"{plty:0.2f}", pt, color=BLACK, size=0.1, reference_point='cc', translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
                    
            if edges is None or edges.shape[0] == 0:
                self.logger.debug(f'no edges to plot')
                continue
            
            fig += k3d.lines(xc, indices=edges, indices_type='segments', color=0x000000, width=0.01, shader=PATH_SHADER, opacity=DEFAULT_PATH_OPACITY,
                                translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            if 'edge_weights' in kwargs and len(kwargs['edge_weights']) > 0:
                edge_centers = xc[edges].mean(dim=1)
                for weight, center in zip(kwargs['edge_weights'][i_shape], edge_centers):
                    self.logger.debug(f"creating k3d text label for {weight}")
                    fig += k3d.text(f'{weight:0.1f}', center, color=BLACK, size=0.1, reference_point='cc', translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])

        return self.handle_plot_show(fig, fig_label=fig_label)

    def plot_hist(self, data, fig_label, log=True):
        if not self.do_plot_only_every_n_epochs():
            return
        
        fig, ax = plt.subplots(figsize=self.fig_size)

        if log:
            data = np.log10(data)
            np.nan_to_num(data, copy=False, neginf=0.0)

        ax.hist(data, bins=20, edgecolor='black')
        # ax.set_xscale('log')
        return self.handle_plot_show_matplotlib(fig, fig_label=fig_label)

    def plot_CCs_of_surfnet_graph(self, surfnet_graph_sub_CCs_list, pc, pyc_u_all, edges_list):
        """surfnet graph"""
        if not self.do_plot_only_every_n_epochs():
            return
        
        fig_label = "surfnet graph"
        fig, axs = plt.subplots(self.n_rows, self.n_cols, figsize=self.fig_size)
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        axs = axs.ravel()
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
        return self.handle_plot_show_matplotlib(fig, fig_label=fig_label)

    def plot_shape(self, constraint_pts_dict):
        fig_label = "Boundary"

        shape_grid = 1.5*(self.bounds[:,1] - self.bounds[:,0]) ## distance between the shape grid for plotting

        fig = k3d.plot(height=800)#, grid_visible=False, camera_fov=1.0)
        for i_shape in range(self.plot_n_shapes):
            i_col = (i_shape  % self.n_cols)
            i_row = (i_shape // self.n_cols)
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
            
        return self.handle_plot_show(fig, fig_label=fig_label)
        
    def plot_shape_and_points(self, pc: PointWrapper, fig_label):
        if not self.do_plot_only_every_n_epochs():
            return
    
        shape_grid = 1.5*(self.bounds[:,1] - self.bounds[:,0]) ## distance between the shape grid for plotting

        fig = k3d.plot(height=800)#, grid_visible=False, camera_fov=1.0)
        for i_shape in range(self.plot_n_shapes):
            i_col = (i_shape  % self.n_cols)
            i_row = (i_shape // self.n_cols)
            verts, faces = self.meshes[i_shape]
            group = f'Shape {i_shape}'
            x = pc.pts_of_shape(i_shape)
            fig += k3d.mesh(verts, faces, group=group, color=MESH_COLOR, side='double', opacity=MESH_OPACITY, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
            fig += k3d.points(x, group=group, point_size=DEFAULT_POINT_SIZE, color=0x000000, opacity=DEFAULT_POINT_OPACITY, shader=DEFAULT_POINT_SHADER, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
        return self.handle_plot_show(fig, fig_label=fig_label)
        
        
    def handle_plot_show(self, fig, fig_label=None, force_save=False, dim=3):
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

    def handle_plot_show_matplotlib(self, fig, fig_label=None, force_save=False):
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