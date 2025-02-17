

import numpy as np
import torch
from GINN.numerical_boundary import find_boundary_points_numerically_with_binsearch, get_grid_starting_pts
from GINN.speed.mp_manager import MPManager
from GINN.speed.timer import Timer
from models.net_w_partials import NetWithPartials
import logging

import torch
import logging
from train.losses import get_mean_curvature_normalized, get_gauss_curvature
from util.sample_utils import precompute_sample_grid


class NumericalBoundaryHelper:
    
    def __init__(self, nx, 
                 bounds, 
                 netp: NetWithPartials, 
                 mp_manager: MPManager, 
                 plotter, 
                 x_interface, 
                 n_points_surface,
                 equidistant_init_grid, ## boolean initial grid for surface points equidistant or same number of points for each dimension
                 level_set,
                 bin_search_steps,
                 **kwargs):
        self.nx = nx
        self.bounds = bounds
        self.netp = netp.detach()
        self.mpm = mp_manager
        self.plotter = plotter
        self.level_set = level_set
        self.x_interface = x_interface
        
        # Surface point specifics
        self.surf_pts_nof_points = n_points_surface
        self.bin_search_steps = bin_search_steps
        
        self.logger = logging.getLogger('surf_pts_helper')
        self.grid_find_surface, self.grid_dist_find_surface, self.init_grid_resolution = precompute_sample_grid(self.surf_pts_nof_points, self.bounds, equidistant=equidistant_init_grid)
        
    def get_surface_pts(self, z, interface_cutoff, plot, plot_max_shapes=None):
        with Timer.record('get_surface_pts'):
            with torch.no_grad():
                level_set = self.level_set
                # TODO: can explore this further
                # if self.flexible_levelset:
                #     p_grid = get_grid_starting_pts(len(z), self.grid_find_surface, self.grid_dist_find_surface)
                #     y = self.netp(p_grid.data, p_grid.z_in(z)).squeeze(1)
                #     # level_set = (torch.max(y) + torch.min(y)) / 2
                #     # find 8% percentile value
                #     level_set = torch.quantile(y, q=1-self.vol_frac)
                #     print(f'Flexible level set: {level_set}')
                    
                success, (p_surface, _) = find_boundary_points_numerically_with_binsearch(netp=self.netp, z=z, n_steps=self.bin_search_steps, x_grid=self.grid_find_surface, x_grid_dist=self.grid_dist_find_surface, 
                                                                                level_set=level_set, resolution=self.init_grid_resolution)
            if not success:
                return None, None
            
        # continue normally
        dist = torch.min(torch.norm(p_surface.data[:, None, :] - self.x_interface[None, :, :], dim=2), dim=1)[0]

        if plot:
            p_surface_plot = p_surface.select_w_shapes(incl_shapes=np.arange(min(len(z), plot_max_shapes)))
            z_plot = z[:min(len(z), plot_max_shapes)]

            y_x_surf = self.netp.grouped_no_grad_fwd('vf_x', p_surface_plot.data, p_surface_plot.z_in(z_plot)).squeeze(1)
            y_xx_surf = self.netp.grouped_no_grad_fwd('vf_xx', p_surface_plot.data, p_surface_plot.z_in(z_plot)).squeeze(1)
            
            mean_curvatures = get_mean_curvature_normalized(y_x_surf, y_xx_surf)
            gauss_curvatures = get_gauss_curvature(y_x_surf, y_xx_surf)
            E_strain = (2*mean_curvatures)**2 - 2*gauss_curvatures

            self.mpm.plot(self.plotter.plot_shape_and_points, 'plot_surface_points',
                    arg_list=[p_surface_plot.detach().cpu().numpy(), 'Surface points interface cutoff'], kwargs_dict=dict(point_attribute=(10*(dist > interface_cutoff)).detach().cpu().numpy()))

            self.mpm.plot(self.plotter.plot_shape_and_points, 'plot_surface_points', 
                                    arg_list=[p_surface_plot.detach().cpu().numpy(), 'Weighted Surface points'], kwargs_dict=dict(point_attribute=(torch.log10(E_strain + 1)).detach().cpu().numpy()))
        
        weights_surf_pts = torch.ones(len(p_surface)) / p_surface.data.shape[0]
        if interface_cutoff > 0:
            mask = dist > interface_cutoff
            p_surface._select_w_mask(incl_mask=mask)
            weights_surf_pts = torch.ones(len(p_surface)) / p_surface.data.shape[0]
            
        if len(p_surface) == 0:
            return None, None

        return p_surface, weights_surf_pts