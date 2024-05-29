

import torch
from tqdm import trange
from GINN.helpers.mp_manager import MPManager
from models.point_wrapper import PointWrapper
from utils import get_is_out_mask, precompute_sample_grid, set_and_true
from GINN.helpers.timer_helper import TimerHelper
import logging

class ShapeBoundaryHelper:
    
    def __init__(self, config, netp, mp_manager: MPManager, plot_helper, timer_helper: TimerHelper, x_interface, device):
        self.config = config
        self.netp = netp.detach()
        self.mpm = mp_manager
        self.logger = logging.getLogger('surf_pts_helper')
        self.plotter = plot_helper
        self.timer_helper = timer_helper
        self.grid_find_surface, self.grid_dist_find_surface = precompute_sample_grid(self.config['n_points_find_surface'], self.config['bounds'])
        self.bounds = self.config['bounds'].to(device)
        
        self.record_time = timer_helper.record        
        self.p_surface = None
        self.x_interface = x_interface
        
    
    def get_surface_pts(self, z):
        success, p_surface = self._get_and_plot_surface_flow(z)
        if not success:
            return None, None
        
        weights_surf_pts = torch.ones(len(p_surface)) / p_surface.data.shape[0]
        if set_and_true('reweigh_surface_pts_close_to_interface', self.config):
            dist = torch.min(torch.norm(p_surface.data[:, None, :] - self.x_interface[None, :, :], dim=2), dim=1)[0]            
            dist = torch.clamp(dist, max=self.config['reweigh_surface_pts_close_to_interface_cutoff'])
            weights_surf_pts = torch.pow(dist, self.config['reweigh_surface_pts_close_to_interface_power'])
            weights_surf_pts = weights_surf_pts / weights_surf_pts.sum()  ## normalize to sum to 1
            
        self.mpm.plot(self.plotter.plot_shape_and_points, 'plot_surface_points', 
                                arg_list=[p_surface.detach().cpu(), 'Surface points'])
        return p_surface, weights_surf_pts
    
    def _get_and_plot_surface_flow(self, z):
        
        with self.record_time('cp_helper: flow_to_surface_points'):
            p = self.get_grid_starting_pts(self.grid_find_surface, self.grid_dist_find_surface)
            success, tup = self.flow_to_surface_pts(p, z, plot_descent=self.plotter.do_plot('plot_surface_descent'))

        if not success:
            self.logger.debug(f'No surface points found')
            return False, None        
        p_surface, x_path_over_iters = tup
        if self.plotter.do_plot('plot_surface_descent'):
            self.mpm.plot(self.plotter.plot_descent_trajectories, 'plot_surface_descent', [p.detach().cpu().numpy(), x_path_over_iters.cpu().numpy()])
        
        return True, p_surface
    

    def flow_to_surface_pts(self, p, z, plot_descent):
        """A simple optimization loop to let starting points sampled from the grid flow to zero."""

        x_path_over_iters = None
        if plot_descent:
            x_path_over_iters = torch.zeros([self.config['find_surface_pts_n_iter'] + 1, len(p), self.config['nx']])

        p.data.requires_grad = True
        opt = torch.optim.Adam([p.data], lr=self.config['lr_find_surface'], betas=self.config['adam_betas_find_surface'])
        
        for i in (pbar := trange(self.config['find_surface_pts_n_iter'])):
            opt.zero_grad()

            out_mask = get_is_out_mask(p.data, self.bounds)
            p_in = p.select_w_mask(incl_mask=~out_mask)
            if len(p_in) == 0:
                self.logger.debug(f'No find_surface_pts_n_iter points found in the domain')
                return False, None

            y_in = self.netp.f(p_in.data, p_in.z_in(z)).squeeze(1)  ## [bx]
            
            if plot_descent:
                x_path_over_iters[i] = p.data.detach()
                
            # L2 loss works better than L1 loss
            loss = y_in.square().mean()
            if torch.isnan(loss):
                self.logger.debug(f'Early stop "Finding surface points" at it {i} due to nan loss')

            loss.backward()
            opt.step()
            pbar.set_description(f"Flow to surface points: {loss.item():.2e}")
            
            if i % self.config['find_surface_pts_converged_interval'] == 0:
                # stop if |points| < thresh
                if (torch.abs(y_in) < self.config['find_surface_pts_prec_eps']).all():
                    self.logger.debug(f'Early stop "Finding surface points" at it {i}')
                    break
        
        out_mask = torch.abs(y_in) < self.config['find_surface_pts_prec_eps']
        p_in = p_in.select_w_mask(incl_mask=out_mask)
        if len(p_in) == 0:
            self.logger.debug(f'No surface points found')
            return False, None
        
        if plot_descent:
            x_path_over_iters[i] = p.data.detach()
            x_path_over_iters = x_path_over_iters[:i+1] ## remove the unfilled part due to early stopping
        
        p_in = p_in.detach()
        p_in.data.requires_grad = False
        return True, (p_in, x_path_over_iters)

    def get_grid_starting_pts(self, x_grid, grid_dist):
        '''
        Create grid once at beginning.
        Translate the grid by a random offset.
        '''
        xc_offset = torch.rand((self.config['batch_size'], self.config['nx'])) * grid_dist  # bz nx
        # x_grid: [n_points nx]
        x = x_grid.unsqueeze(0) + xc_offset.unsqueeze(1)  # bz n_points nx
        return PointWrapper.create_from_equal_bx(x)