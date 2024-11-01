

import torch
from tqdm import trange
from GINN.speed.mp_manager import MPManager
from GINN.speed.timer import Timer
from models.point_wrapper import PointWrapper
from util.sample_utils import inflate_bounds
from util.misc import get_is_out_mask, set_and_true
import logging

import torch
import torch.nn.functional as F
from tqdm import trange
from models.point_wrapper import PointWrapper
from util.misc import get_is_out_mask, set_and_true, set_else_default
import logging
from train.losses import get_mean_curvature_normalized, get_gauss_curvature
from util.sample_utils import precompute_sample_grid


class ShapeBoundaryHelper:
    
    def __init__(self, config, netp, mp_manager: MPManager, plot_helper, x_interface, device):
        self.config = config
        self.netp = netp.detach()
        self.mpm = mp_manager
        self.logger = logging.getLogger('surf_pts_helper')
        self.plotter = plot_helper
        self.bounds = self.config['bounds'].to(device)
        self.bounds = inflate_bounds(self.bounds, amount=set_else_default('surf_pts_inflate_bounds_amount', self.config, 0.05))
        self.grid_find_surface, self.grid_dist_find_surface = precompute_sample_grid(self.config['surf_pts_nof_points'], self.bounds)
        
        self.p_surface = None
        self.x_interface = x_interface
        self.knn_k = set_else_default('surf_pts_uniform_nof_neighbours', config, 16)
        # NOTE: more neighbors pushes the points more to edges, so might be favourable for smoothness 
    
    def get_surface_pts(self, z):
        with Timer.record('get_surface_pts'):
            success, p_surface = self._get_and_plot_surface_flow(z)
            if not success:
                return None, None

        if set_and_true('surf_pts_do_uniform_resampling', self.config) and len(p_surface) > set_else_default('surf_pts_uniform_min_count', self.config, 1000): 
            ## Stop redistributing if there are not enough points.
            ## Better return failure so that the integrals don't have a high variance
            with Timer.record('surf_pts_uniform_resampling'):
                success, p_surface = self.resample(p_surface, z, num_iters=set_else_default('surf_pts_uniform_n_iter', self.config, 10))
                if not success:
                    return None, None

        weights_surf_pts = torch.ones(len(p_surface)) / p_surface.data.shape[0]
        dist = torch.min(torch.norm(p_surface.data[:, None, :] - self.x_interface[None, :, :], dim=2), dim=1)[0]

        y_x_surf = self.netp.vf_x(p_surface.data, p_surface.z_in(z)).squeeze(1)
        y_xx_surf = self.netp.vf_xx(p_surface.data, p_surface.z_in(z)).squeeze(1)
        
        mean_curvatures = get_mean_curvature_normalized(y_x_surf, y_xx_surf)
        gauss_curvatures = get_gauss_curvature(y_x_surf, y_xx_surf)
        E_strain = (2*mean_curvatures)**2 - 2*gauss_curvatures

        self.mpm.plot(self.plotter.plot_shape_and_points, 'plot_surface_points',
                arg_list=[p_surface.detach().cpu().numpy(), 'Surface points interface cutoff'], kwargs_dict=dict(point_attribute=(10*(dist > self.config['exclude_surface_points_close_to_interface_cutoff'])).detach().cpu().numpy()))

        self.mpm.plot(self.plotter.plot_shape_and_points, 'plot_surface_points', 
                                arg_list=[p_surface.detach().cpu().numpy(), 'Weighted Surface points'], kwargs_dict=dict(point_attribute=(torch.log10(E_strain + 1)).detach().cpu().numpy()))
        if set_and_true('exclude_surface_points_close_to_interface', self.config):
            mask = dist > self.config['exclude_surface_points_close_to_interface_cutoff']
            p_surface._select_w_mask(incl_mask=mask)
            E_strain = E_strain[mask]
            weights_surf_pts = torch.ones(len(p_surface)) / p_surface.data.shape[0]

        return p_surface, weights_surf_pts
    
    def _get_and_plot_surface_flow(self, z):
        
        p = self.get_grid_starting_pts(self.grid_find_surface, self.grid_dist_find_surface)
        success, tup = self.flow_to_surface_pts(p, z, 
            lr=self.config['surf_pts_lr'],
            n_iter=self.config['surf_pts_n_iter'],
            plot_descent=self.plotter.do_plot('plot_surface_descent'),
            use_newton=self.config['surf_pts_use_newton'],
            surpress_tqdm=set_else_default('surf_pts_surpress_tqdm', self.config, False),
            )

        if not success:
            self.logger.debug(f'No surface points found')
            return False, None        
        p_surface, x_path_over_iters = tup
        if self.plotter.do_plot('plot_surface_descent'):
            self.mpm.plot(self.plotter.plot_descent_trajectories, 'plot_surface_descent', [p.detach().cpu().numpy(), x_path_over_iters.cpu().numpy()])

        return True, p_surface
    
    def flow_to_surface_pts(self, p, z, lr, n_iter, plot_descent, filter_thr=None, newton_clip=None, min_count=None, use_sgd=False, use_newton=True, surpress_tqdm=False):
        """
        A simple optimization loop to let starting points p flow to zero.
        NOTE: Adam/SGD is kept for historic reasons, but going forward we might want to
        either split or remove it as the current code is a bit unreadable.
        The main difference between Adam and Newton:
        Adam requires to register the variables, which stay fixed size: filtering just selects a subset for evaluation and updating.
        Newton update is manual, so we can always throw away points.
        """

        ## Filter far away from surface so we get a more uniform distribution and need less iterations
        # y = self.netp.f(p.data, p.z_in(z)).squeeze(1)
        # init_mask = torch.abs(y) < 5e-2
        # p = p.select_w_mask(incl_mask=init_mask)

        ## Initialize parameters
        if filter_thr is None:
            filter_thr = set_else_default('surf_pts_prec_eps', self.config, 1e-3)
        if newton_clip is None:
            newton_clip = set_else_default('surf_pts_newton_clip', self.config, 0.15)
        if min_count is None:
            min_count = set_else_default('surf_pts_uniform_min_count', self.config, 100)

        ## Initialize plotting
        x_path_over_iters = None
        if plot_descent:
            x_path_over_iters = torch.full([n_iter + 1, len(p), self.config['nx']], torch.nan)
            idxs_in_orig = torch.arange(0, len(p))

        ## Initialize points and optimizer
        if use_newton:
            p_in = p
        else:
            p.data.requires_grad = True
            opt = torch.optim.Adam([p.data], lr=lr)
            if use_sgd:
                opt = torch.optim.SGD([p.data], lr=lr)
    
        ## Iterate
        for i in (pbar := trange(n_iter, disable=surpress_tqdm)):

            ## Mask
            if use_newton:
                out_mask = get_is_out_mask(p_in.data, self.bounds)
                p_in = p_in.select_w_mask(incl_mask=~out_mask)
                if plot_descent:
                    idxs_in_orig = idxs_in_orig[~out_mask]
                    x_path_over_iters[i][idxs_in_orig] = p_in.data.detach()
            else:
                opt.zero_grad()
                out_mask = get_is_out_mask(p.data, self.bounds)
                p_in = p.select_w_mask(incl_mask=~out_mask)
                if plot_descent:
                    x_path_over_iters[i] = p.data.detach()

            if len(p_in) == 0:
                self.logger.debug(f'No surf_pts_n_iter points found in the domain')
                return False, None

            ## Main update
            if use_newton:
                with torch.no_grad():
                    z_ = p_in.z_in(z)
                    y = self.netp.f(p_in.data, z_).squeeze(1)
                    y_x = self.netp.vf_x(p_in.data, z_).squeeze(1)
                    update = y_x * (torch.clip(y, -newton_clip, newton_clip)/y_x.norm(dim=1))[:,None]
                    p_in.data = p_in.data - update

                    ## For compatibility with remaining code
                    y_in = y

                    ## Logging
                    if not surpress_tqdm:
                        loss = y_in.square().mean()
                        pbar.set_description(f"Flow to surface points: {len(p_in)}/{len(p)}; {loss.item():.2e}")
            else:
                y_in = self.netp.f(p_in.data, p_in.z_in(z)).squeeze(1)  ## [bx]
                
                # L2 loss works better than L1 loss
                loss = y_in.square().mean()
                if torch.isnan(loss):
                    self.logger.debug(f'Early stop "Finding surface points" at it {i} due to nan loss')

                loss.backward()
                opt.step()
                if not surpress_tqdm:
                    pbar.set_description(f"Flow to surface points: {len(p_in)}/{len(p)}; {loss.item():.2e}")
            
            ## Early stopping
            if i % self.config['surf_pts_converged_interval'] == 0:
                # stop if |points| < thresh
                if (torch.abs(y_in) < self.config['surf_pts_prec_eps']).all():
                    self.logger.debug(f'Early stop "Finding surface points" at it {i}')
                    break
                
        
        ## Filter non-converged points
        converged_mask = torch.abs(y_in) < filter_thr
        p_in = p_in.select_w_mask(incl_mask=converged_mask)

        ## Exit early if no points are left
        if len(p_in)<min_count:
            self.logger.debug(f'Only {len(p_in)} surface points found, not continuing')
            return False, None

        ## Handle the last iteration of plotting
        if plot_descent:
            if use_newton:
                idxs_in_orig = idxs_in_orig[converged_mask]
                x_path_over_iters[i+1][idxs_in_orig] = p_in.data.detach()
            else:
                x_path_over_iters[i+1] = p.data.detach()
            x_path_over_iters = x_path_over_iters[:i+2] ## remove the unfilled part due to early stopping
        
        ## Disable gradient tracking for Adam
        if not use_newton:
            p_in = p_in.detach()
            p_in.data.requires_grad = False

        return True, (p_in, x_path_over_iters)

    def get_normals(self, p, z, invert=False):
        f_x = self.netp.vf_x(p.data, p.z_in(z)).squeeze(1)  ## [bx nx]
        if invert:
            f_x = -f_x
        p_normals = PointWrapper(f_x, map=p.get_map())
        p_normals.data = F.normalize(p_normals.data, dim=-1)
        return p_normals
        
    def get_nn_idcs(self, x, k):
        dist = torch.cdist(x, x, compute_mode='use_mm_for_euclid_dist')
        # dist = (x.unsqueeze(1) - x.unsqueeze(0)).norm(dim=-1)
        idcs = dist.argsort(dim=-1)[:, 1:k+1]
        return idcs
    
    def resample(self, points_init, z, num_iters=0, debug=True):
        """
        """

        ## Initialize parameters
        n_iter_reproj = set_else_default('surf_pts_uniform_n_iter_reproj', self.config, 5)
        filter_thr_reproj = set_else_default('surf_pts_uniform_filter_thr_reproj', self.config, 1e-3) ## lower thr requires more n_iter
        stepsize = set_else_default('surf_pts_uniform_stepsize', self.config, 0.75) ## .75 worked well with 8 and 16 nns

        for i_iter in range(num_iters):
            if debug:
                if i_iter>0:
                    self.logger.debug(f'iter: {i_iter} \t density: {density_w.mean().item():.3f} \t nof pts: {len(points_init)}')

            for i_shape in range(len(points_init.get_map())):

                with Timer.record('surf_pts_uniform_move'):
                    points = points_init.pts_of_shape(i_shape)
                    if len(points) == 0:
                        continue
                    
                    normals_init = self.get_normals(points_init, z)
                    normals = normals_init.pts_of_shape(i_shape)
                    num_points = points.shape[0]
                    
                    ## NOTE: not sure if this should be recomputed every iteration if the nof points doesn't change much
                    diag = (points.view(-1, self.config['nx']).max(dim=0).values - points.view(-1, self.config['nx']).min(0).values).norm().item()
                    if diag < 1e-6: ## Fail if the diagonal is too small
                        return False, None
                    inv_sigma_spatial = num_points / diag

                    knn_indices = self.get_nn_idcs(points, self.knn_k) # [n_points, k]
                    knn_nn = points[knn_indices] # [n_points, k, 3]                
                    knn_diff = points.unsqueeze(1) - knn_nn  # [n_points, k, 3]
                    knn_dists_sq = torch.sum(knn_diff**2, dim=-1)  # [n_points, k]
                    spatial_w = torch.exp(-knn_dists_sq * inv_sigma_spatial)  # [n_points, k]
                    move = torch.sum(spatial_w[..., None] * knn_diff, dim=-2)

                    if debug:
                        ## Store the previous points for debugging
                        density_w = torch.sum(spatial_w, dim=-1, keepdim=True)  # [n_points, 1] ## NOTE: can change sum to mean to make invariant to number of neighbors 

                    ## Project the move onto the tangential plane
                    move -= (move * normals) * normals
                    ## Scale the update
                    move *= stepsize ## the update size is a hyperparameter. Larger steps needs better reprojection

                    ## Update the points
                    points += move 
                    points_init.set_pts_of_shape(i_shape, points)
            
            with Timer.record('surf_pts_uniform_reproject'):
                ## Reproject after points have been moved for each shape
                ## NOTE: the majoriy of time is spent here
                success, ret = self.flow_to_surface_pts(
                    points_init,
                    z,
                    lr=None,
                    n_iter=n_iter_reproj,
                    plot_descent=False, 
                    use_newton=True, 
                    surpress_tqdm=set_else_default('surf_pts_uniform_reproject_surpress_tqdm', self.config, True),
                    filter_thr=filter_thr_reproj,
                    )
                if success:
                    points_init, _ = ret
                else:
                    self.logger.debug("No points left after reprojection. Try decreasning the update size, increasing the number of reprojection iterations or decreasing the filtering threshold")
                    return False, None 
                if len(points_init)<set_else_default('surf_pts_uniform_min_count', self.config, 1000):
                    ## Stop redistributing if there are not enough points.
                    ## Better return failure so that the integrals don't have a high variance
                    False, None
        
        return True, points_init


    def get_grid_starting_pts(self, x_grid, grid_dist):
        '''
        Create grid once at the beginning.
        Translate the grid by a random offset.
        '''
        ## Translate the grid by a random offset
        xc_offset = torch.rand((self.config['ginn_bsize'], self.config['nx'])) * grid_dist  # bz nx

        # x_grid: [n_points nx]
        x = x_grid.unsqueeze(0) + xc_offset.unsqueeze(1)  # bz n_points nx

        ## Translate each point by a random offset
        x += torch.randn(x_grid.shape) * grid_dist / 3

        return PointWrapper.create_from_equal_bx(x)