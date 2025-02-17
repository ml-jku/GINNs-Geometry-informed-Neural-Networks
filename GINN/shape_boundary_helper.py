

import math
import einops
import numpy as np
import torch
from tqdm import trange
from GINN.numerical_boundary import get_grid_starting_pts
from GINN.speed.mp_manager import MPManager
from GINN.speed.timer import Timer
from models.net_w_partials import NetWithPartials
from models.point_wrapper import PointWrapper
from util.sample_utils import inflate_bounds
from util.misc import get_is_out_mask
import logging

import torch
import torch.nn.functional as F
from tqdm import trange
from models.point_wrapper import PointWrapper
from util.misc import get_is_out_mask
import logging
from train.losses import get_mean_curvature_normalized, get_gauss_curvature
from util.sample_utils import precompute_sample_grid


class ShapeBoundaryHelper:
    
    def __init__(self, nx, 
                 bounds, 
                 netp: NetWithPartials, 
                 mp_manager: MPManager, 
                 plotter, 
                 x_interface, 
                 n_points_surface,
                 equidistant_init_grid, ## boolean initial grid for surface points equidistant or same number of points for each dimension
                 level_set = 0.0,
                 do_uniform_resampling=True,
                 surf_pts_lr=0.01, ## learning rate for non-Newton optimizer
                 surf_pts_n_iter=10, # iterations of surface flow
                 surf_pts_prec_eps=1.0e-3,  ## precision threshold for early stopping surface flow and filtering the points 
                 surf_pts_converged_interval=1, ## how often to check the convergence
                 surf_pts_use_newton=True, ## whether to use Newton iteration or Adam
                 surf_pts_newton_clip=0.15, ## magnitude for clipping the Newton update
                 surf_pts_inflate_bounds_amount=0.05, ## inflate the (otherwise tight) bounding box by this fraction
                 surf_pts_uniform_n_iter=10, ## nof iterations for repelling the points
                 surf_pts_uniform_nof_neighbours=16, ## nof neighbors for knn
                 surf_pts_uniform_stepsize=0.75, ## step size for the repelling update; 0.75 worked well with 8 and 16 nns
                 surf_pts_uniform_n_iter_reproj=5, ## nof Newton-iterations for reprojecting the points
                 surf_pts_uniform_prec_eps=1.0e-3, ## precision for reprojection (similar to above)
                 surf_pts_uniform_min_count=1000, ## minimum number of points to redistribute. Less than this is meaningless
                 surf_pts_surpress_tqdm=True,
                 surf_pts_uniform_reproject_surpress_tqdm=True,
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
        self.surf_pts_do_uniform_resampling = do_uniform_resampling
        self.surf_pts_lr = surf_pts_lr
        self.surf_pts_n_iter = surf_pts_n_iter
        self.surf_pts_prec_eps = surf_pts_prec_eps
        self.surf_pts_converged_interval = surf_pts_converged_interval
        self.surf_pts_use_newton = surf_pts_use_newton
        self.surf_pts_newton_clip = surf_pts_newton_clip
        self.surf_pts_inflate_bounds_amount = surf_pts_inflate_bounds_amount
        self.surf_pts_uniform_n_iter = surf_pts_uniform_n_iter
        self.surf_pts_uniform_nof_neighbours = surf_pts_uniform_nof_neighbours
        self.surf_pts_uniform_stepsize = surf_pts_uniform_stepsize
        self.surf_pts_uniform_n_iter_reproj = surf_pts_uniform_n_iter_reproj
        self.surf_pts_uniform_prec_eps = surf_pts_uniform_prec_eps
        self.surf_pts_uniform_min_count = surf_pts_uniform_min_count
        self.surf_pts_surpress_tqdm = surf_pts_surpress_tqdm
        self.surf_pts_uniform_reproject_surpress_tqdm = surf_pts_uniform_reproject_surpress_tqdm
        
        self.logger = logging.getLogger('surf_pts_helper')
        self.bounds = inflate_bounds(self.bounds, amount=surf_pts_inflate_bounds_amount)
        self.grid_find_surface, self.grid_dist_find_surface, self.init_grid_resolution = precompute_sample_grid(self.surf_pts_nof_points, self.bounds, equidistant=equidistant_init_grid)
        
        self.p_surface = None
        self.knn_k = surf_pts_uniform_nof_neighbours # NOTE: more neighbors pushes the points more to edges, so might be favourable for smoothness 
        
    
    def get_surface_pts(self, z, interface_cutoff, plot, plot_max_shapes=None):
        with Timer.record('get_surface_pts'):
            p_surface = get_grid_starting_pts(len(z), self.grid_find_surface, self.grid_dist_find_surface)
            
            print(f'WARNING: surface flow is turned off as it does not work')
            success, p_surface = self._get_and_plot_surface_flow(p, z)
            if not success:
                return None, None

        if self.surf_pts_do_uniform_resampling and len(p_surface) > self.surf_pts_uniform_min_count:
            ## Stop redistributing if there are not enough points.
            ## Better return failure so that the integrals don't have a high variance
            with Timer.record('surf_pts_uniform_resampling'):
                success, p_surface = self.resample(p_surface, z, num_iters=self.surf_pts_uniform_n_iter)
                if not success:
                    return None, None

        # continue normally
        weights_surf_pts = torch.ones(len(p_surface)) / p_surface.data.shape[0]
        dist = torch.min(torch.norm(p_surface.data[:, None, :] - self.x_interface[None, :, :], dim=2), dim=1)[0]

        if plot:
            p_surface_plot = p_surface.select_w_shapes(incl_shapes=np.arange(min(len(z), plot_max_shapes)))
            z_plot = z[:min(len(z), plot_max_shapes)]

            y_x_surf = self.netp.grouped_no_grad_fwd('vf_x', p_surface_plot.data, p_surface_plot.z_in(z_plot)).squeeze(1)
            y_xx_surf = self.netp.grouped_no_grad_fwd('vf_xx', p_surface_plot.data, p_surface_plot.z_in(z_plot)).squeeze(1)
            # y_x_surf = self.netp.vf_x(p_surface_plot.data, p_surface_plot.z_in(z_plot)).squeeze(1)
            # y_xx_surf = self.netp.vf_xx(p_surface_plot.data, p_surface_plot.z_in(z_plot)).squeeze(1)
            
            mean_curvatures = get_mean_curvature_normalized(y_x_surf, y_xx_surf)
            gauss_curvatures = get_gauss_curvature(y_x_surf, y_xx_surf)
            E_strain = (2*mean_curvatures)**2 - 2*gauss_curvatures

            self.mpm.plot(self.plotter.plot_shape_and_points, 'plot_surface_points',
                    arg_list=[p_surface_plot.detach().cpu().numpy(), 'Surface points interface cutoff'], kwargs_dict=dict(point_attribute=(10*(dist > interface_cutoff)).detach().cpu().numpy()))

            self.mpm.plot(self.plotter.plot_shape_and_points, 'plot_surface_points', 
                                    arg_list=[p_surface_plot.detach().cpu().numpy(), 'Weighted Surface points'], kwargs_dict=dict(point_attribute=(torch.log10(E_strain + 1)).detach().cpu().numpy()))
        
        if interface_cutoff > 0:
            mask = dist > interface_cutoff
            p_surface._select_w_mask(incl_mask=mask)
            weights_surf_pts = torch.ones(len(p_surface)) / p_surface.data.shape[0]

        return p_surface, weights_surf_pts
    
    def _get_and_plot_surface_flow(self, p, z, plot=False):
        
        success, tup = self.flow_to_surface_pts(p, z, 
            lr=self.surf_pts_lr,
            n_iter=self.surf_pts_n_iter,
            plot_descent=plot,
            use_newton=self.surf_pts_use_newton,
            surpress_tqdm=self.surf_pts_surpress_tqdm,
            )

        if not success:
            self.logger.debug(f'No surface points found')
            return False, None        
        p_surface, x_path_over_iters = tup
        if plot:
            self.mpm.plot(self.plotter.plot_descent_trajectories, 'plot_surface_descent', [p.detach().cpu().numpy(), x_path_over_iters.cpu().numpy()])

        return True, p_surface
    
    def flow_to_surface_pts(self, p, z, lr, n_iter, 
                            plot_descent, 
                            filter_thr=1e-3, 
                            newton_clip=0.15, 
                            min_count=100, 
                            use_sgd=False, 
                            use_newton=True, 
                            surpress_tqdm=False):
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

        ## Initialize plotting
        x_path_over_iters = None
        if plot_descent:
            x_path_over_iters = torch.full([n_iter + 1, len(p), self.nx], torch.nan)
            idxs_in_orig = torch.arange(0, len(p))

        ## Initialize points and optimizer
        if use_newton:
            p_in = p
        else:
            p.data.requires_grad = True
            opt = torch.optim.Adam([p.data], lr=lr)
            if use_sgd:
                opt = torch.optim.SGD([p.data], lr=lr)
    
        self.logger.debug(f'Flowing {len(p)} points to the surface')
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
            self.logger.debug(f'Out of bounds: {out_mask.sum()}, remaining: {len(p_in)}')

            if len(p_in) == 0:
                self.logger.debug(f'No surf_pts_n_iter points found in the domain')
                return False, None

            ## Main update
            if use_newton:
                with torch.no_grad():
                    z_ = p_in.z_in(z)
                    # to adjust for the level set, we say y_adjusted = y - level_set; as y_adjusted' = y', the rest is the same
                    y = self.netp.grouped_no_grad_fwd('vf', p_in.data, z_).squeeze(1)
                    y_x = self.netp.grouped_no_grad_fwd('vf_x', p_in.data, z_).squeeze(1)
                    update = y_x * (torch.clip(y - self.level_set, -newton_clip, newton_clip)/y_x.norm(dim=1))[:,None]
                    p_in.data = p_in.data - update

                    ## For compatibility with remaining code
                    y_in = y

                    ## Logging
                    if not surpress_tqdm:
                        loss = (y_in - self.level_set).square().mean()
                        pbar.set_description(f"Flow to surface points: {len(p_in)}/{len(p)}; {loss.item():.2e}")
            else:
                # note: needs grad
                y_in = self.netp.f(p_in.data, p_in.z_in(z)).squeeze(1)  ## [bx]
                
                # L2 loss works better than L1 loss
                loss = (y_in - self.level_set).square().mean()
                if torch.isnan(loss):
                    self.logger.debug(f'Early stop "Finding surface points" at it {i} due to nan loss')

                loss.backward()
                opt.step()
                if not surpress_tqdm:
                    pbar.set_description(f"Flow to surface points: {len(p_in)}/{len(p)}; {loss.item():.2e}")
            
            ## Early stopping
            if i % self.surf_pts_converged_interval == 0:
                # stop if |points| < thresh
                if (torch.abs(y_in - self.level_set) < self.surf_pts_prec_eps).all():
                    self.logger.debug(f'Early stop "Finding surface points" at it {i}')
                    break
                
        
        ## Filter non-converged points
        converged_mask = torch.abs(y_in - self.level_set) < filter_thr
        p_in = p_in.select_w_mask(incl_mask=converged_mask)
        self.logger.debug(f'Converged points: {converged_mask.sum()}, while converged_mask filtered out {len(converged_mask) - converged_mask.sum()}')

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

    def get_normals_nograd(self, p, z, invert=False):
        f_x = self.netp.grouped_no_grad_fwd('vf_x', p.data, p.z_in(z)).squeeze(1)  ## [bx nx]
        # with torch.no_grad():
        #     f_x = self.netp.vf_x(p.data, p.z_in(z)).squeeze(1)
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
    
    def resample(self, points_init, 
                 z, 
                 num_iters=0, 
                 filter_thr_reproj=1e-3, ## lower thr requires more n_iter
                 debug=True):

        ## Initialize parameters
        n_iter_reproj = self.surf_pts_uniform_n_iter_reproj
        stepsize = self.surf_pts_uniform_stepsize

        for i_iter in range(num_iters):
            if debug:
                if i_iter>0:
                    self.logger.debug(f'iter: {i_iter} \t density: {density_w.mean().item():.3f} \t nof pts: {len(points_init)}')

            normals_init = self.get_normals_nograd(points_init, z)
            for i_shape in range(points_init.bz):

                with Timer.record('surf_pts_uniform_move'):
                    points = points_init.pts_of_shape(i_shape)
                    if len(points) == 0:
                        continue
                    
                    normals = normals_init.pts_of_shape(i_shape)
                    num_points = points.shape[0]
                    
                    ## NOTE: not sure if this should be recomputed every iteration if the nof points doesn't change much
                    diag = (points.view(-1, self.nx).max(dim=0).values - points.view(-1, self.nx).min(0).values).norm().item()
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
                    surpress_tqdm=self.surf_pts_uniform_reproject_surpress_tqdm,
                    filter_thr=filter_thr_reproj,
                    )
                if success:
                    points_init, _ = ret
                else:
                    self.logger.debug("No points left after reprojection. Try decreasning the update size, increasing the number of reprojection iterations or decreasing the filtering threshold")
                    return False, None 
                if len(points_init) < self.surf_pts_uniform_min_count:
                    ## Stop redistributing if there are not enough points.
                    ## Better return failure so that the integrals don't have a high variance
                    False, None
        
        return True, points_init
