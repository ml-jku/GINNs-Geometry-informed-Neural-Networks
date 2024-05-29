from contextlib import nullcontext
import logging
import math

import torch
from tqdm import trange
from sklearn.cluster import DBSCAN
from GINN.helpers.mp_manager import MPManager

from models.point_wrapper import PointWrapper
from GINN.helpers.timer_helper import TimerHelper
from utils import do_plot, get_is_out_mask, precompute_sample_grid


class CriticalPointsHelper():

    def __init__(self, config, netp, mp_manager: MPManager, plot_helper, timer_helper: TimerHelper, device):
        self.config = config
        ## Make the parameters not request gradient computation by creating a detached copy.
        ## The copy is shallow, meaning it shares memory, and is tied to parameter updates in the outer loop.
        self.netp = netp.detach()
        self.mpm = mp_manager
        self.logger = logging.getLogger('cp_helper')
        
        self.bounds = self.config['bounds']
        self.x_grid, self.x_grid_dist = precompute_sample_grid(self.config['n_points_find_cps'], self.bounds.to(device))
        self.x_grid_dist_squared_norm = self.x_grid_dist.square().sum()
        self.pc = None
        self.edges_z_list = None
        self.indexes = None
        self.p_surface = None
        
        self.plotter = plot_helper
        self.record_time = timer_helper.record if timer_helper is not None else nullcontext

    def get_critical_points(self, z, epoch, **kwargs):
        
        ## (-1) Reuse previous graph
        # reuse_output = self.reuse_graph()
        # if reuse_output is not None:
        #     return
        if epoch%self.config['recalc_cps_every_n_epochs']==0:
            ## Recompute the graph on initial epoch and every n epochs
            pass
        else:
            ## Reuse the graph
            self.logger.info(f'=== Reusing the graph ===')
            if self.pc is None or len(self.pc)==0 or self.edges_z_list is None or len(self.edges_z_list)==0:
                self.logger.info(f"(-1) FAIL: no valid pc or edges")
                ## If there are no previously known critical points, we cannot update them.
                ## We have two choices:
                ## (A) skip computing the graph and the scc_loss for the next (at most) recalc_cps_every_n_epochs
                return False, None
                ## (B) try to compute the graph at every stage.
                ## This is cleaner, but more compute intense.
            
            with self.record_time('cp_helper: adjusting CPs'):
                pc_, _, x_path_over_iters = self._gd_to_min_grad_norm(z, self.pc, epoch, self.config['lr_adjust_cps'], early_stop=True, plot_descent=self.plotter.do_plot('plot_cp_descent_refinement'))
            self.mpm.plot(self.plotter.plot_descent_trajectories, 'plot_cp_descent_refinement',
                                          [pc_.cpu().numpy(), x_path_over_iters.cpu().numpy()], 
                                          {'fig_label':'Trajectories of refining CPs', 'plot_anyway':True})

            is_update_valid = len(self.pc) == len(pc_)
            ## TODO: another condition is point consistency:
            ## 1) distance in geometric space should not exceed some maximum
            ## 2) same index
            self.pc = pc_
            if is_update_valid: ## check if the update is valid
                with torch.no_grad():
                    yc_u = self.netp.f(self.pc.data, self.pc.z_in(z)).squeeze(1)
                self.logger.info(f"(-1) SUCCESS: updated {len(self.pc)} critical points")
                return True, (self.pc, PointWrapper(yc_u, self.pc.get_map()), self.edges_z_list, self.indexes)
            else: ## NOTE: there may be something more clever that we can do, like filtering edges etc
                self.logger.info(f"(-1) FAIL: lost some critical points")
        

        self.logger.info(f'=== Recomputing the graph ===')
        self.edges_z_list, self.indexes = None, None ## Clear the edges and indexes to make sure previous ones don't leak through

        ## (0) Get critical point candidates
        with torch.no_grad():
            pc_start = self._get_grid_starting_pts(self.x_grid, self.x_grid_dist)
            if self.pc is not None: ## Add previous points to initial seeds
                pc_start = PointWrapper.merge(pc_start, self.pc) ## TODO: when optimizing, we should treat previous CPs with smaller learning rate. make two parameter groups
                self.pc = None ## Clear the CPs to make sure previous ones don't leak through
        self.mpm.plot(self.plotter.plot_function_and_cps, 'plot_start_points_to_find_cps',
                                        [pc_start.cpu().numpy(), "(0) Starting points to find CPs"])
        self.logger.info(f'(0) SUCCESS')
        
        ## (1) Get critical point candidates
        with self.record_time('cp_helper._gd_to_min_grad_norm'):
            pc_candidates, yc_x_norms, x_path_over_iters = self._gd_to_min_grad_norm(z, pc_start, epoch, self.config['lr_find_cps'], early_stop=True, plot_descent=self.plotter.do_plot('plot_cp_descent_recomputation'))
        if self.plotter.do_plot('plot_cp_descent_recomputation'):
            self.mpm.plot(self.plotter.plot_descent_trajectories, 'plot_cp_descent_recomputation',
                                      [pc_start.detach().cpu().numpy(), x_path_over_iters.cpu().numpy()],
                                      {'fig_label':'Trajectories of recomputing CPs'})
                
        if len(pc_candidates) == 0:
            self.logger.info(f'(1) FAIL: did not find any cp candidates')
            return False, None
        self.mpm.plot(self.plotter.plot_function_and_cps, 'plot_cp_candidates',
                                      [pc_candidates.detach().cpu().numpy(), "(1) Critical point candidates"],
                                        {'xc_attributes':yc_x_norms.cpu().numpy()})
        
        self.logger.info(f'(1) SUCCESS')

        ## (2) Cluster critical point candidates
        with self.record_time('cp_helper._cluster_cp_candidates'):
            success, pc = self._cluster_cp_candidates(z,pc_candidates, yc_x_norms, **kwargs)
        if not success:
            self.logger.info(f'(2) FAIL: no good clusters')
            return False, None
        self.pc = pc ## update CPs even if building edges fails since we can reuse them for tracking
        self.logger.info('(2) SUCCESS')
        self.mpm.plot(self.plotter.plot_function_and_cps, 'plot_cps',
                                      [pc.cpu().numpy(), "(2) Critical points after clustering"],
                                      {'p_backdrop':pc_candidates.cpu()})
        
        ## (3) Characterize critical points
        with self.record_time('cp_helper._characterize_critical_points'):
            eigvals, eigvecs, indexes, saddle_mask = self._characterize_critical_points(z, pc)
        self.mpm.plot(self.plotter.plot_characterized_cps, 'plot_characterized_cps',
                                      [pc.cpu(), eigvecs.cpu(), eigvals.cpu()])
        if (indexes == 1).sum() == 0:
            self.logger.info(f'(3) FAIL: no saddle points to connect from')
            return False, None
        self.logger.info('(3) SUCCESS')

        ## (4) Perturb saddle points
        with self.record_time('cp_helper._perturb_saddle_points'):
            p_path0_list, multipliers_to_descend = self._perturb_saddle_points(eigvals, eigvecs, indexes, pc)
        self.logger.info('(4) SUCCESS')
        
        ## (5) Flow from saddle points
        with self.record_time('cp_helper._let_perturbed_points_flow_to_minima'):
            p_path_list, x_path_over_iters = self._let_perturbed_points_flow_to_minima(z, p_path0_list, multipliers_to_descend, epoch)
        if self.plotter.do_plot('plot_saddle_descent'):
            self.mpm.plot(self.plotter.plot_saddle_trajectories, 'plot_saddle_descent', 
                                      [pc.cpu().numpy(), [p.cpu().numpy() for p in p_path_list], x_path_over_iters.cpu().numpy(), 
                                       indexes.cpu().numpy(), multipliers_to_descend.cpu().numpy()])
        self.logger.info('(5) SUCCESS')

        ## (6) Get edges from trajectories
        with self.record_time('cp_helper._get_edges_from_saddle_point_converges'):
            edge_z_list = self._get_edges_from_saddle_point_converges(pc, p_path_list, saddle_mask)
        self.logger.info('(6) SUCCESS')

        with torch.no_grad():
            yc_u = self.netp.f(pc.data, pc.z_in(z)).squeeze(1)
        ## Save to object
        self.pc, self.edges_z_list, self.indexes = pc, edge_z_list, indexes
        
        pyc_u = PointWrapper(yc_u, pc.get_map())
        return True, (pc, pyc_u, edge_z_list, indexes)

    def _gd_to_min_grad_norm(self, z, pc: PointWrapper, lr, epoch=None, early_stop=False, plot_descent=False, **kwargs):
        """
        Find the critical points of the NN by looking for ||grad f|| = 0.
        This is achieved using gradient descent where the points are optimized and the model is frozen.
        xc: [(bz bx) nx]
        """
        # default values
        max_iter = self.config['max_iter_find_cps']
        x_path_over_iters = None
            
        if plot_descent:
            ## for plotting only
            traj_every_nth_iter = self.config['plot_trajectories_every_n_iter']
            x_path_over_iters = torch.zeros([self.config['max_iter_find_cps'] // traj_every_nth_iter + 1, len(pc), self.config['nx']])
            x_path_over_iters.label = f'x_path in CP descent (epoch {epoch})'

        # work directly with the tensor
        pc.data.requires_grad = True  # to optimize xc, not the model self.params

        opt = torch.optim.Adam([pc.data], lr=lr)

        pc_in = None
        for i in (pbar := trange(max_iter)):
            opt.zero_grad()
            
            if plot_descent and (i % traj_every_nth_iter == 0):
                x_path_over_iters[i // traj_every_nth_iter] = pc.data.detach()
                        
            ## Mask the points that have left the domain
            # out_mask = (pc.data < self.bounds[:, 0]).any(1) | (pc.data > self.bounds[:, 1]).any(1)
            out_mask = get_is_out_mask(pc.data, self.bounds)
            stop_mask = out_mask

            pc_in = pc.select_w_mask(incl_mask=~stop_mask)
                
            if len(pc_in) == 0:
                self.logger.debug(f'early stop at it {i}: all points left the domain')
                break
            ## Compute the gradient norms
            y_x = self.netp.vf_x(pc_in.data, pc_in.z_in(z))  ## [k, ny, (nx+nz)]
            squared_norms = y_x.square().sum(2).sum(1)
            
            ## Check if we can stop early
            norms_small_enough_mask = squared_norms < self.config['dbscan.y_x_mag'] ** 2
            if early_stop and norms_small_enough_mask.all():
                self.logger.debug(f'early stop at it {i}: norms_small_enough_mask all true')
                break

            ## Update
            loss = squared_norms.mean()  ## compute mean of squared norms
            loss.backward()
            opt.step()
                
            pbar.set_description(f"Descending to CPs: {loss.item():.2e}")
        pbar.close() ## close tqdm which interferes with next print line due to async

        if plot_descent:
            x_path_over_iters = x_path_over_iters[:i // traj_every_nth_iter + 1] ## remove the unfilled part due to early stopping
            x_path_over_iters[-1] = pc.data.detach()
    
        with torch.no_grad():
            # TODO: something seems to be off here: we recompute xc but feed xc_in to compute y_x -
            #  probably we can save recomputing squared_norms since we anyways take the value of the last it
            self.logger.debug(f'found {len(pc_in)} candidates in {i} iterations ')
            # xc = xc[~out_mask].detach()
            # update pc
            pc = pc.select_w_mask(incl_mask=~stop_mask)
            pc = pc.detach()

            if len(pc) > 0:
                y_x = self.netp.vf_x(pc_in.data, pc_in.z_in(z))  ## [bx, ny, nx]
                squared_norms = y_x.square().sum(2).sum(1)

            return pc, squared_norms.detach().sqrt(), x_path_over_iters

    def _cluster_cp_candidates(self, z, pc: PointWrapper, y_x_mag=None, **kwargs):
        """Cluster critical point candidates. Currently uses DBSCAN on CPU."""

        ## (1) Remove all points outside the domain
        out_mask = get_is_out_mask(pc.data, self.bounds)
        # pc_ = xc[~out_mask]
        pc_ = pc.select_w_mask(incl_mask=~out_mask)
        self.logger.debug(f"nof points inside bounds: {len(pc_)}")
        if len(pc_) == 0:
            return False, None
        
        # ## (2) Remove all points that have not converged to critical points within some threshold
        if y_x_mag is not None:
            y_x_mag = y_x_mag[~out_mask]
        else:
            with torch.no_grad():
                y_x = self.netp.vf_x(pc_.data, pc_.z_in(z))
            y_x_mag = y_x.square().sum(2).sum(1).sqrt()

        # move to CPU
        y_x_mag = y_x_mag.detach().cpu()
        pc_ = pc_.cpu()
        # plot histogram of gradient magnitudes
        self.mpm.plot(self.plotter.plot_hist, 'plot_grad_mag_hist', 
                                    [y_x_mag, "Histogram of CPs Gradient Magnitudes"], dont_parallelize=True)

        mask = y_x_mag < self.config['dbscan.y_x_mag']
        # pc_ = pc_[mask]
        y_x_mag_ = y_x_mag[mask]
        pc_ = pc_.select_w_mask(incl_mask=mask)
        self.logger.debug(f"nof points with small enough gradient: {len(pc_)}")
        if len(pc_) == 0:
            return False, None

        pts_per_shape_list = []
        for i_shape in range(self.config['batch_size']):
            ## (3) Cluster the remaining points and take representative
            xc = pc_.pts_of_shape(i_shape)
            y_x_mag_shape = y_x_mag_[pc_.get_idcs(i_shape)]
            if len(xc) == 0:
                pts_per_shape_list.append(torch.zeros(0))
                continue
                
            dbscan = DBSCAN(eps=self.config['dbscan.eps'], min_samples=1)
            labels = dbscan.fit_predict(xc)
            ## Old: take any instance from the cluster
            # _, idx = np.unique(labels, return_index=True)
            # xc_u = pc_[idx]
            ## New: take the instance with the smallest gradient
            N_clusters = labels.max().item() + 1
            self.logger.debug(f"nof clusters: {N_clusters}")
            smallest_grad_per_cluster = torch.full([N_clusters, ], torch.inf)
            xc_u = torch.zeros([N_clusters, self.config['nx']])
            for i, label in enumerate(labels):
                if y_x_mag_shape[i] < smallest_grad_per_cluster[label]:
                    smallest_grad_per_cluster[label] = y_x_mag_shape[i]
                    xc_u[label] = xc[i]
            pts_per_shape_list.append(xc_u)
        pc_u = PointWrapper.create_from_pts_per_shape_list(pts_per_shape_list)
        pc_u.data.to(pc.data.device)  # TODO: is this really necessary? Should be on default device already, right?
        return True, pc_u

    def _let_perturbed_points_flow_to_minima(self, z, p_path0_list:list[PointWrapper], multipliers_to_descend:torch.Tensor, epoch):
        """A simple optimization loop to let critical points flow along gradient to minima/maxima"""

        # TODO: was cloning here necessary?
        # x_path = p_path0.clone().detach()
        # p_path = PointWrapper(p_path0.data.clone().detach(), map=p_path0.get_map())
        # nof_half_paths = len(p_path) // 2  # index for selecting min/max paths; number of paths to min/max if half of all paths

        x_path = torch.cat([p.data for p in p_path0_list], dim=0)
        z_path = torch.cat([p.z_in(z) for p in p_path0_list], dim=0)
        # nof_half_paths = len(x_path) // 2  # index for selecting min/max paths; number of paths to min/max if half of all paths
        
        x_path.requires_grad = True
        opt = torch.optim.Adam([x_path], lr=self.config['lr_saddle_descent'])

        plot_saddle_descent = self.plotter.do_plot('plot_saddle_descent')
        x_path_over_iters = None
        if plot_saddle_descent:
            traj_every_nth_iter = self.config['plot_trajectories_every_n_iter']
            x_path_over_iters = torch.zeros(
                [self.config['n_iter_saddle_descent'] // traj_every_nth_iter + 1, len(x_path), self.config['nx']])  ## for plotting only
            # y_path_over_iters = torch.zeros([N_iter + 1, len(x_path), ])  ## for plotting only
            x_path_over_iters.label = f'x_path in saddle descent (epoch {epoch})'

        for i in (pbar := trange(self.config['n_iter_saddle_descent'], leave=True, ncols=80, position=0)):
            opt.zero_grad()
            y = self.netp.f(x_path, z_path).squeeze(1)  ## [bx, ny]

            ## for plotting only
            if plot_saddle_descent and (i % traj_every_nth_iter == 0):
                x_path_over_iters[i // traj_every_nth_iter] = x_path.detach()

            loss = (multipliers_to_descend * y).mean()
            loss.backward()
            opt.step()
            pbar.set_description(f"Connecting CPs: {loss.item():.2e}")

            if i % self.config['check_interval_grad_mag_saddle_descent'] == 0:
                # stop if all grad_mags < thresh
                grad_mags = x_path.grad.square().sum(1).sqrt()
                if (grad_mags > self.config['stop_grad_mag_saddle_descent']).sum() == 0:
                    self.logger.debug(f'Early stop "Connecting CPs" at it {i}')
                    break

        ## for plotting only
        if plot_saddle_descent:
            x_path_over_iters = x_path_over_iters[:i // traj_every_nth_iter + 1]
            x_path_over_iters[-1] = x_path.detach()

        p_map = p_path0_list[0].get_map()
        n_pts = len(p_path0_list[0])
        x_path = x_path.detach()
        p_path_list = [PointWrapper(x_path[i*n_pts:(i+1)*n_pts], p_map) for i in range(2 * self.config['nx'])]

        return p_path_list, x_path_over_iters

    def _characterize_critical_points(self, z, pc):
        # characterize the found critical points
        ## Compute the Hessians at all the critical points
        with torch.no_grad():
            H = self.netp.vf_xx(pc.data, pc.z_in(z)).squeeze(1)
        ## Compute the eigendecomposition of each Hessian
        out = torch.linalg.eigh(H)
        eigvals = out.eigenvalues.detach()
        eigvecs = out.eigenvectors.detach()
        # eigvals = torch.linalg.eigvalsh(H) ## NOTE: this is the numerically stable way of computing eigvals if they are needed for AD
        ## The index of the critical point is the number of negative eigenvalues
        indexes = (eigvals < 0).sum(1)
        saddle_mask = (indexes != 0) & (indexes != self.config['nx'])  ## 0 < index < nx; saddle point
        
        self.logger.debug(f"Index of each critical point: {indexes.tolist()}")
        return eigvals, eigvecs, indexes, saddle_mask

    def _perturb_saddle_points(self, eigvals, eigvecs, indexes, pc):
        """
        TODO: we should specify the stepsize from the function, eg its frequency or Lipshitzness
        """

        ## Perturb each saddle point towards + and - of the eigenvector corresponding to the negative eigenvalue
        saddle_mask = (indexes != 0) & (indexes != self.config['nx'])  ## 0 < index < nx; saddle point
        p_saddle = pc.select_w_mask(saddle_mask)

        eigvals = eigvals[saddle_mask] ## [k, nx]
        eigvecs = eigvecs[saddle_mask] ## [k, nx, ith-vector]  
        eigvecs = eigvecs.transpose(1, 2) ## [k, ith-vector, nx]  # TODO: check if this is correct
        multiplier_to_descend = torch.where(eigvals < 0, 1, -1) ## [k, nx]
        
        multiplier_to_descend_list = []
        p_path0_list = []
        for i in range(eigvals.shape[1]):
            dx = eigvecs[:, i, :] ## [k, nx]
            x_i_pos = p_saddle.data + self.config['perturbation_saddle_descent'] * dx
            x_i_neg = p_saddle.data - self.config['perturbation_saddle_descent'] * dx
            p_path0_list.append(PointWrapper(x_i_pos, map=p_saddle.get_map()))
            p_path0_list.append(PointWrapper(x_i_neg, map=p_saddle.get_map()))
            multiplier_to_descend_list.append(multiplier_to_descend[:, i])
            multiplier_to_descend_list.append(multiplier_to_descend[:, i])
            
        multipliers_to_descend = torch.cat(multiplier_to_descend_list)
        return p_path0_list, multipliers_to_descend

    def _get_edges_from_saddle_point_converges(self, pc, p_path_list:list[PointWrapper], saddle_mask_all_z):

        edges_z_list = []

        for i_shape in range(self.config['batch_size']):
            if len(pc.pts_of_shape(i_shape)) == 0:
                edges_z_list.append(torch.zeros(0))
                continue
            x_path = torch.vstack([p.pts_of_shape(i_shape) for p in p_path_list]) ## [nx*bx, nx]
            xc = pc.pts_of_shape(i_shape)
            saddle_mask = saddle_mask_all_z[pc.get_idcs(i_shape)]

            ## We need to find to which extreme point we converged
            d = (x_path[:, None, :] - xc[None, :, :]).square().sum(2).sqrt()  ## NOTE: we only have to measure to extreme points, so mask saddle points.
            ## NOTE: mby keep track of different xc lists depending on their index
            attractors = d.min(1)
            idxs_of_crits = attractors.indices
            dists_to_crits = attractors.values
            idxs_of_saddles_in_u = torch.where(saddle_mask)[0]
            edges = torch.stack([idxs_of_saddles_in_u.repeat(2 * self.config['nx']), idxs_of_crits]).T
            # self.logger.debug(edges)
            ## For each edge, check if the path actually converged. TODO: we don't know what the exact best approach is
            ## (1) Remove all points outside the domain
            ## (2) Remove all points that have not converged to critical points within some threshold
            ## (3) Remove all points which are not close enough to known critical point
            dist_mask = dists_to_crits < self.config['dbscan.eps']
            edges = edges[dist_mask]
            ## TODO: Eg if we have converged to an extreme point (||grad||<eps), but there is no known point there, we have missed it
            edges_z_list.append(edges)

        return edges_z_list
            
    def _get_grid_starting_pts(self, x_grid, grid_dist):
        '''
        Create grid once at beginning.
        Translate the grid by a random offset.
        '''
        xc_offset = torch.rand((self.config['batch_size'], self.config['nx'])) * grid_dist  # bz nx
        # x_grid: [n_points nx]
        x = x_grid.unsqueeze(0) + xc_offset.unsqueeze(1)  # bz n_points nx
        return PointWrapper.create_from_equal_bx(x)