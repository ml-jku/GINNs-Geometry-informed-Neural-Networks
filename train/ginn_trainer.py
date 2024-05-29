import time
import einops
import torch
from tqdm import trange
import logging

import wandb
from GINN.morse.scc_surfacenet_manager import SCCSurfaceNetManager
from GINN.shape_boundary_helper import ShapeBoundaryHelper
from train.losses import closest_shape_diversity_loss, interface_loss, eikonal_loss, envelope_loss, normal_loss_euclidean, obstacle_interior_loss, strain_curvature_loss
from train.train_utils.autoclip import AutoClip
from models.model_utils import tensor_product_xz

from GINN.visualize.plotter_2d import Plotter2d
from GINN.visualize.plotter_3d import Plotter3d
from GINN.problem_sampler import ProblemSampler
from GINN.helpers.timer_helper import TimerHelper
from train.train_utils.latent_sampler import sample_new_z
from utils import get_stateless_net_with_partials, set_and_true

class Trainer():
    
    def __init__(self, config, model, mp_manager) -> None:
        self.config = config
        self.model = model
        self.mpm = mp_manager
        self.logger = logging.getLogger('trainer')
        self.device = config['device']
        self.netp = get_stateless_net_with_partials(self.model, use_x_and_z_arg=self.config['use_x_and_z_arg'])
        
        self.p_sampler = ProblemSampler(self.config)

        self.timer_helper = TimerHelper(self.config, lock=mp_manager.get_lock())
        self.plotter = Plotter2d(self.config) if config['nx']==2 else Plotter3d(self.config)
        self.mpm.set_timer_helper(self.timer_helper)  ## weak circular reference
        self.scc_manager = SCCSurfaceNetManager(self.config, self.netp, self.mpm, self.plotter, self.timer_helper, self.p_sampler, self.device)
        self.shape_boundary_helper = ShapeBoundaryHelper(self.config, self.netp,self.mpm, self.plotter, self.timer_helper, 
                                                         self.p_sampler.sample_from_interface()[0], self.device)
        self.auto_clip = AutoClip(config)

        self.p_surface = None

    def train(self):

        opt = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        if set_and_true('use_scheduler', self.config):
            def warm_and_decay_lr_scheduler(step: int):
                return self.config['scheduler_gamma'] ** (step / self.config['decay_steps'])
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=warm_and_decay_lr_scheduler)
        z = sample_new_z(self.config, is_init=True)
        self.logger.info(f'Initial z: {z}')
        
        cur_plot_epoch = 0
        log_history_dict = {}
        for epoch in (pbar := trange(self.config['max_epochs'], leave=True, position=0, colour="yellow")):
            self.mpm.update_epoch(epoch)

            opt.zero_grad()
            if 'reset_zlatents_every_n_epochs' in self.config and epoch % self.config['reset_zlatents_every_n_epochs'] == 0:
                z = sample_new_z(self.config)
            
            if self.plotter.do_plot():
                self.plotter.reset_output(self.p_sampler.recalc_output(self.netp.f_, self.netp.params_, z), epoch=epoch)
                self.mpm.plot(self.plotter.plot_shape, 'plot_shape', arg_list=[self.p_sampler.constr_pts_dict], kwargs_dict={})
            
            loss_scc = torch.tensor(0.0)
            if self.config['lambda_scc'] > 0:
                with self.timer_helper.record('train.get_scc_pts_to_penalize'):
                    success, res_tup = self.scc_manager.get_scc_pts_to_penalize(z, epoch)
                if success:
                    p_penalize, p_penalties = res_tup
                    self.logger.debug(f'penalize DCs with {len(p_penalize)} points')
                    y_saddles_opt = self.model(p_penalize.data, p_penalize.z_in(z)).squeeze(1)
                    loss_scc = self.config['lambda_scc'] *  (y_saddles_opt * p_penalties.data).mean()
                            
            loss_env = torch.tensor(0.0)
            if self.config['lambda_env'] > 0:
                ys_env = self.model(*tensor_product_xz(self.p_sampler.sample_from_envelope(), z)).squeeze(1)
                loss_env = self.config['lambda_env'] * envelope_loss(ys_env)

            loss_if = torch.tensor(0.0)
            if self.config['lambda_bc'] > 0:
                ys_BC = self.model(*tensor_product_xz(self.p_sampler.sample_from_interface()[0], z)).squeeze(1)
                loss_if = self.config['lambda_bc'] * interface_loss(ys_BC)
                
            loss_if_normal = torch.tensor(0.0)
            if self.config['lambda_normal'] > 0:
                pts_normal, target_normal = self.p_sampler.sample_from_interface()
                ys_normal = self.netp.vf_x(*tensor_product_xz(pts_normal, z)).squeeze(1)
                loss_if_normal = self.config['lambda_normal'] * normal_loss_euclidean(ys_normal, torch.cat([target_normal for _ in range(self.config['batch_size'])]))

            loss_obst = torch.tensor(0.0)
            if self.config['lambda_obst'] > 0:
                ys_obst = self.model(*tensor_product_xz(self.p_sampler.sample_from_obstacles(), z))
                loss_obst = self.config['lambda_obst'] * obstacle_interior_loss(ys_obst)

            if self.config['lambda_eikonal'] > 0 or self.config['lambda_div'] > 0:
                xs_domain = self.p_sampler.sample_from_domain()
            
            loss_eikonal = torch.tensor(0.0)
            if self.config['lambda_eikonal'] > 0:
                ## Eikonal loss: NN should have gradient norm 1 everywhere
                y_x_eikonal = self.netp.vf_x(*tensor_product_xz(xs_domain, z))
                loss_eikonal = self.config['lambda_eikonal'] * eikonal_loss(y_x_eikonal)

            if self.config['lambda_div'] > 0 or self.config['lambda_curv'] > 0:
                if self.p_surface is None or epoch % self.config['recompute_surface_pts_every_n_epochs'] == 0:
                    success, weights_surf_pts = self.shape_boundary_helper.get_surface_pts(z)
                
            loss_div = torch.tensor(0.0)
            if self.config['lambda_div'] > 0 and self.config['batch_size'] > 1:
                if self.p_surface is None:
                    self.logger.debug('No surface points found - skipping diversity loss')
                else:
                    y_div = self.model(*tensor_product_xz(self.p_surface.data, z)).squeeze(1)  # [(bz k)] whereas k is n_surface_points; evaluate model at all surface points for each shape
                    loss_div = self.config['lambda_div'] * closest_shape_diversity_loss(einops.rearrange(y_div, '(bz k)-> bz k', bz=self.config['batch_size']), 
                                                                                        weights=weights_surf_pts)
                    if torch.isnan(loss_div) or torch.isinf(loss_div):
                        self.logger.warning(f'NaN or Inf loss_div: {loss_div}')
                        loss_div = torch.tensor(0.0) if torch.isnan(loss_div) or torch.isinf(loss_div) else loss_div 
                    
            loss_curv = torch.tensor(0.0)
            if self.config['lambda_curv'] > 0:
                if self.p_surface is None:
                    self.logger.debug('No surface points found - skipping curvature loss')
                else:
                    y_x_surf = self.netp.vf_x(self.p_surface.data, self.p_surface.z_in(z)).squeeze(1)
                    y_xx_surf = self.netp.vf_xx(self.p_surface.data, self.p_surface.z_in(z)).squeeze(1)
                    loss_curv = self.config['lambda_curv'] * strain_curvature_loss(y_x_surf, y_xx_surf, clip_max_value=self.config['strain_curvature_clip_max'],
                                                                                   weights=weights_surf_pts)

            loss = loss_env + loss_if + loss_if_normal + loss_obst + loss_eikonal + loss_scc + loss_div + loss_curv
            
            ## gradients
            loss.backward()
            grad_norm = self.auto_clip.grad_norm(self.model.parameters())
            if self.auto_clip.grad_clip_enabled:
                self.auto_clip.update_gradient_norm_history(grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.auto_clip.get_clip_value())
                
            # take step
            opt.step()
            if set_and_true('use_scheduler', self.config):
                sched.step()
            
            ## Async Logging
            cur_log_dict = {
                'loss': loss,
                'loss_env': loss_env,
                'loss_obst': loss_obst,
                'loss_bc': loss_if,
                'loss_scc': loss_scc,
                'loss_eikonal': loss_eikonal,
                'loss_bc_normal': loss_if_normal,
                'loss_div': loss_div,
                'loss_curv': loss_curv,
                'neg_loss_div': (-1) * loss_div,
                'grad_norm_pre_clip': grad_norm,
                'grad_norm_post_clip': self.auto_clip.get_last_gradient_norm(),
                'grad_clip': self.auto_clip.get_clip_value(),
                'lr': opt.param_groups[0]['lr'],
                'epoch': epoch,
                }
            log_history_dict[epoch] = cur_log_dict
            cur_plot_epoch = self.log_to_wandb(cur_plot_epoch, log_history_dict, epoch)

            if self.config['save_every_n_epochs'] > 0 and \
                    (epoch % self.config['save_every_n_epochs'] == 0 or epoch == self.config['max_epochs']):
                if self.config['model_path'] is not None:
                    torch.save(self.model.state_dict(), self.config['model_path'])
            pbar.set_description(f"{epoch}: env:{loss_env.item():.1e} BC:{loss_if.item():.1e} obst:{loss_obst.item():.1e} eik:{loss_eikonal.item():.1e} cc:{loss_scc.item():.1e} div:{loss_div.item():.1e} curv:{loss_curv.item():.1e}")
            # print(torch.cuda.memory_summary())
            
        ## Finished
        self.timer_helper.print_logbook()

    def log_to_wandb(self, cur_plot_epoch, log_history_dict, epoch, await_all=False):
        with self.timer_helper.record('plot_helper.pool await async results'):
            ## Wait for plots to be ready, then log them
            while cur_plot_epoch <= epoch:
                if not self.mpm.are_plots_available_for_epoch(cur_plot_epoch):
                    ## no plots for this epoch; just log the current losses
                    wandb.log(log_history_dict[cur_plot_epoch])
                    del log_history_dict[cur_plot_epoch]
                    cur_plot_epoch += 1
                elif self.mpm.plots_ready_for_epoch(cur_plot_epoch):
                    ## plots are available and ready
                    wandb.log(log_history_dict[cur_plot_epoch] | self.mpm.pop_plots_dict(cur_plot_epoch))
                    del log_history_dict[cur_plot_epoch]
                    cur_plot_epoch += 1
                elif await_all or (epoch == self.config['max_epochs'] - 1):
                    ## plots are not ready yet - wait for them
                    self.logger.debug(f'Waiting for plots for epoch {cur_plot_epoch}')
                    time.sleep(1)
                else:
                    # print(f'Waiting for plots for epoch {cur_plot_epoch}')
                    break
        return cur_plot_epoch

    def test_plotting(self):
        """Minimal function to test plotting in 3D."""
        epoch = 0

        z = torch.eye(self.config['nz'])[:self.config['batch_size']]
        self.scc_manager.set_z(z)
        self.mpm.update_epoch(epoch)

        self.plotter.reset_output(self.p_sampler.recalc_output(self.f, self.params, z), epoch=epoch)      
        self.mpm.plot(self.plotter.plot_shape, 'plot_helper.plot_shape', arg_list=[None], kwargs_dict={})
        # self.mp_manager.plot(self.plot_helper.plot_shape, 'plot_helper.plot_shape', 
        #                     arg_list=[self.p_sampler.constr_pts_list], kwargs_dict={})
        


        ## Boiler for async wandb logging
        cur_plot_epoch = 0
        log_history_dict = {}
        ## Async Logging
        cur_log_dict = {
            'epoch': epoch,
            }
        log_history_dict[epoch] = cur_log_dict
        cur_plot_epoch = self.log_to_wandb(cur_plot_epoch, log_history_dict, epoch, await_all=True)
        
        