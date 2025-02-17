import math
import time
import torch
from tqdm import trange
import logging
from functools import partial

import wandb
from GINN.data.simjeb_dataloader import SimJebDataloader
from GINN.numerical_boundary_helper import NumericalBoundaryHelper
from GINN.shape_boundary_helper import ShapeBoundaryHelper
from GINN.data.simjeb_dataset import SimJebDataset
from GINN.evaluation.jeb_meter import JebMeter
from GINN.evaluation.simple_obst_meter import SimpleObstacleMeter
from models.net_w_partials import NetWithPartials
from train.opt.opt_util import get_opt, opt_step
from train.train_utils.loss_calculator_alm import AdaptiveAugmentedLagrangianLoss
from train.train_utils.loss_calculator_manual import ManualWeightedLoss
from train.train_utils.loss_keys import LossKey, get_loss_keys_and_lambdas
from train.train_utils.autoclip import AutoClip

from GINN.plot.plotter_2d import Plotter2d
from GINN.plot.plotter_3d import Plotter3d
from GINN.ph.ph_plotter import PHPlotter
from GINN.ph.ph_manager import PHManager
from GINN.speed.timer import Timer
from train.train_utils.latent_sampler import LatentSampler
from util.checkpointing import load_model_optim_sched, save_model_every_n_epochs
from util.misc import combine_dicts, do_plot, get_model, get_problem, is_every_n_epochs_fulfilled
from train.losses_ginn import *
from train.train_utils.loss_calculator_alm import AdaptiveAugmentedLagrangianLoss


class Trainer():
    
    def __init__(self, config, mp_manager, mp_pool_scc, mp_top) -> None:
        self.config = config
        self.mpm = mp_manager
        self.logger = logging.getLogger('trainer')
        self.model = get_model(**config['model'])
        self.netp = NetWithPartials.create_from_model(self.model, **config['model'])
        self.problem = get_problem(problem_config=self.config['problem'], **self.config['problem_sampling'])

        self.timer = Timer(**self.config['timer'], lock=mp_manager.get_lock())
        self.mpm.set_timer(self.timer)  ## weak circular reference
        if self.config['nx'] == 2:
            self.plotter = Plotter2d(envelope=self.problem.envelope, bounds=self.problem.bounds, **self.config['plot'])
        elif self.config['nx'] == 3:
            self.plotter = Plotter3d(bounds=self.problem.bounds, **self.config['plot'])
        
        self.ph_manager = None
        if self.config['lambda_scc'] > 0:
            self.ph_manager = PHManager(bounds=self.problem.bounds, netp=self.netp, func_inside_envelope=self.problem.is_inside_envelope,
                                    mp_pool=mp_pool_scc, **self.config['ph'])
        self.ph_plotter = PHPlotter(iso_level=self.config['ph']['iso_level'], fig_size=self.config['plot']['fig_size'])
        if self.config.get('do_numerical_surface_points', True):
            self.shape_boundary_helper = NumericalBoundaryHelper(netp=self.netp, mp_manager=self.mpm, plotter=self.plotter, 
                                                         x_interface=self.problem.sample_from_interface()[0], bounds=self.problem.bounds, **self.config['surface'])
        else:
            self.shape_boundary_helper = ShapeBoundaryHelper(netp=self.netp, mp_manager=self.mpm, plotter=self.plotter, 
                                                         x_interface=self.problem.sample_from_interface()[0], bounds=self.problem.bounds,
                                                         **self.config['surface'])
        self.auto_clip = AutoClip(**config['grad_clipping'])
        self.z_sampler = LatentSampler(**config['latent_sampling'])
        
        # FEM interface
        self.feni = None
        if self.config['lambda_comp'] > 0 or self.config['lambda_vol'] > 0:
            from GINN.fenitop.fenitop_mp import FenitopMP
            from GINN.fenitop.heaviside_filter import heaviside
            feni_kwargs = self.config['FEM'].copy()
            
            if self.config['problem']['problem_str'] == 'simjeb':
                feni_kwargs['interface_pts'] = self.problem.constr_pts_dict['interface']
                feni_kwargs['envelope_mesh'] = self.problem.mesh_env
                feni_kwargs['envelope_unnormalized'] = self.problem.mesh_env.bounds.T
                feni_kwargs['bounds_unnormalized'] = self.problem.bounds
            else:
                feni_kwargs['envelope_unnormalized'] = self.problem.envelope_unnormalized.cpu().numpy()
                feni_kwargs['bounds_unnormalized'] = self.problem.bounds_unnormalized
            self.feni = FenitopMP(device=torch.get_default_device(), bz=self.config['ginn_bsize'], mp_pool=mp_top, np_func_inside_envelope=self.problem.is_inside_envelope, **feni_kwargs)
            self.beta = self.config['beta_sched']['init']
            
        # data
        if ('lambda_data' in config) and (config['lambda_data'] > 0):
            dataset = SimJebDataset(**config['data'], seed=config['seed'])
            self.dataloader = SimJebDataloader(dataset, config['n_points_data'])
        
        # metrics
        self.meter = None
        if config['problem'] in ['obstacle', 'double_obstacle']:
            self.meter = SimpleObstacleMeter.create_from_problem_sampler(self.problem, **config['metrics'])
        elif config['problem'] == 'simjeb':
            self.meter = JebMeter(config)
            # self.config['n_points_envelope'] = self.config['n_points_envelope'] // 3 * 3 # what's this for? This is redundant. Where parentheses misplaced?

        self.p_surface = None
        self.weights_surf_pts = None
        self.cur_plot_epoch = 0
        self.log_history_dict = {}
        
        # loss balancing
        self.scalar_loss_keys, self.field_loss_keys, lambda_dict, objective_key = get_loss_keys_and_lambdas(self.config)
        self.all_loss_keys = self.scalar_loss_keys + self.field_loss_keys
        self.loss_dispatcher = self.init_loss_dispatcher()
        if self.config['use_augmented_lagrangian']:
            self.loss_calculator = AdaptiveAugmentedLagrangianLoss(scalar_loss_keys=self.scalar_loss_keys, obj_key=objective_key, 
                                                                    lambda_dict=lambda_dict, field_loss_keys=self.field_loss_keys, **self.config['adaptive_penalty'])
        else:
            self.loss_calculator = ManualWeightedLoss(scalar_loss_keys=self.scalar_loss_keys, lambda_dict=lambda_dict, field_loss_keys=self.field_loss_keys)
        
        self.p = None
        
    def train(self):
        
        # get optimizer and scheduler
        opt = get_opt(self.config['opt'], self.config, self.model.parameters())
        sched = None
        if self.config.get('use_scheduler', False):
            def warm_and_decay_lr_scheduler(step: int):
                return self.config['scheduler_gamma'] ** (step / self.config['decay_steps'])
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=warm_and_decay_lr_scheduler)
        # maybe load model, optimizer and scheduler
        self.model, opt, sched = load_model_optim_sched(self.config, self.model, opt, sched)
        
        # get z
        z = self.z_sampler.train_z()
        z_corners = self.z_sampler.get_z_corners(len(self.config['data'].get('simjeb_ids', [])))
        z_val = self.z_sampler.val_z()
        self.logger.info(f'Initial z (resampling: {self.config["latent_sampling"]["z_sample_method"]}): {z}')        
        self.logger.info(f'z_corners (not resampled): {z_corners}')
        self.logger.info(f'z_val (not resampled): {z_val}')
        n_train_shapes_to_plot = min(len(z) + len(z_corners), self.config['train_plot_max_n_shapes'])
        
        # training/validation loop
        for epoch in (pbar := trange(self.config['max_epochs'], leave=True, position=0, colour="yellow")):
            cur_log_dict = {}
            self.mpm.update_epoch(epoch)
            batch = self.dataloader.get_data_batch() if self.config['lambda_data'] > 0 else None
            opt.zero_grad()

            ## validation
            if epoch > 0 and is_every_n_epochs_fulfilled(epoch, self.config, 'valid_every_n_epochs'):
                self.model.eval()
                # plot validation shapes
                if self.config['nx'] == 2:
                    Y = self.problem.recalc_output(self.netp, z_val, **self.config['meshing'])
                    if self.config['lambda_comp'] > 0:
                        Y = heaviside(Y, beta=self.beta, nf_is_density=self.config['nf_is_density'])
                    self.plotter.reset_output(Y.cpu().numpy(), epoch=epoch)
                elif self.config['nx'] == 3:
                    self.plotter.reset_output(self.problem.recalc_output(self.netp, z_val, **self.config['meshing']), epoch=epoch)
                self.mpm.plot(self.plotter.plot_shape, 'val_plot_shape', kwargs_dict={'fig_label': 'Val Boundary', 'is_validation': True})
                
                # compute validation metrics
                if epoch > 0 and is_every_n_epochs_fulfilled(epoch, self.config, 'val_shape_metrics_every_n_epochs'):
                    mesh_or_contour = self.problem.get_mesh_or_contour(self.netp.f_, self.netp.params, z_val)
                    if mesh_or_contour is not None:
                        self.mpm.metrics(self.meter.get_average_metrics_as_dict, arg_list=[mesh_or_contour], kwargs_dict={'prefix': 'm_val_'})

            ## training
            self.model.train()
            if epoch > 0 and is_every_n_epochs_fulfilled(epoch, self.config, 'reset_zlatents_every_n_epochs'):
                z = self.z_sampler.train_z()
            # only plot the GINN points in the first epoch, as they are memory-intensive
            # if self.feni.constraint_to_pts_dict if epoch > 0 else combine_dicts([self.feni.constraint_to_pts_dict, self.problem.constr_pts_dict])
            vis_pts_dict = {} if epoch > 0 else self.problem.constr_pts_dict
            if self.feni is not None: 
                vis_pts_dict = combine_dicts([vis_pts_dict, self.feni.constraint_to_pts_dict])
                
            ## reset base shape if needed
            if do_plot(self.config, epoch) and is_every_n_epochs_fulfilled(epoch, self.config, 'plot_every_n_epochs'):
                z_plot = self.z_sampler.combine_z_with_z_corners(z, z_corners, n_train_shapes_to_plot)
                if self.config['plot_shape']:
                    if self.config['nx'] == 2:
                        Y = self.problem.recalc_output(self.netp, z_plot, **self.config['meshing'])
                        self.plotter.reset_output(Y.cpu().numpy(), epoch=epoch)
                        self.mpm.plot(self.plotter.plot_shape, 'plot_shape', arg_list=['Train Boundary', vis_pts_dict])
                    elif self.config['nx'] == 3:
                        self.plotter.reset_output(self.problem.recalc_output(self.netp, z_plot, **self.config['meshing']), epoch=epoch)
                        self.mpm.plot(self.plotter.plot_shape, 'plot_shape', arg_list=['Train Boundary', vis_pts_dict])
                    
                if self.config['lambda_comp'] > 0:
                    with torch.no_grad():
                        chosen_rho_field = self.netp(*tensor_product_xz(self.feni.fem_xy_torch_device, z_plot)).view(len(z_plot), -1)
                    # for some strange reason k3d.volume expects the axes as (z, y, x)
                    if self.config['nx'] == 2:
                        chosen_rho_field = einops.rearrange(chosen_rho_field[:, self.feni.sort_idcs], 'bz (y x) -> bz x y', x=self.feni.chosen_resolution[1])
                    elif self.config['nx'] == 3:
                        chosen_rho_field = einops.rearrange(chosen_rho_field[:, self.feni.sort_idcs], 'bz (x y z) -> bz z y x', x=self.feni.chosen_resolution[0], y=self.feni.chosen_resolution[1])
                    if not self.config['FEM']['use_filters']:
                        chosen_rho_field = heaviside(chosen_rho_field, beta=self.beta, nf_is_density=self.config['nf_is_density'])
                    self.mpm.plot(self.plotter.plot_grad_field, '', kwargs_dict={'grad_fields': chosen_rho_field.detach().cpu().numpy(), 'constraint_pts_dict': vis_pts_dict, 'fig_label': f'Rho field'})
                    
                if self.ph_manager is not None:
                    # subsample to n_train_shapes_to_plot
                    PH = self.ph_manager.calc_ph(z[:n_train_shapes_to_plot])
                    PH = [ph.get() for ph in PH]
                    self.mpm.plot(self.ph_plotter.plot_ph_diagram, 'plot_ph_diagram', arg_list=[PH], kwargs_dict={})

            ## compute metrics _before_ taking the GD step
            if epoch > 0 and is_every_n_epochs_fulfilled(epoch, self.config, 'shape_metrics_every_n_epochs'):
                mesh_or_contour = self.problem.get_mesh_or_contour(self.netp.f_, self.netp.params, z)
                if mesh_or_contour is not None:
                    self.mpm.metrics(self.meter.get_average_metrics_as_dict, arg_list=[mesh_or_contour], kwargs_dict={'prefix': 'm_train_'})
                
                if self.config['lambda_data'] > 0:
                    mesh_or_contour = self.problem.get_mesh_or_contour(self.netp.f_, self.netp.params, z_corners)
                    if mesh_or_contour is not None:
                        self.mpm.metrics(self.meter.get_average_metrics_as_dict, arg_list=[mesh_or_contour], kwargs_dict={'prefix': 'm_corners_'})
            
            ## gradients and optimizer step
            with Timer.record('compute losses'):
                loss_log_dict = opt_step(opt=opt, epoch=epoch, model=self.model, loss_and_backward_fn=self.losses_and_backward,
                                         z=z, z_corners=z_corners, batch=batch, auto_clip=self.auto_clip)
            cur_log_dict.update(loss_log_dict)
            if self.config.get('use_scheduler', False):
                sched.step()

            self.loss_calculator.adaptive_update()  # is a no-op for ManualWeightedLoss

            ## Async Logging
            cur_log_dict.update({
                'neg_loss_div': (-1) * loss_log_dict['loss_div'] if 'loss_div' in loss_log_dict else 0.0,
                'grad_norm_post_clip': self.auto_clip.get_last_gradient_norm(),
                'grad_clip': self.auto_clip.get_clip_value(),
                'lr': self.config['lr'] if sched is None else sched.get_last_lr(),
                'epoch': epoch,
                })
            self.log_history_dict[epoch] = cur_log_dict
            self.log_to_wandb(epoch)
            save_model_every_n_epochs(self.model, opt, sched, self.config, epoch)
            pbar.set_description(f"{epoch}:" + " ".join([f"{k.replace('loss_', '')}:{v.item():.1e}" for k, v in loss_log_dict.items() if "loss_" in k and "unweighted" not in k]))
        
        ## Final logging
        self.log_to_wandb(epoch, await_all=True)
        ## Finished
        self.timer.print_logbook()


    def losses_and_backward(self, z, epoch, batch=None, z_corners=None):

        loss_dict = {}
        field_dict = {}

        # compute surface points
        surf_for_curv = LossKey('curv') in self.all_loss_keys and epoch >= self.config.get('start_curv', 0)
        surf_for_div = LossKey('div') in self.all_loss_keys and epoch >= self.config.get('start_div', 0) and self.config['diversity_pts_source'] == 'surface'
        surf_for_chamfer_div = LossKey('chamfer_div') in self.all_loss_keys and epoch >= self.config.get('start_chamfer_div', 0)
        surf_needs_update = self.p_surface is None or is_every_n_epochs_fulfilled(epoch, self.config['surface'], 'recompute_every_n_epochs')
        if (surf_for_curv or surf_for_div or surf_for_chamfer_div) and surf_needs_update:
            with Timer.record('shape_boundary_helper.get_surface_pts'):
                self.p_surface, self.weights_surf_pts = self.shape_boundary_helper.get_surface_pts(z, 
                                                                plot=is_every_n_epochs_fulfilled(epoch, self.config, 'plot_surface_pts_every_n_epochs'),
                                                                plot_max_shapes=self.config['train_plot_max_n_shapes'],
                                                                interface_cutoff=self.config['exclude_surface_points_close_to_interface_cutoff'])
        
        ## compute chamfer div loss
        if self.config['lambda_chamfer_div'] > 0:   
            if self.p_surface is not None and epoch >= self.config.get('start_chamfer_div', 0):
                y_surface, p_surface_subsampled, batch_CD, pdCDdy = self.loss_dispatcher['chamfer_div'](z=z, p_surface=self.p_surface)
                loss_dict[LossKey('chamfer_div')] = (batch_CD, pdCDdy)
                field_dict[LossKey('chamfer_div')] = y_surface
                
                if not torch.equal(pdCDdy.data, torch.zeros_like(pdCDdy.data)) and is_every_n_epochs_fulfilled(epoch, self.config, 'plot_every_n_epochs'):
                    self.mpm.plot(self.plotter.plot_shape_and_points, 'plot_chamfer_grads',
                        arg_list=[p_surface_subsampled.detach().cpu().numpy(), 'Chamfer penalty'], kwargs_dict=dict(point_attribute=pdCDdy.data.detach().cpu().numpy()))

            else:
                loss_dict[LossKey('chamfer_div')] = (torch.tensor(0.0), torch.tensor(0.0))            
                field_dict[LossKey('chamfer_div')] = torch.tensor(0.0)
                            
        ## compute TO sublosses
        x_fem = None
        if self.config['lambda_comp'] > 0:
            x_fem = self.feni.fem_xy_torch_device
            if self.config['FEM']['add_x_offset']:
                x_fem = x_fem + (torch.rand(self.feni.fem_xy_torch_device.shape) - 0.5) * self.feni.grid_dist
            rho_flat = self.netp(*tensor_product_xz(x_fem, z)).squeeze(1)
            
            # update beta
            if epoch > self.config['beta_sched'].get('start_iter', 0) and epoch % self.config['beta_sched']['interval'] == 0:
                self.beta = math.exp2(min(math.log2(self.config['beta_sched']['init']) + (epoch / self.config['beta_sched']['stop_iter']) * math.log2(self.config['beta_sched']['max'] / self.config['beta_sched']['init']), 
                                    math.log2(self.config['beta_sched']['max'])))  # exponential increase; log formulation avoids numerical issues
                print(f'beta: {self.beta}')
            
            if not self.config['FEM']['use_filters']:
                rho_flat = heaviside(rho_flat, beta=self.beta, nf_is_density=self.config['nf_is_density'])
            
            if epoch > self.config['p_sched'].get('start_iter', 0) and epoch % self.config['p_sched']['interval'] == 0:
                self.p = min(self.config['p_sched']['init'] + (epoch / self.config['p_sched']['stop_iter'] * (self.config['p_sched']['max'] - self.config['p_sched']['init'])), 
                                    self.config['p_sched']['max'])  # exponential increase; log formulation avoids numerical issues
                print(f'p: {self.p}')
                rho_flat = rho_flat ** self.p
            
            rho_batch = einops.rearrange(rho_flat, '(bz n) -> bz n', bz=z.shape[0])
            (batch_C, batch_dCdrho), (batch_V, batch_dVdrho) = self.feni.calc_2d_TO_grad_field_batch(rho_batch=rho_batch, beta=self.beta)
            assert torch.all(torch.not_equal(batch_C, 0.0)), f'batch_C=0: something is probably off with the FEM setup'  # no C=0 allowed
            assert not torch.all(torch.eq(batch_dCdrho, 0.0)), f'batch_dCdrho all zeros: something is probably off with the FEM setup' # it's not allowed to have all zeros
           
            if self.config['lambda_comp'] > 0 and epoch >= self.config.get('start_comp', 0):
                # plot individual compliance fields
                if self.config['plot_grad_field'] and is_every_n_epochs_fulfilled(epoch, self.config, 'plot_every_n_epochs'):
                    if self.config['nx'] == 2:
                        grad_fields = einops.rearrange(batch_dCdrho[:, self.feni.sort_idcs], 'bz (y x) -> bz x y', x=self.feni.chosen_resolution[1]).detach().cpu().numpy()
                    elif self.config['nx'] == 3:
                        grad_fields = einops.rearrange(batch_dCdrho[:, self.feni.sort_idcs], 'bz (x y z) -> bz z y x', x=self.feni.chosen_resolution[0], y=self.feni.chosen_resolution[1]).detach().cpu().numpy()
                    self.mpm.plot(self.plotter.plot_grad_field, 'plot_grad_field', kwargs_dict={'grad_fields': grad_fields, 'fig_label': f'comp grad field', 'attributes': batch_C.detach().cpu().numpy()})
            
                batch_C, batch_dCdrho = batch_C * self.config.get('scale_comp', 1.0), batch_dCdrho * self.config.get('scale_comp', 1.0)
                loss_dict[LossKey('comp')] = (batch_C.mean(), batch_dCdrho)
                field_dict[LossKey('comp')] = rho_batch
                
            if self.config['lambda_vol'] > 0:
                if self.config['plot_grad_field'] and is_every_n_epochs_fulfilled(epoch, self.config, 'plot_every_n_epochs'):
                    if self.config['nx'] == 2:
                        grad_fields = einops.rearrange(batch_dVdrho[:, self.feni.sort_idcs], 'bz (y x) -> bz x y', x=self.feni.chosen_resolution[1]).detach().cpu().numpy()
                    elif self.config['nx'] == 3:
                        grad_fields = einops.rearrange(batch_dVdrho[:, self.feni.sort_idcs], 'bz (x y z) -> bz z y x', x=self.feni.chosen_resolution[0], y=self.feni.chosen_resolution[1]).detach().cpu().numpy()
                    self.mpm.plot(self.plotter.plot_grad_field, 'plot_grad_field', kwargs_dict={'grad_fields': grad_fields, 'fig_label': f'vol grad field', 'attributes': batch_V.detach().cpu().numpy()})
                batch_V = batch_V * self.config.get('scale_vol', 1.0)
                batch_dVdrho = batch_dVdrho * self.config.get('scale_vol', 1.0)
                loss_dict[LossKey('vol')] = (batch_V.mean(), batch_dVdrho)
                field_dict[LossKey('vol')] = rho_batch
        
        ## compute scalar sublosses
        for key in self.scalar_loss_keys:
            if epoch < self.config.get('start_'+key.base_key, 0):
                loss_dict[key] = torch.tensor(0.0)
            else:
                loss_dict[key] = self.loss_dispatcher[key.base_key](z=z, z_corners=z_corners, p=self.p,
                        epoch=epoch, batch=batch, p_surface=self.p_surface, weights_surf_pts=self.weights_surf_pts, x_fem=x_fem)

        ## compute total loss
        loss = self.loss_calculator.compute_loss_and_save_sublosses(loss_dict)
        
        # do backward here directly, as we need x_field
        if self.config.get('use_config', False):
            self.loss_calculator.backward_config(field_dict, self.model)
        else:
            self.loss_calculator.backward(field_dict)
        

        return loss, self.loss_calculator.get_dicts()
    
    def log_to_wandb(self, epoch, await_all=False):
        with Timer.record('plot_helper.pool await async results'):
            ## Wait for plots to be ready, then log them
            while self.cur_plot_epoch <= epoch:
                if not self.mpm.are_plots_available_for_epoch(self.cur_plot_epoch):
                    ## no plots for this epoch; just log the current losses
                    wandb.log(self.log_history_dict[self.cur_plot_epoch])
                    del self.log_history_dict[self.cur_plot_epoch]
                    self.cur_plot_epoch += 1
                elif self.mpm.plots_ready_for_epoch(self.cur_plot_epoch):
                    ## plots are available and ready
                    wandb.log(self.log_history_dict[self.cur_plot_epoch] | self.mpm.pop_results_dict(self.cur_plot_epoch))
                    del self.log_history_dict[self.cur_plot_epoch]
                    self.cur_plot_epoch += 1
                elif await_all:
                    ## plots are not ready yet - wait for them
                    self.logger.debug(f'Waiting for plots for epoch {self.cur_plot_epoch}')
                    time.sleep(1)
                else:
                    # print(f'Waiting for plots for epoch {cur_plot_epoch}')
                    break

    def init_loss_dispatcher(self):
        """
        Initializes the loss dispatcher, which is a dictionary mapping LossKey.base_key to loss functions.
        The loss functions are partial functions of the actual loss functions, with the model, logger, etc. already set.
        All the parameter set here are static and do not change during training.
        Dynamic parameters (e.g. z, p_surface, epoch) are passed to the (partial) loss functions at train time.
        At train time all (partial) loss functions are called with the same parameters, i.e all losses have the same signature.
        """
        loss_dispatcher = {
            # ginn losses
            # local
            'env': partial(loss_env, netp=self.netp, p_sampler=self.problem, level_set=self.config['level_set'], nf_is_density=self.config['nf_is_density']),
            'obst': partial(loss_obst, netp=self.netp, p_sampler=self.problem, level_set=self.config['level_set'], nf_is_density=self.config['nf_is_density']),
            'if': partial(loss_if, netp=self.netp, p_sampler=self.problem, level_set=self.config['level_set']),
            'if_normal': partial(loss_if_normal, netp=self.netp, p_sampler=self.problem, ginn_bsize=self.config['ginn_bsize'], loss_scale=self.config.get('scale_if_normal', 1), nf_is_density=self.config['nf_is_density']),
            
            # global
            'eikonal': partial(loss_eikonal, p_sampler=self.problem, netp=self.netp, scale_eikonal=self.config.get('scale_eikonal', 1), nf_is_density=self.config['nf_is_density']),
            'scc': partial(loss_scc, ph_manager=self.ph_manager),
            'curv': partial(loss_curv, netp=self.netp, logger=self.logger, \
                            max_curv=self.config['max_curv'], device=torch.get_default_device(), \
                            loss_scale=self.config.get('scale_curv', 1.0),
                            curvature_expression=self.config['curvature_expression'], \
                            strain_curvature_clip_max=self.config['strain_curvature_clip_max'], \
                            curvature_use_gradnorm_weights=self.config.get('curvature_use_gradnorm_weights', False), \
                            curvature_after_5000_epochs=self.config.get('curvature_after_5000_epochs', False),
                            curvature_pts_source=self.config['curvature_pts_source']),

            'div': partial(loss_div, netp=self.netp, logger=self.logger, p_sampler=self.problem, \
                           max_div=self.config['max_div'], ginn_bsize=self.config['latent_sampling']['ginn_bsize'], \
                           diversity_pts_source=self.config['diversity_pts_source'], \
                           div_norm_order=self.config['div_norm_order'], div_neighbor_agg_fn=self.config['div_neighbor_agg_fn'],
                           leinster_temp=self.config.get('leinster_temp', None), leinster_q=self.config.get('leinster_q', None),
                           loss_scale=self.config.get('scale_div', 1.0)),
            'chamfer_div': partial(loss_chamfer_div, netp=self.netp, chamfer_p=self.config.get('chamfer_p', 1), 
                                   max_div=self.config['max_div'], chamfer_div_eps=self.config.get('chamfer_div_eps', 1e-6), 
                                   loss_scale=self.config.get('scale_chamfer_div', 1), subsample=self.config.get('chamfer_div_subsample', None)),
            
            
           
            # data losses
            'lip': partial(loss_lip, netp=self.netp),
            'dirichlet': partial(loss_dirichlet, netp=self.netp, p_sampler=self.problem),
            'data': partial(loss_data, netp=self.netp),
            # null loss for disabling the objective loss
            'null': partial(loss_null, netp=self.netp),
            
            'rotsym': partial(loss_rotation_symmetric, netp=self.netp, p_sampler=self.problem, n_cycles=self.config['problem']['rotation_n_cycles']),
            
        }
        return loss_dispatcher