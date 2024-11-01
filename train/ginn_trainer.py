from math import e
import time
from tracemalloc import start
from turtle import st
from typing import final
from attr import s
from matplotlib.artist import kwdoc
import numpy as np
import torch
from torch import vmap
from torch.utils.data import DataLoader
from tqdm import trange
import logging

import wandb
from GINN.shape_boundary_helper import ShapeBoundaryHelper
from GINN.data.simjeb_dataset import SimJebDataset
from GINN.evaluation.jeb_meter import JebMeter
from GINN.evaluation.simple_obst_meter import SimpleObstacleMeter
from train.opt.opt_util import get_opt, opt_step
from train.opt.adaptive_util import init_mu_dict, init_aug_lagr_lambas, init_nu_dict, lambda_balancing, scale_losses, adaptive_penalty_update
from train.train_utils.loss_keys import LossKey, get_loss_key_list
from train.train_utils.autoclip import AutoClip
from util.model_utils import get_stateless_net_with_partials, tensor_product_xz

from GINN.plot.plotter_2d import Plotter2d
from GINN.plot.plotter_3d import Plotter3d
from GINN.ph.ph_plotter import PHPlotter
from GINN.ph.ph_manager import PHManager
from GINN.speed.timer import Timer
from train.train_utils.latent_sampler import get_z_corners, sample_z
from util.checkpointing import load_model_optim_sched, save_model_every_n_epochs
from util.misc import get_model, get_problem_sampler, is_every_n_epochs_fulfilled, set_and_true
from train.losses_ginn import *
from functools import partial


class Trainer():
    
    def __init__(self, config, model, mp_manager) -> None:
        self.config = config
        self.model = model
        self.mpm = mp_manager
        self.logger = logging.getLogger('trainer')
        self.device = config['device']
        self.netp = get_stateless_net_with_partials(self.model, nz=self.config['nz'])
        self.p_sampler = get_problem_sampler(self.config)

        self.timer = Timer(self.config, lock=mp_manager.get_lock())
        self.mpm.set_timer(self.timer)  ## weak circular reference
        self.plotter = Plotter2d(self.config) if config['nx']==2 else Plotter3d(self.config)
        self.ph_manager = PHManager(self.config, self.model)
        self.ph_plotter = PHPlotter(self.config)

        self.shape_boundary_helper = ShapeBoundaryHelper(self.config, self.netp, self.mpm, self.plotter, self.p_sampler.sample_from_interface()[0], self.device)
        self.auto_clip = AutoClip(config)
        
        # data
        if ('lambda_data' in config) and (config['lambda_data'] > 0):
            # self.dataloader = DataLoader(SimJebDataset(config), ginn_bsize=config['data_bsize'], shuffle=True, num_workers=config['dataloader_num_workers'])
                                        #  generator=torch.Generator(device=config['device']))
            self.data = SimJebDataset(config)
            self.shuffle_idcs = torch.randperm(len(self.data), device=self.device)
            self.cur_data_idx = 0
        
        self.meter = None
        if config['problem'] in ['simple_2d', 'double_obstacle']:
            self.meter = SimpleObstacleMeter.create_from_problem_sampler(config, self.p_sampler)
        elif config['problem'] == 'simjeb':
            self.meter = JebMeter(config)
            self.config['n_points_envelope'] = self.config['n_points_envelope'] // 3 * 3

        self.z_corners = get_z_corners(self.config)
        self.p_surface = None
        self.weights_surf_pts = None
        self.cur_plot_epoch = 0
        self.log_history_dict = {}
        
        self.sub_grad_norm_dict = {}
        self.loss_keys = get_loss_key_list(self.config)
        self.mu_dict = init_mu_dict(self.loss_keys, self.config['device'])
        self.nu_dict = init_nu_dict(self.loss_keys)
        self.aug_lagr_lambda_dict = init_aug_lagr_lambas(self.config, self.loss_keys)
        self.init_loss_dispatcher()
        self.init_objective_loss()

    def train(self):
        
        # get optimizer and scheduler
        opt = get_opt(self.config['opt'], self.config, self.model.parameters(), self.aug_lagr_lambda_dict)
        sched = None
        if set_and_true('use_scheduler', self.config):
            def warm_and_decay_lr_scheduler(step: int):
                return self.config['scheduler_gamma'] ** (step / self.config['decay_steps'])
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=warm_and_decay_lr_scheduler)
        # maybe load model, optimizer and scheduler
        self.model, opt, sched = load_model_optim_sched(self.config, self.model, opt, sched)
        
        # get z
        z = sample_z(self.config, epoch=0, previous_z=None)
        z_corners = self.z_corners
        self.logger.info(f'Initial z: {z}')        
        self.logger.info(f'z_corners: {z_corners}')
        if self.config['train_plot_n_shapes'] < len(z) + len(z_corners):
            print(f'WARNING: plotting only {self.config["train_plot_n_shapes"]} out of {len(z) + len(z_corners)} shapes, namely z_corner[0], zs, z_corner[1] (if available)')
        
        # training/validation loop
        for epoch in (pbar := trange(self.config['max_epochs'], leave=True, position=0, colour="yellow")):
            cur_log_dict = {}
            self.mpm.update_epoch(epoch)

            batch = None
            if self.config['lambda_data'] > 0:
                batch = self._get_data_batch()

            ## validation
            if epoch > 0 and is_every_n_epochs_fulfilled(epoch, self.config, 'valid_every_n_epochs'):
                self.model.eval()
                # get validation z
                z_val = sample_z(self.config, epoch, previous_z=None, is_validation=True)
                # plot validation shapes
                self.plotter.reset_output(self.p_sampler.recalc_output(self.netp.f_, self.netp.params_, z_val), epoch=epoch)
                self.mpm.plot(self.plotter.plot_shape, 'val_plot_shape', arg_list=[self.p_sampler.constr_pts_dict, 'Val Boundary'])
                # compute validation metrics
                if epoch > 0 and is_every_n_epochs_fulfilled(epoch, self.config, 'val_shape_metrics_every_n_epochs'):
                    mesh_or_contour = self.p_sampler.get_mesh_or_contour(self.netp.f_, self.netp.params_, z_val)
                    if mesh_or_contour is not None:
                        self.mpm.metrics(self.meter.get_average_metrics_as_dict, arg_list=[mesh_or_contour], kwargs_dict={'prefix': 'm_val_'})

            ## training
            self.model.train()
            z = sample_z(self.config, epoch, z)
            
            ## reset base shape if needed
            if self.plotter.do_plot():
                z_plot = z
                if self.config['lambda_data'] > 0:
                    # sub-sample z to only train_plot_n_shapes
                    z_sub = z[:self.config['train_plot_n_shapes'] - len(z_corners)] if self.config['train_plot_n_shapes'] < len(z) - len(z_corners) else z
                    z_plot = torch.cat([z_corners[:1], z_sub, z_corners[1:]], dim=0) # works if there are 1 or 2 data shapes in the corners
                self.plotter.reset_output(self.p_sampler.recalc_output(self.netp.f_, self.netp.params_, z_plot), epoch=epoch)
                self.mpm.plot(self.plotter.plot_shape, 'plot_shape', arg_list=[self.p_sampler.constr_pts_dict])

                if self.ph_manager is not None:
                    PH = self.ph_manager.calc_ph(z)
                    PH = [ph.get() for ph in PH]
                    # subsample to train_plot_n_shapes
                    PH = PH[:self.config['train_plot_n_shapes']] if self.config['train_plot_n_shapes'] < len(PH) else PH
                    self.mpm.plot(self.ph_plotter.plot_ph_diagram, 'plot_ph_diagram', arg_list=[PH], kwargs_dict={})

            ## compute metrics _before_ taking the GD step
            if epoch > 0 and is_every_n_epochs_fulfilled(epoch, self.config, 'shape_metrics_every_n_epochs'):
                mesh_or_contour = self.p_sampler.get_mesh_or_contour(self.netp.f_, self.netp.params_, z)
                if mesh_or_contour is not None:
                    self.mpm.metrics(self.meter.get_average_metrics_as_dict, arg_list=[mesh_or_contour], kwargs_dict={'prefix': 'm_train_'})
                
                if self.config['lambda_data'] > 0:
                    mesh_or_contour = self.p_sampler.get_mesh_or_contour(self.netp.f_, self.netp.params_, z_corners)
                    if mesh_or_contour is not None:
                        self.mpm.metrics(self.meter.get_average_metrics_as_dict, arg_list=[mesh_or_contour], kwargs_dict={'prefix': 'm_corners_'})
            
            ## gradients and optimizer step
            loss_log_dict = opt_step(opt, epoch, self.model, self.compute_losses, z, z_corners, batch, self.auto_clip)
            cur_log_dict.update(loss_log_dict)
            cur_log_dict.update(self.sub_grad_norm_dict)
            cur_log_dict.update(self.mu_dict)
            if set_and_true('use_scheduler', self.config):
                sched.step()

            if ('adaptive_penalty_updates' in self.config) and (self.config['adaptive_penalty_updates']):
                self.mu_dict, self.aug_lagr_lambda_dict, self.nu_dict = adaptive_penalty_update(self.loss_keys, self.mu_dict, self.aug_lagr_lambda_dict, self.nu_dict, loss_log_dict, self.config)

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
            pbar.set_description(f"{epoch}:" + " ".join([f"{k}:{v.item():.1e}" for k, v in loss_log_dict.items()]))
        
        ## Final logging
        self.log_to_wandb(epoch, await_all=True)
        ## Finished
        self.timer.print_logbook()

    def _get_data_batch(self):
        ginn_bsize = self.config['data_bsize'] if self.config['data_bsize'] > 0 else len(self.data)
        start_idx = self.cur_data_idx
        end_idx = min(start_idx + ginn_bsize, len(self.data))
        
        perm_idcs = self.shuffle_idcs[start_idx:end_idx]
        pts = self.data.pt_coords[perm_idcs]
        sdf = self.data.sdf_vals[perm_idcs]
        idcs = self.data.idcs[perm_idcs]
        
        if self.cur_data_idx + ginn_bsize > len(self.data):
            start_idx = 0
            end_idx = ginn_bsize - (len(self.data) - self.cur_data_idx)
            perm_idcs = self.shuffle_idcs[start_idx:end_idx]
            pts = torch.cat([pts, self.data.pt_coords[perm_idcs]])
            sdf = torch.cat([sdf, self.data.sdf_vals[perm_idcs]])
            idcs = torch.cat([idcs, self.data.idcs[perm_idcs]])
            self.shuffle_idcs = torch.randperm(len(self.data), device=self.device)
            
        self.cur_data_idx = end_idx
        return pts, sdf, idcs
        
        # if not hasattr(self, 'data_iter'):
        #     self.data_iter = iter(self.dataloader)
        
        # try:
        #     return next(self.data_iter)
        # except StopIteration:
        #     self.data_iter = iter(self.dataloader)
        #     return next(self.data_iter)

    def compute_losses(self, z, epoch, batch=None, z_corners=None):

       # compute surface points
        if any(key in self.loss_keys+[self.objective_loss[2]] for key in [LossKey('curv'), LossKey('div')]) and (epoch >= self.config['surf_pts_warmup_n_epochs']):
            if self.p_surface is None or epoch % self.config['surf_pts_recompute_every_n_epochs'] == 0:
                with Timer.record('shape_boundary_helper.get_surface_pts'):
                    self.p_surface, self.weights_surf_pts = self.shape_boundary_helper.get_surface_pts(z)

        sub_loss_dict = {}
        sub_loss_unweighted_dict = {}
        sub_al_loss_dict = {}
        al_vec_l2_dict = {}

        obj_loss, obj_loss_unweighted = self.objective_loss[1](z=z, lambda_vec=None, epoch=epoch, batch=batch, p_surface=self.p_surface, weights_surf_pts=self.weights_surf_pts)
        sub_loss_dict[self.objective_loss[2].loss_key] = self.objective_loss[0] * obj_loss
        sub_loss_unweighted_dict[self.objective_loss[2].loss_unweighted_key] = obj_loss_unweighted

        for key in self.loss_keys:

            lambda_value, lambda_vec = self.aug_lagr_lambda_dict[key.lambda_key]
            mu_value = self.mu_dict[key.mu_key]
            if lambda_vec is not None:
                lambda_vec = lambda_value * lambda_vec

            sub_loss, sub_al_loss = self.loss_dispatcher[key.base_key](z=z, lambda_vec=lambda_vec, epoch=epoch, batch=batch, p_surface=self.p_surface, weights_surf_pts=self.weights_surf_pts)

            sub_loss_unweighted_dict[key.loss_unweighted_key] = sub_loss
            if key.base_key == 'curv': #WARNING: hack to scale curvature loss
                sub_loss = sub_loss * self.config['curv_weight']
        
            penalty_term = sub_loss
            lambda_term = torch.sqrt(sub_loss)
            sub_al_loss_dict[key.lagrangian_key] = lambda_value * lambda_term
            sub_al_loss_dict[key.mu_key] = 0.5 * mu_value * penalty_term
            
            if self.config['use_augmented_lagrangian']:
                al_vec_l2_dict[key.lambda_key] = torch.norm(lambda_value)
                al_vec_l2_dict[key.mu_key] = mu_value
                
                sub_loss_dict[key.loss_key] = 0.5 * mu_value * penalty_term + lambda_value * lambda_term
            else: 
                sub_loss_dict[key.loss_key] = mu_value * penalty_term

        if ('weight_rescale_on' in self.config) and (self.config['weight_rescale_on']) and (epoch>0):
            raise NotImplementedError('Weight rescaling not implemented yet with new loss structure')
            if ("weight_rescale_interval" in self.config) and (epoch%self.config['weight_rescale_interval'] == 0):
                self.sub_grad_norm_dict = grad_norm_sub_losses(self.model, sub_loss_unweighted_dict)
                self.mu_dict = lambda_balancing(self.mu_dict, self.sub_grad_norm_dict, self.config['weight_rescale_alpha'])


        loss = sum(sub_loss_dict.values())
        return loss, sub_loss_dict, sub_loss_unweighted_dict, sub_al_loss_dict, al_vec_l2_dict
    
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
        self.loss_dispatcher = {
            # ginn losses
            'curv': partial(loss_curv, netp=self.netp, logger=self.logger, \
                            max_curv=self.config['max_curv'], device=self.config['device'], \
                            curvature_expression=self.config['curvature_expression'], \
                            strain_curvature_clip_max=self.config['strain_curvature_clip_max'], \
                            curvature_use_gradnorm_weights=set_and_true('curvature_use_gradnorm_weights', self.config), \
                            curvature_after_5000_epochs=set_and_true('curvature_after_5000_epochs', self.config)),

            'div': partial(loss_div, model=self.model, logger=self.logger, \
                           max_div=self.config['max_div'], ginn_bsize=self.config['ginn_bsize'], \
                           div_norm_order=self.config['div_norm_order'], div_neighbor_agg_fn=self.config['div_neighbor_agg_fn']),

            'eikonal': partial(loss_eikonal, p_sampler=self.p_sampler, netp=self.netp),
            'obst': partial(loss_obst, model=self.model, p_sampler=self.p_sampler),
            'if_normal': partial(loss_if_normal, model=self.model, p_sampler=self.p_sampler, \
                                 netp=self.netp, ginn_bsize=self.config['ginn_bsize'], nf_is_density=set_and_true('use_density', self.config)),
            'if': partial(loss_if, model=self.model, p_sampler=self.p_sampler),
            'env': partial(loss_env, model=self.model, p_sampler=self.p_sampler),
            'scc': partial(loss_scc, ph_manager=self.ph_manager),

            # data losses
            'lip': partial(loss_lip, model=self.model),
            'dirichlet': partial(loss_dirichlet, model=self.model),
            'data': partial(loss_data, model=self.model, z_coners=self.z_corners),
        }


    def init_objective_loss(self):

        assert self.loss_dispatcher is not None, 'Loss dispatcher must be initialized before objective loss'
        assert self.aug_lagr_lambda_dict is not None, 'Augmented lagrangian lambdas must be initialized before objective loss'
        max_objective = 'max_'+self.config['objective']
        assert not (max_objective in self.config and abs(self.config[max_objective]) > 0.), f'Max value for objective loss not allowed in config'
        objective = self.config['objective']
        obj_key = LossKey(objective)
        assert ('lambda_'+objective in self.config) and (self.config['lambda_'+objective] > 0), f'Weighting for objective loss not found in config or not positive'
        self.objective_loss = (self.config['lambda_'+objective] * torch.tensor(1.0, device=self.config['device']), 
                            self.loss_dispatcher[obj_key.base_key], obj_key)
        self.loss_dispatcher.pop(obj_key.base_key)
        self.aug_lagr_lambda_dict.pop(obj_key.lambda_key)
        self.mu_dict.pop(obj_key.mu_key)
        self.loss_keys.remove(obj_key)
        self.logger.info(f'Using objective loss: {self.config["objective"]}')