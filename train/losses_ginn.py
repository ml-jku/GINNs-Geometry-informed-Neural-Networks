import einops
import torch

from torch._functorch.eager_transforms import jacrev
from torch import vmap


from util.model_utils import tensor_product_xz
from train.losses import closest_shape_diversity_loss, dirichlet_loss, expression_curvature_loss, strain_curvature_loss, interface_loss, eikonal_loss, envelope_loss, l1_loss, mse_loss, normal_loss_euclidean, obstacle_interior_loss


def loss_eikonal(z, lambda_vec, p_sampler, netp, **kwargs):
    loss_eikonal = torch.tensor(0.0)
    xs_domain = p_sampler.sample_from_domain()
        ## Eikonal loss: NN should have gradient norm 1 everywhere
    y_x_eikonal = netp.vf_x(*tensor_product_xz(xs_domain, z))
    loss_eikonal, loss_al_eikonal = eikonal_loss(y_x_eikonal, lambda_vec=lambda_vec)
    return loss_eikonal, loss_al_eikonal

def loss_obst(z, lambda_vec, model, p_sampler, **kwargs):
    loss_obst = torch.tensor(0.0)
    ys_obst = model(*tensor_product_xz(p_sampler.sample_from_obstacles(), z))
    loss_obst, loss_al_obst = obstacle_interior_loss(ys_obst, lambda_vec=lambda_vec)
    return loss_obst, loss_al_obst

def loss_env(z, lambda_vec, model, p_sampler, **kwargs):
    loss_env = torch.tensor(0.0)
    ys_env = model(*tensor_product_xz(p_sampler.sample_from_envelope(), z)).squeeze(1)
    loss_env, loss_al_env = envelope_loss(ys_env, lambda_vec=lambda_vec)
    return loss_env, loss_al_env

def loss_if(z, lambda_vec, model, p_sampler, **kwargs):
    ys_BC = model(*tensor_product_xz(p_sampler.sample_from_interface()[0], z)).squeeze(1)
    loss_if, loss_al_if = interface_loss(ys_BC, lambda_vec=lambda_vec)
    return loss_if, loss_al_if

def loss_if_normal(z, lambda_vec, model, p_sampler, netp, ginn_bsize=None, nf_is_density=False, **kwargs):
    pts_normal, target_normal = p_sampler.sample_from_interface()
    ys_normal = netp.vf_x(*tensor_product_xz(pts_normal, z)).squeeze(1)
    loss_if_normal, loss_al_if_normal = normal_loss_euclidean(ys_normal, 
                                                                target_normal=torch.cat([target_normal for _ in range(ginn_bsize)]), 
                                                                lambda_vec=lambda_vec,
                                                                nf_is_density=nf_is_density)
    return loss_if_normal, loss_al_if_normal

def loss_scc(z, lambda_vec, ph_manager, **kwargs):
    loss_scc = torch.tensor(0.0)
    loss_sub0 = torch.tensor(0.0)
    loss_super0 = torch.tensor(0.0)
    success, loss_scc, loss_super0, loss_sub0 = ph_manager.calc_ph_loss_cripser(z)
    return loss_scc, torch.tensor(0.0, device=loss_scc.device)


#TODO implement augmented lagrangian (_al) terms for losses below 
def loss_data(z, lambda_vec, model, batch, z_corners, **kwargs):
    loss_data = torch.tensor(0.0)
    x, y, idcs = batch
    # x, y, idcs = x.to(self.device), y.to(self.device), idcs.to(self.device)
    z_data = z_corners[idcs]
    y_pred = model(x, z_data).squeeze(1)
    loss_data = mse_loss(y_pred, y)

    return loss_data, torch.tensor(0.0)

def loss_dirichlet(z, lambda_vec, model, batch, **kwargs):
    loss_dirichlet =  torch.tensor(0.0)
    
    x, y, _ = batch
    y_pred = model(*tensor_product_xz(x, z)).squeeze(1)
    loss_dirichlet = l1_loss(y_pred, y)
    return loss_dirichlet, torch.tensor(0.0)

def loss_lip(z, lambda_vec, model, **kwargs):
    loss_lip = torch.tensor(0.0)
    loss_lip = model.get_lipschitz_loss()

    return loss_lip, torch.tensor(0.0)

# DIV LOSS

def loss_div(z, lambda_vec, model, p_surface, weights_surf_pts, logger, max_div, ginn_bsize, div_norm_order, div_neighbor_agg_fn, **kwargs):
    loss_div = torch.tensor(0.0)
    loss_al_div = torch.tensor(0.0)
    if p_surface is None:
        logger.info('No surface points found - skipping diversity loss')
    else:
        y_div = model(*tensor_product_xz(p_surface.data, z)).squeeze(1)  # [(bz k)] whereas k is n_surface_points; evaluate model at all surface points for each shape
        loss_div, loss_al_div = closest_shape_diversity_loss(einops.rearrange(y_div, '(bz k)-> bz k', bz=ginn_bsize), 
                                                                    lambda_vec=lambda_vec,
                                                                    weights=weights_surf_pts,
                                                                    norm_order=div_norm_order,
                                                                    neighbor_agg_fn=div_neighbor_agg_fn)
        if torch.isnan(loss_div) or torch.isinf(loss_div):
            logger.warning(f'NaN or Inf loss_div: {loss_div}')
            loss_div = torch.tensor(0.0)
            loss_al_div = torch.tensor(0.0)
    
    loss_div = torch.clamp(loss_div - max_div, min=0)
    return loss_div, loss_al_div

# CURVATURE LOSS

def loss_curv(z, lambda_vec, epoch, netp, p_surface, weights_surf_pts, logger, max_curv, device, \
              curvature_expression, strain_curvature_clip_max, curvature_use_gradnorm_weights, \
              curvature_after_5000_epochs, **kwargs):
    loss_curv = torch.tensor(0.0)
    loss_curv_unweighted = torch.tensor(0.0)
    k_theta_gradnorm_fn = get_k_theta_gradnorm_func(netp, curvature_expression, strain_curvature_clip_max, max_curv)
    if p_surface is None:
        logger.debug('No surface points found - skipping curvature loss')
    else:
        # check this here, as for vmap-ed curvature it can't be checked there
        assert weights_surf_pts is None or torch.allclose(weights_surf_pts.sum(), torch.tensor(1.0)), f"weights must sum to 1"
        
        weights = weights_surf_pts
        if curvature_use_gradnorm_weights:
            # weights = self.k_theta_gradnorm_fn(self.netp.params_, self.p_surface.data, self.p_surface.z_in(z), torch.ones(size=(len(self.p_surface.data), 1), device=self.config['device']) / len(self.p_surface.data)) # gradnorm - unsqueeze is needed for vmap
            gn = k_theta_gradnorm_fn(netp.params_, p_surface.data, p_surface.z_in(z), 
                                     torch.ones(size=(len(p_surface.data), 1), device=device) / len(p_surface.data))
            weights = gn.sum() / gn
            weights = weights / weights.sum()
        
        y_x_surf = netp.vf_x(p_surface.data, p_surface.z_in(z)).squeeze(1)
        y_xx_surf = netp.vf_xx(p_surface.data, p_surface.z_in(z)).squeeze(1)
        loss_curv, loss_curv_unweighted = expression_curvature_loss(y_x_surf, y_xx_surf, 
                                                expression=curvature_expression,
                                                clip_max_value=strain_curvature_clip_max,
                                                weights=weights)
        
        loss_curv = max(torch.tensor(0.0, device=device), loss_curv - max_curv)

    if curvature_after_5000_epochs and epoch < 5000:
        loss_curv = torch.tensor(0.0, device=device)
        loss_curv_unweighted = torch.tensor(0.0, device=device)

    return loss_curv, loss_curv_unweighted


def get_k_theta_gradnorm_func(netp, curvature_expression, strain_curvature_clip_max, max_curv): 

    # non-vectorized loss function
    def curvature_loss_wrapper(params, x, z, weights):
        # for netp calls, use the properties f_x_ and f_xx_ instead of the methods f_x and f_xx
        y_x = netp.f_x_(params, x, z).squeeze(1)
        y_xx = netp.f_xx_(params, x, z).squeeze(1)
        loss_curv, loss_curv_unweighted = expression_curvature_loss(y_x, y_xx, 
                                            expression=curvature_expression,
                                            clip_max_value=strain_curvature_clip_max,
                                            weights=weights)
        loss_curv = torch.clamp(loss_curv - max_curv, min=0)
        return loss_curv

    # compute gradient wrt to first argument, which is theta
    k_theta = jacrev(curvature_loss_wrapper, argnums=0) # params, nx, nz, nx -> [ny, params]

    # vectorize
    vk_theta = vmap(k_theta, in_dims=(None, 0, 0, 0), out_dims=(0))  ## params, [bxz, nx], [bxz, nz] [bxz, nx] -> [bxz, params]

    def final_func(params_, x, z, weights):
        params = {key: tensor.detach() for key, tensor in params_.items()}
        res = vk_theta(params, x, z, weights)
        grads = torch.hstack([g.flatten(start_dim=1) for g in res.values()]) ## flatten batched grads per parameter
        # grads = [param.grad.detach().flatten() for param in params if param.grad is not None ]
        grad_norm = grads.norm(dim=1)
        return grad_norm
        
    return final_func
