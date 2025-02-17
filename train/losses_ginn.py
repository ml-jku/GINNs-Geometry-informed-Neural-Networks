import time
from cycler import V
import einops
import torch
import typing

from torch._functorch.eager_transforms import jacrev
from torch import vmap
from torch.nn import functional as F


from GINN.fenitop.heaviside_filter import heaviside
from models.net_w_partials import NetWithPartials
from models.point_wrapper import PointWrapper
from util.model_utils import tensor_product_xz
from train.losses import CD_dCDdy_loss, chamfer_diversity_loss, diversity_loss, dirichlet_loss, envelope_loss_density, expression_curvature_loss, interface_loss, eikonal_loss, envelope_loss_sdf, l1_loss, mse_loss, normal_loss_euclidean, wasserstein_diversity_loss

type Scalar = torch.Tensor #TODO: need to implement this more rigorously using torchtypeing https://github.com/patrick-kidger/torchtyping
type Grad_Field = torch.Tensor

def loss_eikonal(z, p_sampler, netp, scale_eikonal, **kwargs) -> Scalar:
    loss_eikonal = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    xs_domain = p_sampler.sample_from_domain()
        ## Eikonal loss: NN should have gradient norm 1 everywhere
    y_x_eikonal = netp.vf_x(*tensor_product_xz(xs_domain, z))
    loss_eikonal = eikonal_loss(y_x_eikonal)
    loss_eikonal = scale_eikonal * loss_eikonal
    return loss_eikonal

def loss_obst(z, netp, p_sampler, level_set, nf_is_density, **kwargs) -> Scalar:
    loss_obst = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    ys_obst = netp(*tensor_product_xz(p_sampler.sample_from_obstacles(), z))
    if nf_is_density:
        loss_obst = envelope_loss_density(ys_obst, level_set=level_set)
    else:
        loss_obst = envelope_loss_sdf(ys_obst, level_set=level_set)
    return loss_obst

def loss_env(z, netp, p_sampler, level_set, nf_is_density, **kwargs) -> Scalar:
    loss_env = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    ys_env = netp(*tensor_product_xz(p_sampler.sample_from_envelope(), z)).squeeze(1)
    if nf_is_density:
        loss_env = envelope_loss_density(ys_env, level_set=level_set)
    else:
        loss_env = envelope_loss_sdf(ys_env)
    return loss_env

def loss_if(z, netp, p_sampler, level_set, **kwargs) -> Scalar:
    ys_BC = netp(*tensor_product_xz(p_sampler.sample_from_interface()[0], z)).squeeze(1)
    loss_if = interface_loss(ys_BC, level_set=level_set)
    return loss_if

def loss_if_normal(z, netp, p_sampler, nf_is_density, loss_scale, ginn_bsize=None, **kwargs) -> Scalar:
    pts_normal, target_normal = p_sampler.sample_from_interface()
    ys_normal = netp.vf_x(*tensor_product_xz(pts_normal, z)).squeeze(1)
    
    if nf_is_density:
        # if the neural field is a density field, make normal vectors unit vectors
        ys_normal = F.normalize(ys_normal, p=2, dim=-1)
    
    loss_if_normal = normal_loss_euclidean(ys_normal, target_normal=torch.cat([target_normal for _ in range(ginn_bsize)]))
    loss_if_normal = loss_scale * loss_if_normal
    return loss_if_normal

def loss_scc(z, ph_manager, **kwargs) -> Scalar:
    loss_scc = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    loss_sub0 = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    loss_super0 = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    success, loss_scc, loss_super0, loss_sub0 = ph_manager.calc_ph_loss_cripser(z)
    return loss_scc

def loss_data(z, netp, batch, z_corners, **kwargs) -> Scalar:
    loss_data = torch.tensor(0.0, device=z_corners.device, dtype=z_corners.dtype)
    x, y, idcs = batch
    # x, y, idcs = x.to(self.device), y.to(self.device), idcs.to(self.device)
    z_data = z_corners[idcs]
    y_pred = netp(x, z_data).squeeze(1)
    loss_data = mse_loss(y_pred, y)

    return loss_data

def loss_dirichlet(z, netp, p_surface, p_sampler, batch, **kwargs) -> Scalar:
    '''Uses surface points to enforce Dirichlet energy'''
    l_dirich =  torch.tensor(0.0, device=z.device, dtype=z.dtype)
    if p_surface is None:
        xs_domain = p_sampler.sample_from_domain()
        y_z = netp.vf_z(*tensor_product_xz(xs_domain, z)).squeeze(-1)
    else:
        y_z = netp.vf_z(p_surface.data, p_surface.z_in(z)).squeeze(1)
    l_dirich = dirichlet_loss(y_z)
    return l_dirich

def loss_lip(z, netp, **kwargs) -> Scalar:
    loss_lip = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    loss_lip = netp.get_lipschitz_loss()

    return loss_lip

def loss_vol2(rho_batch, vol_frac, nf_is_density, beta, **kwargs) -> Scalar:
    print(f'WARNING: vol2 heaviside is hardcoded to 128')
    rho_batch_vol = heaviside(rho_batch, beta, nf_is_density)
    vol_batch = rho_batch_vol.mean(dim=1)
    vol_loss_2 = (vol_batch / vol_frac - 1) ** 2
    vol_loss_2 = torch.clip(vol_loss_2, min=0.).mean()
    return vol_loss_2

# DIV LOSS

def loss_chamfer_div(z, netp: NetWithPartials, p_surface: PointWrapper, subsample, chamfer_p, max_div, chamfer_div_eps, loss_scale, **kwargs) -> Scalar:
    
    print(f'Total len of surface points: {len(p_surface)}')
    
    if subsample > 0:
        # subsample points from the surface
        list_of_pts = []
        for i_shape in range(p_surface.bz):
            # get at most 5000 points
            x_i = p_surface.pts_of_shape(i_shape)
            if len(x_i) > subsample:
                idx = torch.randperm(len(x_i))[:subsample]
                x_i = x_i[idx]
            list_of_pts.append(x_i)
        p_surface = PointWrapper.create_from_pts_per_shape_list(list_of_pts)
    
    y_x = netp.grouped_no_grad_fwd('vf_x', p_surface.data, p_surface.z_in(z)).squeeze(1)
    py_x = PointWrapper(data=y_x, map=p_surface._map) 

    start_t = time.time()
    CD, pdCDdy = chamfer_diversity_loss(p_surface, py_x, p=chamfer_p, eps=chamfer_div_eps)
    print(f'CD time: {time.time() - start_t}')
    
    if max_div is not None:
        print(f'CD before max_div: {CD}')
        CD = CD - max_div
        if CD < 0:
            return (torch.tensor(0.0, device=z.device, dtype=z.dtype),
                    p_surface,
                    torch.tensor(0.0, device=z.device, dtype=z.dtype),
                    torch.tensor(0.0, device=z.device, dtype=z.dtype))
    
    CD = CD * loss_scale
    pdCDdy.data = pdCDdy.data * loss_scale
    
    # TODO: is this superfluous? and we can return y_x?
    y_surface = netp(p_surface.data, p_surface.z_in(z)).squeeze(1)
    return y_surface, p_surface, CD, pdCDdy

def loss_wasserstein_div(geomloss_wasserstein_dist_fn, z, netp: NetWithPartials, p_sampler, beta, n_subsample, max_div, loss_scale, **kwargs) -> Scalar:
    
    x_domain_full = p_sampler.sample_from_inside_envelope()
    p_domain_full = PointWrapper.create_from_equal_bx(einops.repeat(x_domain_full, 'n d -> bz n d', bz=z.shape[0]))

    y_domain_full = netp.grouped_no_grad_fwd('vf', p_domain_full.data, p_domain_full.z_in(z)).squeeze(1)
    # y_domain_full = heaviside(y_domain_full, beta, nf_is_density=nf_is_density)
    y_domain_full = y_domain_full ** 2
    print(f'WARNING: Wasserstein loss is hardcoded to power 2')

    y_batch = einops.rearrange(y_domain_full, '(bz n) -> bz n', bz=z.shape[0])
    # sample along dim 1; y_batch does not need to be normalized for multinomial
    idx = torch.multinomial(y_batch, num_samples=n_subsample, replacement=False)
    x_domain = x_domain_full[idx]
    p_domain = PointWrapper.create_from_equal_bx(x_domain)

    y_x = netp.grouped_no_grad_fwd('vf_x', p_domain.data, p_domain.z_in(z)).squeeze(1)
    py_x = PointWrapper(data=y_x, map=p_domain._map)

    start_t = time.time()
    W, p_dWdy = wasserstein_diversity_loss(geomloss_wasserstein_dist_fn, p_domain, py_x)
    print(f'Wasserstein time: {time.time() - start_t}')
    
    W = W - max_div
    if W < 0:
        return (torch.tensor(0.0, device=z.device, dtype=z.dtype),
                torch.tensor(0.0, device=z.device, dtype=z.dtype),
                torch.tensor(0.0, device=z.device, dtype=z.dtype))
        
    W = W * loss_scale
    p_dWdy.data = p_dWdy.data * loss_scale
    
    # to pass gradients later through the network
    y_domain = netp(p_domain.data, p_domain.z_in(z)).squeeze(1)
    
    return p_domain, y_domain, W, p_dWdy
    
    

def loss_div(z, netp, p_sampler, p_surface, weights_surf_pts, logger, max_div, ginn_bsize, loss_scale,
             diversity_pts_source, div_norm_order, div_neighbor_agg_fn, leinster_temp, leinster_q, **kwargs) -> Scalar:
    assert diversity_pts_source in ['domain', 'surface'], f"diversity_pts_source ({diversity_pts_source}) must be 'domain' or 'surface'"
    loss_div = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    
    if diversity_pts_source == 'domain':
        y_div = netp(*tensor_product_xz(p_sampler.sample_from_inside_envelope(), z)).squeeze(1)
        weights_surf_pts = None
    elif diversity_pts_source == 'surface':
        if p_surface is None:
            logger.info('No surface points found - skipping diversity loss')
            return loss_div
        else:    
            y_div = netp(*tensor_product_xz(p_surface.data, z)).squeeze(1)  # [(bz k)] whereas k is n_surface_points; evaluate netp at all surface points for each shape
    loss_div = diversity_loss(einops.rearrange(y_div, '(bz k)-> bz k', bz=ginn_bsize), 
                                                                weights=weights_surf_pts,
                                                                norm_order=div_norm_order,
                                                                neighbor_agg_fn=div_neighbor_agg_fn,
                                                                leinster_temp=leinster_temp,
                                                                leinster_q=leinster_q)
    if torch.isnan(loss_div) or torch.isinf(loss_div):
        logger.warning(f'NaN or Inf loss_div: {loss_div}')
        loss_div = torch.tensor(0.0, device=z.device, dtype=z.dtype)

    loss_div = torch.clamp(loss_div - max_div, min=0)
    loss_div = loss_scale * loss_div
    return loss_div

# CURVATURE LOSS

def loss_curv(z, epoch, netp, p_surface, weights_surf_pts, logger, max_curv, loss_scale, device, curvature_pts_source, \
              curvature_expression, strain_curvature_clip_max, curvature_use_gradnorm_weights, \
              curvature_after_5000_epochs, **kwargs) -> Scalar:
    loss_curv = torch.tensor(0.0, device=device, dtype=z.dtype)
    k_theta_gradnorm_fn = get_k_theta_gradnorm_func(netp, curvature_expression, strain_curvature_clip_max, max_curv)
    if p_surface is None:
        logger.debug('No surface points found - skipping curvature loss')
    else:
        # check this here, as for vmap-ed curvature it can't be checked there
        assert weights_surf_pts is None or torch.allclose(weights_surf_pts.sum(), torch.tensor(1.0, device=weights_surf_pts.device, dtype=weights_surf_pts.dtype)), f"weights must sum to 1"
        
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
        
        loss_curv = max(torch.tensor(0.0, device=device, dtype=z.dtype), loss_curv - max_curv)

    if curvature_after_5000_epochs and epoch < 5000:
        loss_curv = torch.tensor(0.0, device=device)

    loss_curv = loss_scale * loss_curv

    return loss_curv


def get_k_theta_gradnorm_func(netp, curvature_expression, strain_curvature_clip_max, max_curv): 

    # non-vectorized loss function
    def curvature_loss_wrapper(params, x, z, weights):
        # for netp calls, use the properties f_x_ and f_xx_ instead of the methods f_x and f_xx
        y_x = netp.vf_x_(params, x, z).squeeze(1)
        y_xx = netp.vf_xx_(params, x, z).squeeze(1)
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

def loss_rotation_symmetric(z, netp, p_sampler, n_cycles, **kwargs) -> Scalar:
    loss_rot = torch.tensor(0.0, device=z.device, dtype=z.dtype)
    pts_rot_sym = p_sampler.sample_rotation_symmetric()
    y_rot_sym = netp(*tensor_product_xz(pts_rot_sym, z)).squeeze(1)
    y_reshaped = einops.rearrange(y_rot_sym, '(z k n) -> n k z', k=n_cycles, z=len(z))
    y_var = y_reshaped.var(dim=1)
    # replace NaNs or Infs with 0
    if torch.isnan(y_var).any() or torch.isinf(y_var).any():
        print(f'NaN or Inf in y_var')
    y_var[torch.isnan(y_var) | torch.isinf(y_var)] = 0.0
    loss_rot = y_var.mean()  # omitting the square as it's in the var
    return loss_rot

def loss_null(z, **kwargs) -> Scalar:
    return torch.tensor(0.0, device=z.device)


def loss_volume(z, netp: NetWithPartials, p_sampler, vol_frac, x_fem, beta, p, p_const, vol_loss, loss_scale, nf_is_density, **kwargs) -> Grad_Field:
    
    x = x_fem
    # print(f'WARNING: hard coded taking from envelop')
    # x = None
    if x is None:
        x = p_sampler.sample_from_inside_envelope()
        assert len(x) > 15000, f'x_domain should be at least as big as the mesh'
    
    y = netp(*tensor_product_xz(x, z)).squeeze(1)
    y = einops.rearrange(y, '(bz n) -> bz n', bz=z.shape[0])
    
    y = heaviside(y, beta=beta, nf_is_density=nf_is_density)
    exponent = p if p is not None else p_const
    y = y ** exponent
    
    mass_frac = y.mean(dim=1)

    if vol_loss == 'mse':
        V_loss = (mass_frac / vol_frac - 1) ** 2
    elif vol_loss == 'mae':
        V_loss = torch.abs(mass_frac / vol_frac - 1)
    elif vol_loss == 'mse_pushdown':
        V_loss = (mass_frac / vol_frac - 1) ** 2
        V_loss = torch.where(mass_frac > vol_frac, V_loss, torch.tensor(0.0, device=z.device, dtype=z.dtype))
    elif vol_loss == 'mse_unscaled_pushdown':
        V_loss = (mass_frac - vol_frac) ** 2
        V_loss = torch.where(mass_frac > vol_frac, V_loss, torch.tensor(0.0, device=z.device, dtype=z.dtype))
    elif vol_loss == 'mae_unscaled_pushdown':
        V_loss = torch.abs(mass_frac - vol_frac)
        V_loss = torch.where(mass_frac > vol_frac, V_loss, torch.tensor(0.0, device=z.device, dtype=z.dtype))
    else:
        raise ValueError(f'Unknown volume loss: {vol_loss}')
    
    V_loss = loss_scale * V_loss.mean()
    V_loss = V_loss ** 2 # square the loss to compensate for the square root in the ALM
    return V_loss