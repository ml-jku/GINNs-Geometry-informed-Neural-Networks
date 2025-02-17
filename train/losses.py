from math import dist
import einops
from numpy import square
import torch
import torch.nn.functional as F

from models.point_wrapper import PointWrapper
from util.misc import geometric_mean, inverse_geometric_mean


def l1_loss(y_pred, y_true):
    assert y_pred.shape == y_true.shape, "avoid errors where unsqueezed dim is present"
    return F.l1_loss(y_pred, y_true)

def mse_loss(y_pred, y_true):
    assert y_pred.shape == y_true.shape, "avoid errors where unsqueezed dim is present"
    return F.mse_loss(y_pred, y_true)

# GINN losses

def envelope_loss_sdf(ys, level_set=0.0):
    '''
    If the SDF-value in the exterior of the envelope is smaller than the level_set, push it up. 
    '''
    # note that ys[ys<level_set] - level_set) is always negative
    loss = (ys[ys<level_set] - level_set).square().sum()
    return torch.tensor(0.0) if torch.isnan(loss) else loss

def envelope_loss_density(ys_env, level_set):
    '''
    If the Density-value in the exterior of the envelope is larger than the level_set, push it down.
    '''
    # note that ys_env[ys_env>level_set] - level_set) is always positive
    loss = (ys_env[ys_env>level_set] - level_set).square().sum()
    return torch.tensor(0.0) if torch.isnan(loss) else loss

def interface_loss(ys, level_set):
    '''
    The SDF-value at the boundary should be 'level_set'. This is done with mean-squared error.
    '''
    loss = (ys-level_set).square().mean()
    return loss

def normal_loss_euclidean(y_x, target_normal):
    """
    Prescribe normals at the interface via MSE.
    
    
    """
    assert y_x.shape == target_normal.shape, f"Shapes do not match: {y_x.shape} vs {target_normal.shape}"
    loss = (y_x - target_normal).square().mean()
    return loss

def eikonal_loss(y_x, **kwargs):
    '''
    Takes the gradient of points and pushes them towards unit-length
    '''
    y_x_mag = y_x.square().sum(2).sqrt()
    loss = (y_x_mag - 1).square().mean()
    return loss

## interpolation

def dirichlet_loss(y_z):
    '''
    The gradient of y wrt to z is pushed towards 0.
    '''
    loss = y_z.square().mean()
    return loss

## Differentiable Chamfer Distance

def CD_dCDdy_loss(x, y_x, x2, eps, **kwargs):
    '''Compute the Chamfer Distance between two point sets x and x2
        The euclidean distance is used as the distance metric.
        CD = 1/N sum_{x in x} min_{x2 in x2} d(x - x2)
        with d(x) = ||x||_2
    '''
    assert x.shape[0] == y_x.shape[0], f"Shapes do not match: {x.shape} vs {y_x.shape}"
    N = x.shape[0]
    
    diff = x[:, None, :] - x2[None, :, :].detach()
    dist = torch.norm(diff, dim=-1, p=2)
    min_dists, min_idcs = torch.min(dist, dim=-1)
    min_diff_v = diff[torch.arange(diff.size(0)), min_idcs]
    
    # compute CD
    CD = min_dists.mean()
    
    # compute dCD/dy
    dCDdx = 1/N * min_diff_v / (eps + min_dists[:, None])
    dxdy = y_x / (eps + torch.norm(y_x, dim=-1, p=2, keepdim=True).square())
    dCDdy_ = einops.einsum(dCDdx, dxdy, 'n d, n d -> n')
    
    # Why negative here?
    # AB seemed to have an explanation for it.
    # reverse sign for the gradient field; this is independent of the sdf/density field
    dCDdy_ = -dCDdy_
    return CD, dCDdy_

def CD_dCDdy_loss_with_p(x, y_x, x2, p=1, eps=1e-9):
    '''Compute the Chamfer Distance between two point sets x and x2
        The euclidean distance is used as the distance metric.
        CD = 1/N sum_{x in x} min_{x2 in x2} d(x - x2)^p
        with d(x) = ||x||_2
    '''
    assert x.shape[0] == y_x.shape[0], f"Shapes do not match: {x.shape} vs {y_x.shape}"
    N = x.shape[0]
    
    diff = x[:, None, :] - x2[None, :, :].detach()
    dist = torch.norm(diff, dim=-1, p=2)
    min_dists, min_idcs = torch.min(dist, dim=-1)
    min_diff_v = diff[torch.arange(diff.size(0)), min_idcs]
    
    # compute CD
    inner = (min_dists**p).mean()
    CD = inner**(1/p)
    
    # compute dCD/dy
    dCDdinner = 1/p * inner**(1/p - 1)
    dCDdx = dCDdinner * ( p/N * min_dists[:, None]**(p-1) ) * ( min_diff_v / (eps + min_dists[:, None]) )
    dxdy = y_x / (eps + torch.norm(y_x, dim=-1, p=2, keepdim=True).square())
    dCDdy_ = einops.einsum(dCDdx, dxdy, 'n d, n d -> n')
    
    # Why negative here?
    # AB seemed to have an explanation for it.
    # reverse sign for the gradient field; this is independent of the sdf/density field
    dCDdy_ = -dCDdy_
    return CD, dCDdy_


def chamfer_diversity_loss(px: PointWrapper, py_x: PointWrapper, **kwargs):
    '''
    This loss is used to enforce diversity with the chamfer distance as a metric.
    '''
    L_list = []
    dLdy_list = []
    with torch.no_grad():
        for i_shape in range(px.bz):
            L_min = torch.inf
            dLdy_min = None
            for j_shape in range(px.bz):
                if i_shape == j_shape:
                    continue
                
                if len(px.pts_of_shape(i_shape)) == 0 or len(px.pts_of_shape(j_shape)) == 0:
                    continue
                # L, dLDdy = CD_dCDdy_loss(px.pts_of_shape(i_shape), py_x.pts_of_shape(i_shape), px.pts_of_shape(j_shape), **kwargs)
                L, dLDdy = CD_dCDdy_loss_with_p(px.pts_of_shape(i_shape), py_x.pts_of_shape(i_shape), px.pts_of_shape(j_shape), **kwargs)
                if L < L_min:
                    L_min = L
                    dLdy_min = dLDdy
                    
            # negative as we want to maximize the minimum CD
            if L_min == torch.inf:
                L_list.append(torch.tensor(0.0))
                dLdy_list.append(torch.zeros(px.pts_of_shape(i_shape).shape[0]))
            else:
                L_list.append(-L_min)
                dLdy_list.append(-dLdy_min)
    # aggregate the individual CD and dCDdy; use sum as in Eq. 26 of GINNs (https://arxiv.org/pdf/2402.14009)
    L = torch.stack(L_list).sum()
    p_dLdy = PointWrapper.create_from_pts_per_shape_list(dLdy_list)
    # pdCDdy.data = pdCDdy.data / len(dCDdy_list)  # no normalization as we sum over all shapes
    return L, p_dLdy

def wasserstein_diversity_loss(loss_fn, px: PointWrapper, py_x: PointWrapper, **kwargs):
    '''
    This loss is used to enforce diversity with the chamfer distance as a metric.
    '''
    with torch.no_grad():
        # compute distances only once as wasserstein distance is symmetric
        L_dict = {}
        dLdy_dict = {}
        for i_shape in range(px.bz):
            for j_shape in range(i_shape+1, px.bz, 1):
                L, dLdy, dLdy2 = W_dWdy_dWdy2_loss(loss_fn, px.pts_of_shape(i_shape), py_x.pts_of_shape(i_shape), px.pts_of_shape(j_shape), py_x.pts_of_shape(j_shape), **kwargs)
                
                key1 = f'{i_shape}{j_shape}'
                L_dict[key1] = L
                dLdy_dict[key1] = dLdy

                key2 = f'{j_shape}{i_shape}'
                L_dict[key2] = L
                dLdy_dict[key2] = dLdy2

        
        L_list = []
        dLdy_list = []
        for i_shape in range(px.bz):
            min_L = torch.inf
            min_dLdy = None
            for j_shape in range(px.bz):
                if i_shape == j_shape: 
                    continue
                key = f'{i_shape}{j_shape}'
                if L_dict[key] < min_L:
                    min_L = L_dict[key]
                    min_dLdy = dLdy_dict[key]
                    
            # negative as we want to maximize the minimum CD
            L_list.append((-1) * min_L)
            dLdy_list.append((-1) * min_dLdy)
        
        # aggregate the individual CD and dCDdy; use sum as in Eq. 26 of GINNs (https://arxiv.org/pdf/2402.14009)
        L = torch.stack(L_list).sum()
        p_dLdy = PointWrapper.create_from_pts_per_shape_list(dLdy_list)
        return L, p_dLdy


def W_dWdy_dWdy2_loss(loss_fn, x, y_x, x2, y_x2, eps=1e-9):
    '''Compute the Wasserstein distance L_w and gradients dWdx between two point sets x and x2
        from: https://www.kernel-operations.io/geomloss/_auto_examples/optimal_transport/plot_optimal_transport_2D.html#sphx-glr-auto-examples-optimal-transport-plot-optimal-transport-2d-py
    '''
    with torch.enable_grad():
        x_in = x.clone().detach().requires_grad_(True)
        x2_in = x2.clone().detach().requires_grad_(True)
        L_w = loss_fn(x_in, x2_in)
        [dWdx, dWdx2] = torch.autograd.grad(L_w, [x_in, x2_in])
        
    dxdy = y_x / (eps + torch.norm(y_x, dim=-1, p=2, keepdim=True).square())
    dCDdy_ = einops.einsum(dWdx, dxdy, 'n d, n d -> n')
    
    dx2dy = y_x2 / (eps + torch.norm(y_x2, dim=-1, p=2, keepdim=True).square())
    dCDdy2_ = einops.einsum(dWdx2, dx2dy, 'n d, n d -> n')
    
    # Why negative here?
    # AB seemed to have an explanation for it.
    # reverse sign for the gradient field; this is independent of the sdf/density field
    dCDdy_ = -dCDdy_
    dCDdy2_ = -dCDdy2_
    return L_w, dCDdy_, dCDdy2_


def pairwise_dist_fields(y, norm_order, weights=None):
    """
    Compute the pairwise distance between fields via a Monte-Carlo approximation.
    does a generalization of (y.unsqueeze(0) - y.unsqueeze(1)).square().mean(-1).sqrt() to arbitrary p
    accepts arbitrary number of fields, points, and weights
    y: [num_fields, num_points] tensor
    pairwise_dist[i, j] contains the distance between fields i and j
    """
    assert y.ndim == 2, f"y must be 2D, got {y.ndim}"
    n_fields, n_points = y.shape
    if weights is not None:
        assert weights.shape[0] == y.shape[1], f"weights {weights} and y {y} must have same length"
        assert torch.allclose(weights.sum(), torch.tensor(1.0)), f"weights {weights} must sum to 1"
        
    ## NOTE: AR did some workaround below because the following commented code is unstable and leads to NaNs
    # pairwise_dist = (y.unsqueeze(0) - y.unsqueeze(1)).square()
    # if weights is not None:
    #     pairwise_dist = (pairwise_dist * weights).sum(-1)
    # else:
    #     pairwise_dist = pairwise_dist.mean(-1)
    # res = pairwise_dist.sqrt()
    
    diff = y.unsqueeze(0) - y.unsqueeze(1)  # [num_fields, num_fields, num_points]
    weighted_diff = diff if weights is None else diff * weights**(1/norm_order)  # [num_fields, num_fields, num_points]
    ## HINT: the norm-function is stable, but the square().mean().sqrt() is not
    pairwise_dist = torch.linalg.vector_norm(weighted_diff, ord=norm_order, dim=-1)  # [num_fields, num_fields]
    # the norm takes the sum over all points, so we need to divide by the sqrt of the number of points
    if weights is None:
        pairwise_dist = pairwise_dist / (n_points**(1/norm_order))  # divide to get the mean
    ## for debugging only
    # res.register_hook(lambda x: print(f'res pairwise: {x}'))
    return pairwise_dist 

def diversity_loss(y, norm_order, weights=None, neighbor_agg_fn='min', leinster_temp=None, leinster_q=None):
    """
    Returns the diversity loss.
    Compute all the pairwise distances.
    neighbors_agg_fn: 'min', 'mean', 'sum', 'leinster'
    Leinster works with similarities, the others with distances.
        - Leinsters: For every field, compute the similarity to all other fields. Then mean().square()
        - Min/mean/sum: For every field, aggregate the distances to all other fields. Then sqrt().mean().square()
    y: [num_fields, num_points] tensor
    """
    ## Compute pairwise distances
    distances = pairwise_dist_fields(y, norm_order=norm_order, weights=weights)
    
    if neighbor_agg_fn == 'leinster':
        # IMPORTANT: We assume p is uniform and take formula from page 175 of Leinster's book
        # the temperature implicitly sets the base of the exponential
        assert leinster_temp is not None, "temperature must be provided for neighbor_agg_fn 'leinster'"
        assert leinster_q is not None, "q must be provided for neighbor_agg_fn 'leinster'"
        similarities = torch.exp(-1 * distances / leinster_temp) # [num_fields, num_fields]
        ordinariness = similarities.mean(dim=1)  # [num_fields]
        
        if leinster_q == 1:
            diversity = inverse_geometric_mean(ordinariness, dim=0)
        else:
            ordinariness = ordinariness.pow(leinster_q - 1)
            diversity = ordinariness.mean().pow(1/(1 - leinster_q))
        return (-1) * diversity
    
    if neighbor_agg_fn == 'min':
        ## Create a mask to exclude diagonal elements
        ## Find the minimum distance for each point
        distances = distances.masked_fill(torch.eye(distances.size(0), dtype=torch.bool), float('inf'))
        dists, _ = distances.min(dim=1)
    elif neighbor_agg_fn == 'mean':
        dists = distances.mean(dim=1)
    elif neighbor_agg_fn == 'sum':
        dists = distances.sum(dim=1)
    else:
        raise NotImplementedError(f"neighbor_agg_fn {neighbor_agg_fn} not implemented")

    ## NOTE: we need .sqrt() or in general raising to some p<1, to prevent one distance from taking over the loss.
    ## This is analogous to a uniform distribution maximizing entropy.
    diversity = (-1) * dists.sqrt().mean().square()
    return diversity


## curvature

def mean_mean_curvature(F, H):
    '''
    Mean-curvature in D-dimensions
    F: grad(f) gradients, shape [N,D]
    H: hess(f) Hessian, shape [N,D,D]

    https://u.math.biu.ac.il/~katzmik/goldman05.pdf
    For a shape implicitly defined by f<0:
    - div(F/|F|) = -(FHF^T - |F|^2 tr(H)) / 2*|F|^3
    In <=3D we can expand the formula, if we want to validate https://www.archives-ouvertes.fr/hal-01486547/document
    fx, fy, fz = F.T
    fxx, fxy, fxz, fyx, fyy, fyz, fzx, fzy, fzz = H.flatten(start_dim=1).T
    k = (fx*fx*(fyy+fzz) + fy*fy*(fxx+fzz) + fz*fz*(fxx+fyy) - 2*(fx*fy*fxy+fx*fz*fxz+fy*fz*fyz)) / (2*(fx*fx+fy*fy+fz*fz).pow(3/2))
    '''
    ## Quadratic form
    FHFT = torch.einsum('bi,bij,bj->b', F, H, F)
    ## Trace of Hessian
    trH = torch.einsum('bii->b', H)
    ## Norm of gradient
    N = F.square().sum(1).sqrt()
    ## Mean-curvature
    mean_curvatures = -(FHFT - N.pow(2)*trH) / (2*N.pow(3))
    squared_mean_curvatures = mean_curvatures.square()
    return squared_mean_curvatures


def expression_curvature_loss(y_x, y_xx, expression, clip_min_value=None, clip_max_value=None, weights=None):
    H = get_mean_curvature_normalized(y_x, y_xx)
    K = get_gauss_curvature(y_x, y_xx)
    k1, k2 = get_principal_curvature(H, K) # used for the expression
    
    # We want to similar things as in Fig. 4 in https://arxiv.org/abs/2103.04856
    # But they use the non-normalized version of mean-curvature, i.e. H_unnormalized = k1 + k2
    # The expression should therefore usually write H*2 instead of H
    curvature_loss =  eval(expression)
    curvature_loss = torch.clamp(curvature_loss, min=clip_min_value, max=clip_max_value)

    if weights is not None:
        assert len(weights) == len(curvature_loss), f"weights {weights} and curvature_loss {curvature_loss} must have same length"
        # assert torch.allclose(weights.sum(), torch.tensor(1.0)), f"weights {weights} must sum to 1"
        return (curvature_loss * weights).sum(), curvature_loss.mean()
    return curvature_loss.mean()
    
def strain_curvature_loss(F, H, clip_max_value=None, weights=None):
    loss = torch.tensor(0.0)

    mean_curvatures = get_mean_curvature_normalized(F, H)
    gauss_curvatures = get_gauss_curvature(F, H)

    ## These are enough to compute the strain
    E_strain = (2*mean_curvatures)**2 - 2*gauss_curvatures
    
    E_strain = torch.clamp(E_strain, max=clip_max_value)
    # E_strain = torch.log10(E_strain + 1) 

    if weights is not None:
        assert len(weights) == len(F), f"weights {weights} and F {F} must have same length"
        assert torch.allclose(weights.sum(), torch.tensor(1.0)), f"weights {weights} must sum to 1"
        E_strain *= weights
    else:
        E_strain *= 1/E_strain.shape[0]

    loss = E_strain.sum()
    return loss

def compute_strain_curvatures(F, H, clip_max_value=None):
    mean_curvatures = get_mean_curvature_normalized(F, H)
    gauss_curvatures = get_gauss_curvature(F, H)

    ## These are enough to compute the strain
    E_strain = (2*mean_curvatures)**2 - 2*gauss_curvatures
    
    if clip_max_value is not None:
        E_strain = torch.clamp(E_strain, max=clip_max_value)
    return E_strain
    
## helpers for curvature
def get_mean_curvature_normalized(F, H):
    '''
    Mean-curvature in D-dimensions
    F: grad(f) gradients, shape [N,D]
    H: hess(f) Hessian, shape [N,D,D]

    https://u.math.biu.ac.il/~katzmik/goldman05.pdf
    For a shape implicitly defined by f<0:
    - div(F/|F|) = -(FHF^T - |F|^2 tr(H)) / 2*|F|^3
    In <=3D we can expand the formula, if we want to validate https://www.archives-ouvertes.fr/hal-01486547/document
    fx, fy, fz = F.T
    fxx, fxy, fxz, fyx, fyy, fyz, fzx, fzy, fzz = H.flatten(start_dim=1).T
    k = (fx*fx*(fyy+fzz) + fy*fy*(fxx+fzz) + fz*fz*(fxx+fyy) - 2*(fx*fy*fxy+fx*fz*fxz+fy*fz*fyz)) / (2*(fx*fx+fy*fy+fz*fz).pow(3/2))
    '''
    ## Quadratic form
    FHFT = torch.einsum('bi,bij,bj->b', F, H, F)
    ## Trace of Hessian
    trH = torch.einsum('bii->b', H)
    ## Norm of gradient
    N = F.square().sum(1).sqrt()
    N = torch.clamp(N, min=1e-5) #prevent numerical instabilities in N.pow(3) calculation
    ## Mean-curvature
    mean_curvatures = -(FHFT - N.pow(2)*trH) / (2*N.pow(3))
    return mean_curvatures

def get_gauss_curvature(F, H):
    '''
    Gauss-curvature in D-dimensions
    F: grad(f) gradients, shape [N,D]
    H: hess(f) Hessian, shape [N,D,D]
    Eq 7.2 in
    https://dsilvavinicius.github.io/differential_geometry_in_neural_implicits/assets/Differential_geometry_of_implicit_functions.pdf
    K = -1/|F|^4 det(|  H, F|
                     |F.T, 0|)
    '''
    # gauss_curvatures = (-1)/(F.square().sum(1).square())*torch.det(torch.cat([
    #     torch.cat([H, F.unsqueeze(2)], dim=2),
    #     torch.cat([F.unsqueeze(1), torch.zeros(len(F), 1,1)], dim=2) 
    #     ], dim=1))
    # return gauss_curvatures
    F4 = (F.square().sum(1).square())
    F4 = torch.clamp(F4, min=1.e-15) #prevent numerical instabilities in 1/F4 calculation
    gauss_curvatures = (-1)/F4*torch.det(torch.cat([
        torch.cat([H, F.unsqueeze(2)], dim=2),
        torch.cat([F.unsqueeze(1), torch.zeros(len(F), 1,1)], dim=2) 
        ], dim=1))
    return gauss_curvatures

def get_principal_curvature(mean_curvatures_normalized, gauss_curvatures):
    '''
    There are 2 versions for mean-curvature, 
        1) H = (k1 + k2)/2.
        2) H = k1 + k2
    We assume the first (normalized) version.
    '''
    # According to Eq. 4 in https://arxiv.org/abs/2103.04856
    # k1 = (H + sqrt(H^2 - K/4))/2
    # k2 = (H - sqrt(H^2 - K/4))/2
    # sqr = torch.sqrt(mean_curvatures.square() - gauss_curvatures / 4)
    # k1s = (mean_curvatures + sqr) / 2
    # k2s = (mean_curvatures - sqr) / 2
    
    # if mean_curvatures_are normalized
    sqr = torch.sqrt(mean_curvatures_normalized.square() - gauss_curvatures)
    k1s = mean_curvatures_normalized - sqr
    k2s = mean_curvatures_normalized + sqr
    return k1s, k2s


if __name__ == '__main__':
    
    y = torch.arange(6)