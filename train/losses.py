from math import dist
import einops
from numpy import square
import torch
import torch.nn.functional as F

def l1_loss(y_pred, y_true):
    assert y_pred.shape == y_true.shape, "avoid errors where unsqueezed dim is present"
    return F.l1_loss(y_pred, y_true)

def mse_loss(y_pred, y_true):
    assert y_pred.shape == y_true.shape, "avoid errors where unsqueezed dim is present"
    return F.mse_loss(y_pred, y_true)


def envelope_loss(ys_env, lambda_vec=None):
    '''
    If the SDF-value in the exterior of the envelope is negative, push it up. 
    '''
    loss = ys_env[ys_env<0].square().sum()
    loss_al = torch.tensor(0.0)
    if lambda_vec is not None and not torch.isnan(loss):
        loss_al = (lambda_vec * ys_env)[ys_env<0].mean()

    return torch.tensor(0.0) if torch.isnan(loss) else loss, loss_al

def obstacle_interior_loss(ys, lambda_vec=None):
    '''
    If the SDF-value in the interior of the obstacle is negative, push it up. 
    '''
    loss = ys[ys<0].square().mean()
    loss_al = torch.tensor(0.0)
    if lambda_vec is not None:
        loss = (lambda_vec * ys)[ys<0].mean()
    return torch.tensor(0.0) if torch.isnan(loss) else loss, loss_al

def interface_loss(ys, lambda_vec=None):
    '''
    The SDF-value at the boundary should be 0. This is done with mean-squared error.
    '''
    loss = ys.square().mean()
    loss_al = torch.tensor(0.0)
    if lambda_vec is not None:
        loss_al = (lambda_vec * ys).mean()
    return loss, loss_al

def normal_loss_euclidean(y_x, target_normal, lambda_vec=None, nf_is_density=False):
    """
    Prescribe normals at the interface via MSE.
    """
    assert y_x.shape == target_normal.shape, f"Shapes do not match: {y_x.shape} vs {target_normal.shape}"

    if nf_is_density:
        # normalize y_x
        y_x = F.normalize(y_x, p=2, dim=-1)
    
    loss = (y_x - target_normal).square().mean()
    loss_al = torch.tensor(0.0)
    if lambda_vec is not None:
        loss_al = (lambda_vec * (y_x - target_normal).view(-1)).mean()
    return loss, loss_al  # TODO: check scaling for dimension

def eikonal_loss(y_x, lambda_vec, **kwargs):
    '''
    Takes the gradient of points and pushes them towards unit-length
    '''
    y_x_mag = y_x.square().sum(2).sqrt()

    loss = (y_x_mag - 1).square().mean()
    loss_al = torch.tensor(0.0)
    if lambda_vec is not None:
        loss_al = (lambda_vec * (y_x_mag - 1)).mean()
    return loss, loss_al

## interpolation

def dirichlet_loss(y_z, lambda_vec=None):
    '''
    The gradient of y wrt to z is pushed towards 0.
    '''
    loss = y_z.square().mean()
    loss_al = torch.tensor(0.0)
    if lambda_vec is not None:
        loss_al = (lambda_vec * y_z).mean()
    return loss, loss_al

## curvature

def mean_mean_curvature(F, H, lambda_vec=None):
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
    mean_curvatures_al = torch.tensor(0.0)
    if lambda_vec is not None:
        mean_curvatures_al = (lambda_vec * mean_curvatures).mean()
    return squared_mean_curvatures, mean_curvatures_al


def pairwise_dist_fields(y, norm_order=2, weights=None):
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

def closest_shape_diversity_loss(y, lambda_vec=None, weights=None, norm_order=2, neighbor_agg_fn='min'):
    """
    Returns the diversity loss.
    Compute all the pairwise distances.
    For every field, take the distance to the closest neighbour.
    Exponentiate all the distances for concavity/uniformity.
    Take mean of those.
    Square to be same order as constraints.
    y: [num_fields, num_points] tensor
    """
    ## Compute pairwise distances
    distances = pairwise_dist_fields(y, norm_order=norm_order, weights=weights)
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
    loss = (-1) * dists.sqrt().mean().square()
    loss_al = torch.tensor(0.0)
    if lambda_vec is not None:
        loss_al = (-1) * (lambda_vec * dists.sqrt()).mean()
    # res.register_hook(lambda x: print(f'res closest: {x}'))
    return loss, loss

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
    return curvature_loss.mean(), curvature_loss.mean()
    
def strain_curvature_loss(F, H, lambda_vec=None, clip_max_value=None, weights=None):
    loss = torch.tensor(0.0)
    loss_al = torch.tensor(0.0)

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
    if lambda_vec is not None:
        loss_al = (lambda_vec[0:E_strain.shape[0]] * E_strain).sum()
    return loss, loss_al

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