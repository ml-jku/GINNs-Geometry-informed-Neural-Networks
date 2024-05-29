import einops
from numpy import square
import torch
import torch.nn.functional as F

def envelope_loss(ys_env):
    '''
    If the SDF-value in the exterior of the envelope is negative, push it up. 
    '''
    loss = ys_env[ys_env<0].square().mean()
    return torch.tensor(0.0) if torch.isnan(loss) else loss

def obstacle_interior_loss(ys):
    '''
    If the SDF-value in the interior of the obstacle is negative, push it up. 
    '''
    loss = ys[ys<0].square().mean()
    return torch.tensor(0.0) if torch.isnan(loss) else loss

def interface_loss(ys):
    '''
    The SDF-value at the boundary should be 0. This is done with mean-squared error.
    '''
    return ys.square().mean()

def normal_loss_euclidean(y_x, target_normal):
    """
    Prescribe normals at the interface via MSE.
    """
    assert y_x.shape == target_normal.shape, f"Shapes do not match: {y_x.shape} vs {target_normal.shape}"
    return (y_x - target_normal).square().mean()  # TODO: check scaling for dimension
    

def normal_loss_dot(y_x, target_normal):
    """
    Prescribe normals at the interface via Cosine-Similarity.
    """
    y_x_norm = F.normalize(y_x)
    return -1 * torch.einsum('n d, d -> n', y_x_norm, target_normal).mean()  # TODO: check scaling for dimension


def eikonal_loss(y_x):
    '''
    Takes the gradient of points and pushes them towards unit-length
    '''
    y_x_mag = y_x.square().sum(2).sqrt()
    return (y_x_mag - 1).square().mean()


def mean_squared_mean_curvature(F, H):
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
    return squared_mean_curvatures.mean()


def pairwise_dist_fields(y, weights=None):
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
    
    diff = y.unsqueeze(0) - y.unsqueeze(1)
    weighted_diff = diff if weights is None else diff * weights.sqrt()
    ## the norm-function is stable, but the square().mean().sqrt() is not
    pairwise_dist = weighted_diff.norm(dim=-1)
    # the norm takes the sum over all points, so we need to divide by the sqrt of the number of points
    res = pairwise_dist / (n_points**0.5)
    ## for debugging only
    # res.register_hook(lambda x: print(f'res pairwise: {x}'))
    return res 

def closest_shape_diversity_loss(y, weights=None):
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
    distances = pairwise_dist_fields(y, weights=weights)
    ## Create a mask to exclude diagonal elements
    distances = distances.masked_fill(torch.eye(distances.size(0), dtype=torch.bool), float('inf'))
    ## Find the minimum distance for each point
    closest_dists, closest_inds = distances.min(dim=1)

    ## NOTE: we need .sqrt() or in general raising to some p<1, to prevent one distance from taking over the loss.
    ## This is analogous to a uniform distribution maximizing entropy.
    res = (-1) * closest_dists.sqrt().mean().square()
    # res.register_hook(lambda x: print(f'res closest: {x}'))
    return res
    
def strain_curvature_loss(F, H, clip_max_value=None, weights=None):
    mean_curvatures = get_mean_curvature(F, H)
    gauss_curvatures = get_gauss_curvature(F, H)

    ## These are enough to compute the strain
    E_strain = (2*mean_curvatures)**2 - 2*gauss_curvatures
    
    if clip_max_value is not None:
        E_strain = torch.clamp(E_strain, max=clip_max_value)
    
    if weights is not None:
        assert len(weights) == len(F), f"weights {weights} and F {F} must have same length"
        assert torch.allclose(weights.sum(), torch.tensor(1.0)), f"weights {weights} must sum to 1"
        return (E_strain * weights).sum()
    return E_strain.mean()

def compute_strain_curvatures(F, H, clip_max_value=None):
    mean_curvatures = get_mean_curvature(F, H)
    gauss_curvatures = get_gauss_curvature(F, H)

    ## These are enough to compute the strain
    E_strain = (2*mean_curvatures)**2 - 2*gauss_curvatures
    
    if clip_max_value is not None:
        E_strain = torch.clamp(E_strain, max=clip_max_value)
    
    return E_strain
    
## helpers for curvature
def get_mean_curvature(F, H):
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
    gauss_curvatures = (-1)/(F.square().sum(1).square())*torch.det(torch.cat([
        torch.cat([H, F.unsqueeze(2)], dim=2),
        torch.cat([F.unsqueeze(1), torch.zeros(len(F), 1,1)], dim=2) 
        ], dim=1))
    return gauss_curvatures

def get_principal_curvature(mean_curvatures, gauss_curvatures):
    sqr = torch.sqrt(mean_curvatures.square() - gauss_curvatures)
    k1s = mean_curvatures - sqr
    k2s = mean_curvatures + sqr
    return k1s, k2s


if __name__ == '__main__':
    
    y = torch.arange(6)