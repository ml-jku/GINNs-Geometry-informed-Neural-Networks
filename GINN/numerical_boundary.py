import math
import einops
import torch

from models.net_w_partials import NetWithPartials
from models.point_wrapper import PointWrapper


def get_grid_starting_pts(n_shapes, x_grid, grid_dist):
    '''
    Create grid once at the beginning.
    Translate the grid by a random offset.
    '''
    nx = x_grid.shape[1]
    
    ## Translate the grid by a random offset
    xc_offset = torch.rand((n_shapes, nx)) * grid_dist  # bz nx

    # x_grid: [n_points nx]
    x = x_grid.unsqueeze(0) + xc_offset.unsqueeze(1)  # bz n_points nx

    ## Translate each point by a random offset
    x += torch.randn(x_grid.shape) * grid_dist / 3

    return PointWrapper.create_from_equal_bx(x)
    

def find_boundary_points_numerically_with_binsearch(x_grid, x_grid_dist, level_set, netp: NetWithPartials, z, n_steps=10, resolution=None, nf_is_density=False):
    '''
    Find the boundary points numerically.
    First find points inside the boundary where neighboring points are on opposite sides of the level set.
    Then use binary search to refine the boundary points.
    '''
    nx = len(x_grid_dist)
    
    def fwd_and_standardize(x, z):
        y = netp.grouped_no_grad_fwd('vf', x, z).squeeze(1)
        y = y - level_set
        if nf_is_density == False:
            y = -1 * y
        return y
    
    p_grid = get_grid_starting_pts(len(z), x_grid, x_grid_dist)
    y_grid = fwd_and_standardize(p_grid.data, p_grid.z_in(z))

    if resolution is None:
        # Infer resolution for a uniform grid
        n_points_root = int(round(math.pow(x_grid.shape[0], 1. / nx)))
        assert n_points_root**nx == x_grid.shape[0], "y_grid shape does not match nx dimensions."
        resolution = [n_points_root] * nx

    # Reshape the indices and grid
    y_grid_reshaped = einops.rearrange(y_grid, 
                                       f'(b {" ".join([f"x{i}" for i in range(nx)])}) -> b {" ".join([f"x{i}" for i in range(nx)])}', 
                                       b=len(z), **{f'x{i}': resolution[i] for i in range(nx)})

    # Define masks for points above and below the level set
    above_mask = y_grid_reshaped > 0
    below_mask = y_grid_reshaped < 0
    inside_boundary_mask = torch.zeros_like(above_mask, dtype=torch.bool)
    
    # Define shifts
    shifts = [tuple([0] + [1 if i == d else 0 for i in range(nx)]) for d in range(nx)] + \
             [tuple([0] + [-1 if i == d else 0 for i in range(nx)]) for d in range(nx)]
    
    # create valid mask for boundary points of the domain 
    valid_mask = torch.ones_like(y_grid_reshaped, dtype=torch.bool)
    for d in range(1, valid_mask.dim(), 1):
        # Create a slice for the current dimension
        slice_indices = [slice(None)] * valid_mask.dim()
        slice_indices[d] = 0
        valid_mask[tuple(slice_indices)] = False

        slice_indices[d] = -1
        valid_mask[tuple(slice_indices)] = False

    for shift in shifts:
        # for density, inside points are above the level set
        shifted_below_level_set = torch.roll(below_mask, shifts=shift, dims=tuple(range(nx+1)))
        inside_boundary_mask = inside_boundary_mask | (above_mask & shifted_below_level_set & valid_mask)     

    if not inside_boundary_mask.any():
        print("No points found below the level set.")
        return False, (None, None)

    # Points inside the boundary mask
    p_higher = p_grid.select_w_mask(inside_boundary_mask.flatten())
    
    # Initialize outside points
    x_lower = torch.zeros_like(p_higher.data)
    z_in = p_higher.z_in(z)
    # Perturbation to find initial outside points
    for d in range(nx):
        perturb = torch.zeros_like(x_lower)
        perturb[:, d] = x_grid_dist[d] * 1.5
        outside_perturb_plus = p_higher.data + perturb
        outside_perturb_minus = p_higher.data - perturb

        # Evaluate level set to find which perturbations are outside
        level_plus = fwd_and_standardize(outside_perturb_plus, z_in)
        level_minus = fwd_and_standardize(outside_perturb_minus, z_in)

        # Update higher points
        plus_mask = level_plus < 0
        x_lower[plus_mask] = outside_perturb_plus[plus_mask]
        minus_mask = level_minus < 0
        x_lower[minus_mask] = outside_perturb_minus[minus_mask]

    p_lower = PointWrapper(x_lower, p_higher._map)

    # Binary search refinement
    high = p_higher.data
    low = p_lower.data
    z_in = p_higher.z_in(z)
    for i in range(n_steps):
        mid = (low + high) / 2
        level = fwd_and_standardize(mid, z_in)
        # print(f'i: {i} level.mean(): {level.mean()}, level.std(): {level.std()}')
        low = torch.where((level < 0).unsqueeze(1), mid, low)
        high = torch.where((level > 0).unsqueeze(1), mid, high)
        
        ## could optimize for the case where level == 0, but takes some compute and is not necessary
        # low = torch.where((level == 0).unsqueeze(1), mid, low)
        # high = torch.where((level == 0).unsqueeze(1), mid, high)
        
        ## for debugging
        # y_low = grouped_fwd(low, z_in)
        # y_high = grouped_fwd(high, z_in)
        # assert (y_low <= 0).all()
        # assert (y_high >= 0).all()

    p_res = PointWrapper(data=(low + high) / 2, map=p_higher._map)
    y_sel = fwd_and_standardize(p_res.data, p_res.z_in(z))
    # print(f'final level.mean(): {y_sel.mean()}, level.std(): {y_sel.std()}')
    return True, (p_res, y_sel)
