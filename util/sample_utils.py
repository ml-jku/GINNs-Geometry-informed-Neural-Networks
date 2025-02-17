import torch
import math

def inflate_bounds(bounds, amount=0.10):
    """
    Inflate a bounding box by the specified fraction of the extent on each side.
    To finding good surface points, the bounding box should not be tight.
    This is because
        (A) we want to sample initial points around the surface which cannot be fulfilled if the surface is near bbox,
        (B) point trajectories may overshoot the surface and leave the bbox getting filtered.
    This is not required if the bbox is not tight.
    """
    lengths = bounds[:,1] - bounds[:,0]
    bounds_ = bounds.clone()
    bounds_[:,0] -= lengths*amount
    bounds_[:,1] += lengths*amount
    return bounds_


def precompute_sample_grid(n_points: int, bounds: torch.Tensor, equidistant: bool):
        '''
        An grid of points is computed. 
        If equidistant=True, the grid is equidistant in each dimension.
        Else the number of total points defined in n_points is equally distributed among all dimensions.
        :return:
        xc_grid: the 2d or 3d grid of equidistant points over the domain
        xc_grid_dist: the distance as a 2d or 3d vector neighbors along the respective dimension
        '''
        nx = bounds.shape[0]
        bound_widths = bounds[:, 1] - bounds[:, 0]

        if equidistant:
            # Derivation:
            # n_points = (n_points_x * n_points_y * n_points_z)
            # n_points_x = n_points_stem * bound_widths[0]
            # n_points_y = n_points_stem * bound_widths[1]
            # n_points_z = n_points_stem * bound_widths[2]
            # n_points = n_points_stem ** nx * torch.prod(bound_widths)
            # -> n_points_stem = torch.pow(n_points / (bound_widths[0] * bound_widths[1] * bound_widths[2]), 1/nx)
            n_points_stem = int(math.pow(n_points / torch.prod(bound_widths), 1 / nx))
            resolution = (n_points_stem * bound_widths).to(torch.int)
        else:
            n_points_root = int(n_points ** (1 / nx))
            resolution = torch.tensor([n_points_root] * nx)

        if nx == 2:
            x1_grid = torch.linspace(bounds[0, 0], bounds[0, 1], resolution[0])
            x2_grid = torch.linspace(bounds[1, 0], bounds[1, 1], resolution[1])
            x1_grid, x2_grid = torch.meshgrid(x1_grid, x2_grid)
            x_grid = torch.stack((x1_grid.reshape(-1), x2_grid.reshape(-1)), dim=1)
        elif nx == 3:
            x1_grid = torch.linspace(bounds[0, 0], bounds[0, 1], resolution[0])
            x2_grid = torch.linspace(bounds[1, 0], bounds[1, 1], resolution[1])
            x3_grid = torch.linspace(bounds[2, 0], bounds[2, 1], resolution[2])
            x1_grid, x2_grid, x3_grid = torch.meshgrid(x1_grid, x2_grid, x3_grid)
            x_grid = torch.stack((x1_grid.reshape(-1), x2_grid.reshape(-1), x3_grid.reshape(-1)), dim=1)

            
            # dist_along_a_dim = 1 / (n_points_root + 1)
            # xi_range = torch.arange(start=0, end=n_points_root, step=1) * dist_along_a_dim + dist_along_a_dim / 2
            # if nx == 2:
            #     x1_grid, x2_grid = torch.meshgrid(xi_range, xi_range, indexing="ij")
            #     x_grid = torch.stack((x1_grid.reshape(-1), x2_grid.reshape(-1)), dim=1)
            # elif nx == 3:
            #     x1_grid, x2_grid, x3_grid = torch.meshgrid(xi_range, xi_range, xi_range, indexing="ij")
            #     x_grid = torch.stack((x1_grid.reshape(-1), x2_grid.reshape(-1), x3_grid.reshape(-1)), dim=1)
        
        # x_grid = bounds[:, 0] + (bounds[:, -1] - bounds[:, 0]) * x_grid
        xyz_grid_dist = bound_widths / resolution
        return x_grid, xyz_grid_dist, resolution