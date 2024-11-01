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


def precompute_sample_grid(n_points, bounds):
        '''
        An equidistant grid of points is computed. These are later taken as starting points to discover critical points
        via gradient descent. The number of total points defined in config['n_points_find_cps'] is equally distributed
        among all dimensions.
        :return:
        xc_grid: the 2d or 3d grid of equidistant points over the domain
        xc_grid_dist: the distance as a 2d or 3d vector neighbors along the respective dimension
        '''
        nx = bounds.shape[0]

        n_points_root = int(math.floor(math.pow(n_points, 1 / nx)))
        dist_along_a_dim = 1 / (n_points_root + 1)
        xi_range = torch.arange(start=0, end=n_points_root, step=1) * dist_along_a_dim + dist_along_a_dim / 2
        if nx == 2:
            x1_grid, x2_grid = torch.meshgrid(xi_range, xi_range, indexing="ij")
            xc_grid = torch.stack((x1_grid.reshape(-1), x2_grid.reshape(-1)), dim=1)
        elif nx == 3:
            x1_grid, x2_grid, x3_grid = torch.meshgrid(xi_range, xi_range, xi_range, indexing="ij")
            xc_grid = torch.stack((x1_grid.reshape(-1), x2_grid.reshape(-1), x3_grid.reshape(-1)), dim=1)
        xc_grid = bounds[:, 0] + (bounds[:, -1] - bounds[:, 0]) * xc_grid
        xc_grid_dist = torch.tensor(dist_along_a_dim).repeat(nx) * (bounds[:, -1] - bounds[:, 0])
        return xc_grid.to(torch.float32), xc_grid_dist.to(torch.float32)