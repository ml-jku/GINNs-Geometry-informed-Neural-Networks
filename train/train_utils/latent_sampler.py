import math
import re
from typing import Tuple
import einops
import torch
from einops import repeat


'''
Sampling protocol:
- sample_z are latents for GINN without data
    - excludes the corners of the interval, to avoid training on data AND GINN
- get_z_corners are the latents for the provided data

'''


def _linspace_excluding_ends(start, end, steps):
    # Generate a tensor with steps + 2 values to include both start and end
    full_tensor = torch.linspace(start, end, steps + 2)
    # Exclude the first and last elements
    return full_tensor[1:-1]

class LatentSampler:

    def __init__(self, 
               nz, 
               ginn_bsize,
               z_sample_method,
               z_sample_interval,
               val_plot_grid):
        assert nz > 0, 'nz must be > 0'
        self.nz = nz
        self.batch_size = ginn_bsize
        self.z_sample_method = z_sample_method
        self.interv = z_sample_interval
        self.val_plot_grid = val_plot_grid

    def train_z(self):
        '''Returns z: [num_latents, nz]'''    
        ## get training z
        if self.z_sample_method == 'equidistant':
            '''
            for training, exclude the corners of the interval
            for validation, include them
            e.g. if nz=1, ginn_bsize=4, z_sample_interval=[-1, 1]
                training z is [-1, -0.33, 0.33, 1]
                validation z are the middle points [-0.66, 0, 0.66]
            '''
            assert self.nz <= 2, 'z_sample_method equidistant only supported for nz <= 2'
            
            if self.nz == 1:
                z = _linspace_excluding_ends(self.interv[0], self.interv[1], self.batch_size)[:, None]
                return z
            elif self.nz == 2:
                n_rows = int(math.sqrt(self.batch_size))
                assert n_rows * n_rows == self.batch_size, 'ginn_bsize must be a square number'
                z = _linspace_excluding_ends(self.interv[0], self.interv[1], n_rows)
                z = torch.cartesian_prod(z, z)
                return z
                

        if self.z_sample_method == 'uniform':
            z = torch.rand(size=(self.batch_size, self.nz)) * (self.interv[1] - self.interv[0]) + self.interv[0]
            return z
            
        if self.z_sample_method == 'normal':
            raise NotImplementedError('z_sample_method normal not implemented')
            z = torch.randn(size=(self.batch_size, self.nz))
            mu, var = self.z_mean_var[0], self.z_mean_var[1]
            z = z * math.sqrt(var) + mu
            return z
        
        raise ValueError(f'z_sample_method {self.z_sample_method} not supported')

    def val_z(self):
        if self.nz == 1:
            n_shapes = self.val_plot_grid[0] * self.val_plot_grid[1]
            z = torch.linspace(self.interv[0], self.interv[1], n_shapes)[:, None]
        elif self.nz == 2:
            # plot as grid, eg 2x3
            # returns e.g.
            # [[[0.0000, 0.0000],
            #  [0.0000, 0.5000],
            #  [0.0000, 1.0000]],
            # [[1.0000, 0.0000],
            #  [1.0000, 0.5000],
            #  [1.0000, 1.0000]]]
            z_rows = torch.linspace(self.interv[0], self.interv[1], self.val_plot_grid[0])
            z_cols = torch.linspace(self.interv[0], self.interv[1], self.val_plot_grid[1])
            z = torch.cartesian_prod(z_rows, z_cols)
        else:
            # for nz>2 the zlatents are not easily displayed. Therefore we plot interpolations along the first n dimensions with subsampling
            # valid_plot_n_shapes_grid: [4, 5] # [n_first_dims, shapes_per_dim]
            '''
            Ideally all points would have the same pair-wise distance if they were a simplex.
            As this is impractical to compute, we place the anchors on the corners of a n-cube.
            E.g. for nz=8, valid_plot_n_shapes_grid=[4, 5]
            The first number tells how many dimensions to iterate over, the second how many shapes per dimension.
            z10 = [0, 0, 0, 0, 0, 0, 0, 0]
            z11 = [0.25, 0, 0, 0, 0, 0, 0, 0]
            z12 = [0.5, 0, 0, 0, 0, 0, 0, 0]
            z13 = [0.75, 0, 0, 0, 0, 0, 0, 0]
            z14 = [1, 0, 0, 0, 0, 0, 0, 0]
            z20 = [0, 0, 0, 0, 0, 0, 0, 0]
            z21 = [0.25, 0.25, 0, 0, 0, 0, 0, 0]
            z22 = [0.5, 0.5, 0, 0, 0, 0, 0, 0]
            z23 = [0.75, 0.75, 0, 0, 0, 0, 0, 0]
            z24 = [1, 1, 0, 0, 0, 0, 0, 0]
            z30 = [0, 0, 0, 0, 0, 0, 0, 0]
            z31 = [0.25, 0.25, 0.25, 0, 0, 0, 0, 0]
            z32 = [0.5, 0.5, 0.5, 0, 0, 0, 0, 0]
            z33 = [0.75, 0.75, 0.75, 0, 0, 0, 0, 0]
            z34 = [1, 1, 1, 0, 0, 0, 0, 0]
            z40 = [0, 0, 0, 0, 0, 0, 0, 0]
            etc.
            '''
            
            n_first_dims, shapes_per_dim = self.val_plot_grid
            assert self.nz >= n_first_dims / shapes_per_dim, "Plot_grid could be chosen smaller"
            total_shapes = n_first_dims * shapes_per_dim
            z = torch.zeros((total_shapes, self.nz))
            z_for_shapes = torch.linspace(self.interv[0], self.interv[1], shapes_per_dim)
            for i in range(total_shapes):
                i_row = i % shapes_per_dim
                i_col = i // shapes_per_dim + 1
                z[i, :i_col] = z_for_shapes[i_row]
        return z
    
    def get_z_corners(self, n_corners):        
        z_corners = torch.tensor(self.interv)[:n_corners]
        z_corners = repeat(z_corners, 'n -> n d', d=self.nz)
        return z_corners

    def combine_z_with_z_corners(self, z, z_corners, n_train_shapes_to_plot):
        z_plot = z[:n_train_shapes_to_plot]
        if len(z_corners) > 0:
            # sub-sample z
            z_sub = z[:n_train_shapes_to_plot - len(z_corners)]
            z_plot = torch.cat([z_corners[:1], z_sub, z_corners[1:]], dim=0) # works if there are 1 or 2 data shapes in the corners
        return z_plot

# helper for visualization
def sample_3d_z_grid(interval, grid: Tuple):
    '''
    IMPORTANT: The first dimension of the grid is the number of frames in the GIF.
    To create a GIF like a Daumenkino, we sample 2D slices from a 3D grid.
    returns a 3d tensor of shape as given by the grid.
    '''
    assert len(grid) == 3, "Only done for 3D grids"
    z_rows = torch.linspace(interval[0], interval[1], grid[0])
    z_cols = torch.linspace(interval[0], interval[1], grid[1])
    z_diag = torch.linspace(interval[0], interval[1], grid[2])
    z = torch.cartesian_prod(z_rows, z_cols, z_diag)
    z = einops.rearrange(z, '(a b) d -> a b d', a=grid[0])
    return z