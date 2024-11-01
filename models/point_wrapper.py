import copy

import einops
import numpy as np
import torch

from typing_extensions import Self


class PointWrapper:
    ''' A wrapper to keep a shallow copy for points belonging to different shapes. '''

    @classmethod
    def merge(cls, x1: Self, x2: Self):
        ''' Merges 2 PointWrapper to another PointWrapper '''
        assert x1.bz == x2.bz, 'PointWrapper must have same ginn_bsize'

        x3 = torch.cat([torch.zeros_like(x1.data), torch.zeros_like(x2.data)], dim=0)
        map = {}
        prev_idx = 0
        for i in range(x1.bz):
            pts_i_1 = x1.pts_of_shape(i)
            x3[prev_idx : prev_idx+len(pts_i_1)] = pts_i_1
            pts_i_2 = x2.pts_of_shape(i)
            x3[prev_idx+len(pts_i_1) : prev_idx+len(pts_i_1)+len(pts_i_2)] = pts_i_2

            map[i] = torch.arange(len(pts_i_1)+len(pts_i_2)) + prev_idx
            prev_idx += len(pts_i_1)+len(pts_i_2)

        return PointWrapper(x3, map)

    @classmethod
    def create_from_equal_bx(cls, x):
        assert len(x.shape) == 3, 'needs to be 3 dimensional'
        bz, bx, nx = x.shape
        map = {}  # track the indices of the latents
        for i in range(bz):
            map[i] = torch.arange(bx) + i * bx
        x_flat = einops.rearrange(x, 'bz bx nx -> (bz bx) nx')
        return PointWrapper(x_flat, map)

    @classmethod
    def create_from_pts_per_shape_list(cls, pts_per_shape_list):
        bz = len(pts_per_shape_list)
        map = {}  # track the indices of the latents
        i_prev = 0
        for i in range(bz):
            n_pts_for_shape_i = len(pts_per_shape_list[i])
            map[i] = torch.arange(n_pts_for_shape_i) + i_prev
            i_prev += n_pts_for_shape_i
        x_flat = torch.cat(pts_per_shape_list, dim=0)
        return PointWrapper(x_flat, map)

    def __init__(self, data: torch.Tensor, map: dict):
        '''
        :param x: [(bz bx_z) any]; note that bx_z is dependent on z
        :param map: TODO
        :param incl_mask: TODO
        '''
        self.data = data
        self._map = map  # track the indices of the latents; MUST NOT be changed in place !
        self.bz = len(self._map.keys())
        
        if len(data.shape) == 0:
            return
        
        n_points = len(data)
        # check consistency
        n_points_in_dict = 0
        for i in range(self.bz):
            n_points_in_dict += len(self._map[i])
        assert n_points == n_points_in_dict, 'all n_points must be specified in dict'

    ## get/set methods for private variables (those with _ as prefix)##

    def get_map(self):
        return self._map

    def get_idcs(self, i_shape):
        return self._map[i_shape]


    ## torch methods ##

    def cpu(self):
        if not self.data.is_cuda and not self._map[0].is_cuda:
            # x tensor is already on CPU
            return self
        new_map = {}
        for i in range(self.bz):
            new_map[i] = self._map[i].cpu()
        return PointWrapper(self.data.cpu(), new_map)
    
    def to(self, device):
        if self.data.device == device:
            # x tensor is already on CPU
            return self
        new_map = {}
        for i in range(self.bz):
            new_map[i] = self._map[i].to(device)
        return PointWrapper(self.data.to(device), new_map)

    def detach(self):
        return PointWrapper(self.data.detach(), self._map)

    def numpy(self):
        if isinstance(self.data, np.ndarray) and isinstance(self._map[0], np.ndarray):
            return self
        new_map = {}
        for i in range(self.bz):
            new_map[i] = self._map[i].numpy()
        return PointWrapper(self.data.numpy(), new_map)

        

    ## ACCESS METHODS ##

    def pts_of_shape(self, shape_idx):
        pts_of_shape_i = self.data[self._map[shape_idx]]
        return pts_of_shape_i

    def set_pts_of_shape(self, shape_idx, new_pts):
        self.data[self._map[shape_idx]] = new_pts

    def select_w_mask(self, incl_mask):
        new_x = self.data[incl_mask]
        new_map = {}
        i_prev = 0
        for i_shape in range(self.bz):
            # get the indices that belong to this shape
            idcs_i = self._map[i_shape]
            # get the mask that belong to this shape and see how many entries are selected.
            n_new_pts = incl_mask[idcs_i].sum()
            # add the new indices
            new_map[i_shape] = torch.arange(n_new_pts, device=self.data.device) + i_prev
            i_prev += n_new_pts
        return PointWrapper(new_x, new_map)
    
    def _select_w_mask(self, incl_mask):
        new_x = self.data[incl_mask]
        new_map = {}
        i_prev = 0
        for i_shape in range(self.bz):
            # get the indices that belong to this shape
            idcs_i = self._map[i_shape]
            # get the mask that belong to this shape and see how many entries are selected.
            n_new_pts = incl_mask[idcs_i].sum()
            # add the new indices
            new_map[i_shape] = torch.arange(n_new_pts, device=self.data.device) + i_prev
            i_prev += n_new_pts
        self.data = new_x
        self._map = new_map
        self.bz = len(self._map.keys()) ## TODO: is this necessary?

    def z_in(self, z):
        '''
        Prepares the z latents to be fed into the network. They have to match with the flattened points x.
        Uses self.x: [k nx] and the indices map.
        :param z: [bz nz]
        :return: z_in: [k nz]
        '''
        bz, nz = z.shape
        z_in = torch.zeros((len(self.data), nz), device=self.data.device)
        for i_shape in range(bz):
            if len(self._map[i_shape]) == 0:
                continue
            z_i = einops.repeat(z[i_shape], 'nz -> n nz', n=len(self._map[i_shape]))
            z_in[self.get_idcs(i_shape)] = z_i
        return z_in
    
    # def z_repeat_for_shape(self, z, i_shape):
    #     '''
    #     Prepares the z latents to be fed into the network. They have to match with the flattened points x.
    #     :param z: [bz nz]
    #     :param i_shape: [1] - the shape index
    #     :return: z_in: [k nz]
    #     '''
    #     shape_len = len(self._map[i_shape])
    #     z_shape = z[i_shape]
    #     z_in = einops.repeat(z_shape, 'nz -> n nz', n=shape_len)
    #     return z_in

    def __len__(self):
        return len(self.data)


class PointWrapperVec:
    ''' A vectorized version of PointWrapper. Uses a boolean mask instead of the dictionary to select the points. '''

    @classmethod
    def merge(cls, x1: Self, x2: Self):
        ''' Merges 2 PointWrapper to another PointWrapper '''
        assert x1.bz == x2.bz, 'PointWrapper must have same ginn_bsize'
        x3 = torch.cat([torch.zeros_like(x1.data), torch.zeros_like(x2.data)], dim=0)
        mask = torch.zeros((x1.bz, x1.k+x2.k), dtype=torch.bool)
        prev_idx = 0
        for i in range(x1.bz):
            pts_i_1 = x1.pts_of_shape(i)
            x3[prev_idx : prev_idx+len(pts_i_1)] = pts_i_1
            mask[i, prev_idx : prev_idx+len(pts_i_1)] = True
            pts_i_2 = x2.pts_of_shape(i)
            x3[prev_idx+len(pts_i_1) : prev_idx+len(pts_i_1)+len(pts_i_2)] = pts_i_2
            mask[i, prev_idx+len(pts_i_1) : prev_idx+len(pts_i_1)+len(pts_i_2)] = True
            prev_idx += len(pts_i_1)+len(pts_i_2)
            
        return PointWrapperVec(x3, mask)
            
    @classmethod
    def create_from_equal_bx(cls, x):
        '''
        e.g. mask = [[True, True, False], [False, False, True]]
        means that the first shape has 2 points and the second shape has 1 point.
        '''
        assert len(x.shape) == 3, 'needs to be 3 dimensional'
        bz, bx, nx = x.shape
        x_flat = einops.rearrange(x, 'bz bx nx -> (bz bx) nx')
        mask = torch.zeros((bz, bx), dtype=torch.bool)
        for i in range(bz):
            mask[i, i*bx:(i+1)*bx] = True
        return PointWrapperVec(x_flat, mask)
    
    @classmethod
    def create_from_pts_per_shape_list(cls, pts_per_shape_list):
        bz = len(pts_per_shape_list)
        mask = torch.zeros((bz, sum([len(pts) for pts in pts_per_shape_list])), dtype=torch.bool)
        i_prev = 0
        for i in range(bz):
            n_pts_for_shape_i = len(pts_per_shape_list[i])
            mask[i, i_prev:i_prev+n_pts_for_shape_i] = True
            i_prev += n_pts_for_shape_i
        x_flat = torch.cat(pts_per_shape_list, dim=0)
        return PointWrapperVec(x_flat, mask)

    def __init__(self, data: torch.Tensor, mask: torch.Tensor):
        '''
        :param x: [(bz bx_z) any]; note that bx_z is dependent on z
        :param mask: [bz bx_z]
        '''
        self.data = data
        self.mask = mask
        self.bz, self.k = mask.shape

        if len(data.shape) == 0:
            return
        
        n_points = len(data)
        # check consistency
        n_points_in_mask = mask.sum()
        assert n_points == n_points_in_mask, 'all n_points must be specified in mask'
    
    ## get/set methods for private variables (those with _ as prefix)##

    def get_map(self):
        return self.mask

    def get_idcs(self, i_shape):
        return self.mask[i_shape]
    
    ## torch methods ##
    
    def cpu(self):
        if not self.data.is_cuda:
            # x tensor is already on CPU
            return self
        return PointWrapperVec(self.data.cpu(), self.mask.cpu())
    
    def to(self, device):
        if self.data.device == device:
            # x tensor is already on CPU
            return self
        return PointWrapperVec(self.data.to(device), self.mask.to(device))

    def detach(self):
        return PointWrapperVec(self.data.detach(), self.mask.detach())
    
    def numpy(self):
        if isinstance(self.data, np.ndarray) and isinstance(self.mask, np.ndarray):
            return self
        return PointWrapperVec(self.data.numpy(), self.mask.numpy())
    
    ## ACCESS METHODS ##
    
    def pts_of_shape(self, shape_idx):
        pts_of_shape_i = self.data[self.mask[shape_idx]]
        return pts_of_shape_i
    
    def select_w_mask(self, incl_mask):
        new_x = self.data[incl_mask]
        new_mask = self.mask[:, incl_mask]
        return PointWrapperVec(new_x, new_mask)

    def z_in(self, z):
        '''
        Prepares the z latents to be fed into the network. They have to match with the flattened points x.
        Uses self.x: [k nx] and the indices map.
        :param z: [bz nz]
        :return: z_in: [k nz]
        '''
        bz, nz = z.shape
        z_in = torch.zeros((len(self.data), nz), device=self.data.device)
        for i_shape in range(bz):
            n_pts_in_shape_i = self.mask[i_shape].sum()
            if n_pts_in_shape_i == 0:
                continue
            z_i = einops.repeat(z[i_shape], 'nz -> n nz', n=n_pts_in_shape_i)
            z_in[self.get_idcs(i_shape)] = z_i
        return z_in
    

if __name__ == '__main__':
    shape_list = [
            torch.tensor([[1., 2.]]),
            torch.tensor([
                [3., 4.],
                [5., 6.],
            ]),
            ]

    z = torch.tensor([
            [7., 8.],
            [9., 10.],
        ])

    res = torch.tensor([
            [7., 8.],
            [9., 10.],
            [9., 10.],
        ])

    p = PointWrapper.create_from_pts_per_shape_list(shape_list)
    z_in = p.z_in(z)
    print(z_in)
    assert len(z_in.shape) == 2
    assert torch.equal(z_in, res)
