import torch

from models.point_wrapper import PointWrapper


def update_proximity_mask(pc, x_path_over_iters, squared_norms, eps):
    '''
    The mask is True if the point is the one with the smallest gradient norm in its proximity.
    Attention: The mask is only transitive for points with small gradient norm.
    E.g. if point A is close to B and B is close to C, then A is close to C only if the gradient norms
    are either increasing or decreasing for the sequence [A, B, C]. 
    If increasing then A is the only point chosen and the mask is [True, False, False].
    If decreasing then C is the only point chosen and the mask is [False, False, True].
    If grad(A) < grad(B) > grad(C) then the mask is [True, False, True].
    If grad(A) > grad(B) < grad(C) then the mask is [False, True, False].
    pc: PointWrapper
    x_path_over_iters: [n_iters, k, nx]
    squared_norms: [k]
    eps: float
    return: proximity_mask: [k]
    '''
    assert len(x_path_over_iters) > 0
    assert len(squared_norms) == len(pc)
    assert x_path_over_iters.shape[1] == len(pc)
    # assert len(stop_mask) == len(pc)
    
    new_proximity_mask = torch.ones(len(pc), dtype=torch.bool, device=x_path_over_iters.device)
    for i_shape in range(pc.bz):
        # get the indices that belong to this shape
        idcs_i = pc._map[i_shape]
        x_i = pc.pts_of_shape(i_shape)
        x_path_i = x_path_over_iters[:, idcs_i, :]
        squared_norms_i = squared_norms[idcs_i]
        # stop_mask_i = stop_mask[idcs_i]
        
        if len(x_i) == 0:
            new_proximity_mask[idcs_i] = False
            continue
        
        assert torch.equal(x_i, x_path_i[-1, :, :])
        
        # get permutation and its inverse that sorts the points by gradient norm
        perm = torch.argsort(squared_norms_i, descending=False)
        perm_inv = torch.argsort(perm)
        
        # sort the points by gradient norm
        x_i_perm = x_i[perm, :]  # [n_points_i, nx]
        x_path_i = x_path_i[:, perm, :]  # [n_iters, n_points_i, nx]
        
        # print('=======================')
        # print('x_i_perm:', x_i_perm)
        # print('x_path_i:', x_path_i)
        
        ## get the squared distances between all current points and the paths
        dists = torch.sum((x_i_perm[None, :, None, :] - x_path_i[:, None, :, :])**2, dim=-1)  # [n_iters, n_points_i, n_points_i]
        have_prox_point_on_path = dists < eps**2  # [n_iters, n_points_i, n_points_i]
        have_prox_point = have_prox_point_on_path.any(dim=0)  # [n_points_i, n_points_i]
        
        assert have_prox_point.sum() >= len(x_i)  # at least each point is close to itself
        
        # print('distances:', dists)
        # print('have_prox_point_on_path:', have_prox_point_on_path)
        # print('have_prox_point:', have_prox_point)
                
        ## in each row get the first index that is True
        have_prox_point = have_prox_point.int()  ## convert to int to use argmax
        first_prox_point = have_prox_point.argmax(dim=0)  # [n_points_i]
        
        # print('have_prox_point:', have_prox_point)
        # print('first_prox_point:', first_prox_point)

        ## get the mask by checking which values are equal to the row index (i.e. the first True value in each row)
        new_prox_mask_i = first_prox_point == torch.arange(len(x_i), device=x_path_over_iters.device)
        # print('new_prox_mask_i:', new_prox_mask_i)
        ## unsort the mask    
        new_prox_mask_inv_i = new_prox_mask_i[perm_inv]
        # print('new_prox_mask_inv_i:', new_prox_mask_inv_i)
        new_proximity_mask[idcs_i] = new_prox_mask_inv_i
    
    return new_proximity_mask

def compute_convergence_to_cp_metric(pc_C: PointWrapper, indexes_C,
                                     pc_T: PointWrapper, indexes_T, eps):

    res_dict = {}

    p_C_and_T = PointWrapper.merge(pc_C, pc_T)
    
    # r_pts_covered_by_C = n_common_clusters(p_C_and_T, pc_C, eps)
    # r_pts_covered_by_T = n_common_clusters(p_C_and_T, pc_T, eps)
    res_dict['r_pts_covered_by_C'] = n_common_clusters(p_C_and_T, pc_C, eps)
    res_dict['r_pts_covered_by_T'] = n_common_clusters(p_C_and_T, pc_T, eps)
    
    pc_C_saddles = pc_C.select_w_mask(indexes_C == 1)
    pc_C_mins = pc_C.select_w_mask(indexes_C == 0)
    pc_C_maxs = pc_C.select_w_mask(indexes_C == 2)
    
    pc_T_saddles = pc_T.select_w_mask(indexes_T == 1)
    pc_T_mins = pc_T.select_w_mask(indexes_T == 0)
    pc_T_maxs = pc_T.select_w_mask(indexes_T == 2)
    
    pc_C_and_T_saddles = PointWrapper.merge(pc_C_saddles, pc_T_saddles)
    pc_C_and_T_mins = PointWrapper.merge(pc_C_mins, pc_T_mins)
    pc_C_and_T_maxs = PointWrapper.merge(pc_C_maxs, pc_T_maxs)
    
    res_dict['r_saddles_covered_by_C'] = n_common_clusters(pc_C_and_T_saddles, pc_C_saddles, eps)
    res_dict['r_saddles_covered_by_T'] = n_common_clusters(pc_C_and_T_saddles, pc_T_saddles, eps)
    res_dict['r_mins_covered_by_C'] = n_common_clusters(pc_C_and_T_mins, pc_C_mins, eps)
    res_dict['r_mins_covered_by_T'] = n_common_clusters(pc_C_and_T_mins, pc_T_mins, eps)
    res_dict['r_maxs_covered_by_C'] = n_common_clusters(pc_C_and_T_maxs, pc_C_maxs, eps)
    res_dict['r_maxs_covered_by_T'] = n_common_clusters(pc_C_and_T_maxs, pc_T_maxs, eps)
    
    return res_dict

def n_common_clusters(pc_A, pc_B, eps):
    
    if len(pc_A) == 0 or len(pc_B) == 0:
        return 0.0
    
    n_pts_of_A_covered_by_B = 0
    
    for i_shape in range(pc_A.bz):
        x_A = pc_A.pts_of_shape(i_shape)
        x_B = pc_B.pts_of_shape(i_shape)
        
        if len(x_A) == 0 or len(x_B) == 0:
            continue

        ## get the squared distances between the points
        dists = torch.sum((x_A[:, None, :] - x_B[None, :, :])**2, dim=-1).sqrt()  # [n_points_A, n_points_B]
        in_proximity = dists < eps
        
        # allow only one point of B to cover a point of A
        in_proximity = in_proximity.int()
        in_proximity = in_proximity.sum(dim=1)
        in_proximity = in_proximity > 0
        n_pts_of_A_covered_by_B += in_proximity.sum()
        
    return n_pts_of_A_covered_by_B / len(pc_A)
        