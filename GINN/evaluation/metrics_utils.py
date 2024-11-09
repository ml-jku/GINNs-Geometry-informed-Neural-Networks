import numpy as np

agg_fns = {
            'sum': np.sum,
            'mean': np.mean,
            'min': np.min
        }

def chamfer_divergence(x, y, dist_norm_order=2):
    '''
    Chamfer discrepancy is not a distance metric. 
    It is the sum of the average closest point distances between two point sets.
    Formula here: https://openaccess.thecvf.com/content/ICCV2021/papers/Nguyen_Point-Set_Distances_for_Learning_Representations_of_3D_Point_Clouds_ICCV_2021_paper.pdf
    :param x: [n_points, n_dims]
    :param y: [n_points, n_dims]
    '''
    pt_dist_pairs = np.linalg.norm(x[:, None, :] - y[None, :, :], ord=dist_norm_order, axis=-1)
    pt_dist_pairs = pt_dist_pairs ** 2  # square to get squared distances
    closest_pt_dists_x = pt_dist_pairs.min(axis=1)
    closest_pt_dists_y = pt_dist_pairs.min(axis=0)
    classical_chamfer = closest_pt_dists_x.mean() + closest_pt_dists_y.mean()
    return classical_chamfer ** 0.5  # square root to make it more interpretable

def diversity_metric(obj_list, dist_func, p, inner_agg_fn='sum', outer_agg_fn='sum'):
    '''
    Compute the diversity metric based on distances between the objects.
    '''
    n_shapes = len(obj_list)
    chamfer_mat = np.zeros((n_shapes, n_shapes))
    
    for i in range(n_shapes):
        for j in range(i+1, n_shapes):
            chamfer_mat[i, j] = dist_func(obj_list[i], obj_list[j])
            chamfer_mat[j, i] = chamfer_mat[i, j]
    
    if inner_agg_fn == 'min':
        ## set diagonal to infinity
        np.fill_diagonal(chamfer_mat, np.inf)
    
    row_aggregated = agg_fns[inner_agg_fn](chamfer_mat, axis=1)  # [n_shapes]
    if inner_agg_fn == 'mean':
        ## account for the fact that we have n_shapes-1 distances
        row_aggregated *= n_shapes / (n_shapes - 1)
    row_aggregated = row_aggregated ** p
    col_aggregated = agg_fns[outer_agg_fn](row_aggregated)  # []
    col_aggregated = col_aggregated ** (1/p)
    return col_aggregated

def chamfer_diversity_metric_vec(pts_mat, p, row_agg_fn='sum', col_agg_fn='sum', norm_order=2):
    '''
    Compute the diversity metric based on distances between the objects.
    :param pts_mat: [n_shapes, n_points, n_dims]
    '''
    assert row_agg_fn in agg_fns.keys(), f'row_agg_fn must be one of {agg_fns.keys()}'
    assert col_agg_fn in agg_fns.keys(), f'col_agg_fn must be one of {agg_fns.keys()}'
    
    n_shapes, n_points, n_dims = pts_mat.shape
    
    dist_mat = np.linalg.norm(pts_mat[:, None, :, None, :] - pts_mat[None, :, None, :, :], ord=norm_order, axis=-1)  # [n_shapes, n_shapes, n_points]
    dist_mat = dist_mat ** 2  # square to get squared distances
    min_distances = np.min(dist_mat, axis=-1)  # [n_shapes, n_shapes, n_points]
    chamfer_mat = min_distances.mean(axis=-1)  # [n_shapes, n_shapes]
    
    chamfer_mat = chamfer_mat ** 0.5  # [n_shapes, n_shapes]; square root to make it more interpretable
    
    if row_agg_fn == 'min':
        ## set diagonal to infinity
        np.fill_diagonal(chamfer_mat, np.inf)
    
    row_aggregated = agg_fns[row_agg_fn](chamfer_mat, axis=1)  # [n_shapes]
    if row_agg_fn == 'mean':
        ## account for the fact that we have n_shapes-1 distances
        row_aggregated *= n_shapes / (n_shapes - 1)
    row_aggregated = row_aggregated ** p
    col_aggregated = agg_fns[col_agg_fn](row_aggregated)  # []
    if col_agg_fn == 'mean':
        ## account for the fact that we have n_shapes-1 distances
        col_aggregated *= n_shapes / (n_shapes - 1)
    col_aggregated = col_aggregated ** (1/p)
    return col_aggregated

def diversity_metric_multi_agg(obj_list, dist_func, p_list, inner_agg_fns=['sum'], outer_agg_fns=['sum']):
    '''
    Compute the diversity metric based on distances between the objects.
    '''
    outer_aggs = {}
    for i in range(len(obj_list)):
        
        ## do inner aggregation
        inner_aggs = []
        for j in range(len(obj_list)):
            inner_aggs.append(dist_func(obj_list[i], obj_list[j]))
        
        inner_agg_dict = {}
        for p in p_list:
            for inner_agg_fn in inner_agg_fns:
                inner_agg_dict[f'p_{p}-inner_agg_{inner_agg_fn}'] = agg_fns[inner_agg_fn](inner_aggs) ** p
                
        outer_aggs[i] = inner_agg_dict
    
    out_agg_dict = {}
    for p in p_list:
        for outer_agg_fn in outer_agg_fns:
            for inner_agg_fn in inner_agg_fns:
                out_agg_dict[f'p_{p}-inner_agg_{inner_agg_fn}-outer_agg_{outer_agg_fn}'] = agg_fns[outer_agg_fn]([outer_aggs[i][f'p_{p}-inner_agg_{inner_agg_fn}'] for i in range(len(obj_list))]) ** (1/p)
                
    return out_agg_dict
                    
       