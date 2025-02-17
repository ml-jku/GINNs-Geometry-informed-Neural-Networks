import igl
import numpy as np
import torch

agg_fns = {
            'sum': np.sum,
            'mean': np.mean,
            'min': np.min
        }

def one_sided_chamfer_divergence(x, y, dist_norm_order=2):
    '''
    Chamfer discrepancy is not a distance metric. 
    It is the sum of the average closest point distances between two point sets.
    Formula here: https://openaccess.thecvf.com/content/ICCV2021/papers/Nguyen_Point-Set_Distances_for_Learning_Representations_of_3D_Point_Clouds_ICCV_2021_paper.pdf
    :param x: [n_points, n_dims]
    :param y: [n_points, n_dims]
    '''
    pt_dist_pairs = np.linalg.norm(x[:, None, :] - y[None, :, :], ord=dist_norm_order, axis=-1)
    closest_pt_dists_x = pt_dist_pairs.min(axis=1)
    one_sided = np.mean(closest_pt_dists_x)
    return one_sided

def two_sided_chamfer_divergence(x, y, dist_norm_order=2):
    '''
    Chamfer discrepancy is not a distance metric. 
    It is the sum of the average closest point distances between two point sets.
    Formula here: https://openaccess.thecvf.com/content/ICCV2021/papers/Nguyen_Point-Set_Distances_for_Learning_Representations_of_3D_Point_Clouds_ICCV_2021_paper.pdf
    :param x: [n_points, n_dims]
    :param y: [n_points, n_dims]
    '''
    with torch.no_grad():
        pt_dist_pairs = torch.linalg.norm(x[:, None, :] - y[None, :, :], ord=dist_norm_order, axis=-1)
        # pt_dist_pairs = pt_dist_pairs ** 2  # square to get squared distances
        closest_pt_dists_x = torch.min(pt_dist_pairs, dim=1)[0]
        closest_pt_dists_y = torch.min(pt_dist_pairs, dim=0)[0]
        classical_chamfer = closest_pt_dists_x.mean() + closest_pt_dists_y.mean()
    return classical_chamfer # ** 0.5  # square root to make it more interpretable

def compute_pairwise_chamfer_divergence(meshes, n_surface_points, device):
    # compute surface points
    pts_list = []
    for mesh in meshes:
        pts = mesh.sample(n_surface_points)
        pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
        pts_list.append(pts_t)
    print(f'Computed {n_surface_points} surface points for {len(meshes)} meshes')
    
    with torch.no_grad():
        # compute mean minimum distance by iterating over all pairs of meshes
        dists = torch.zeros((len(meshes), len(meshes)), device=device)
        for i in range(len(meshes)):
            for j in range(i+1, len(meshes)):
                dist = two_sided_chamfer_divergence(pts_list[i], pts_list[j])
                dists[i, j] = dist
                dists[j, i] = dist
        
    return dists


def _get_face_areas_for_each_vertex(mesh):
    # Compute the area contribution for each vertex in each face
    face_areas_contribution = mesh.area_faces[:, np.newaxis] / 3
    # Accumulate the area contributions for each vertex
    vertex_areas = np.zeros(len(mesh.vertices))
    # very nice operation that sums the face areas for each vertex
    np.add.at(vertex_areas, mesh.faces, face_areas_contribution)
    return vertex_areas

def compute_integral_curvature(mesh):
    # Extract vertices and faces from the mesh
    vertices = mesh.vertices
    faces = mesh.faces

    # https://libigl.github.io/libigl-python-bindings/tut-chapter1/
    v1, v2, k1, k2 = igl.principal_curvature(vertices, faces)
    
    # mean curvature
    H = (k1 + k2) / 2.0
    # gaussian curvature
    K = k1 * k2
    # curvature_expression: '4*H**2 - 2*K'  # E_strain
    # for a sphere of radius=1, E_strain = 2 everywhere
    E_strain = 4 * H ** 2 - 2 * K

    vertex_areas = _get_face_areas_for_each_vertex(mesh)
    # Integrate the mean curvature over the mesh
    total_area = np.sum(vertex_areas)
    integral_curvature = np.sum(E_strain * vertex_areas)
    print(f'total_area: {total_area}, integral_curvature: {integral_curvature}')
    mean_integral_curvature = integral_curvature / total_area
    return integral_curvature, total_area, mean_integral_curvature


def diversity_metric_dist_mat(mat: np.ndarray, p, inner_agg_fn='sum', outer_agg_fn='sum'):
    '''
    Compute the diversity metric based on distances between the objects.
    '''
    n_shapes = mat.shape[0]
    
    if inner_agg_fn == 'min':
        ## set diagonal to infinity
        np.fill_diagonal(mat, np.inf)
    
    row_aggregated = agg_fns[inner_agg_fn](mat, axis=1)  # [n_shapes]
    if inner_agg_fn == 'mean':
        ## account for the fact that we have n_shapes-1 distances
        row_aggregated *= n_shapes / (n_shapes - 1)
    row_aggregated = row_aggregated ** p
    col_aggregated = agg_fns[outer_agg_fn](row_aggregated)  # []
    col_aggregated = col_aggregated ** (1/p)
    return col_aggregated

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
                    
       