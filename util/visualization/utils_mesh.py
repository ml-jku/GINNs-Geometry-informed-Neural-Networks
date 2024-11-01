'''
Utils for meshing.
'''

import math
from operator import is_
from sympy import Polygon
import torch
import numpy as np
from skimage import measure
import trimesh
import shapely
from shapely.geometry import MultiPolygon
from util.model_utils import tensor_product_xz

def mc(V, level=0, return_normals=False):
    '''Marching cubes'''
    V = V.detach().cpu().numpy()
    try:
        verts, faces, normals, values = measure.marching_cubes(V, level)
    except:
        print("WARNING: level not within the volume data range. Returning empty mesh.")
        verts, faces, normals = np.zeros([0,3], dtype=float), np.zeros([0,3], dtype=int), np.zeros([0,3], dtype=float)
    if return_normals: return verts, faces, -normals
    return verts, faces

def preprocess_for_watertight(vals):
    """Pad a 3D array with zeros plus some extra TODO"""
    vals_ = vals * -1 # not 100% sure why this is needed here
    vals_ = torch.nn.functional.pad(vals_, (1,1,1,1,1,1))
    return vals_

def get_2d_contour_for_latent(f, params, z, bounds, device='cpu'):
    def f_fixed_z(x):
        """A wrapper for calling the model with a single fixed latent code"""
        with torch.no_grad():
            return f(params, *tensor_product_xz(x, z.unsqueeze(0))).squeeze(0)
    
    bounds = bounds.cpu()
    
    # evaluate at the meshgrid
    X0, X1, xs = get_meshgrid_in_domain(bounds)
    xs = torch.tensor(xs, dtype=torch.float32, device=device)
    Y = f_fixed_z(xs).detach().cpu().numpy().reshape(X0.shape)
    
    # print(f'X0: {X0.shape}, {X0}')
    # print(f'Y: {Y.shape}, {Y}')
    
    return get_2d_contour_for_grid(X0, Y, bounds)
    
def get_2d_contour_for_grid(X0, Y, bounds):
    # transpose to match the meshgrid
    Y = Y.T
    
    ## find the contours
    '''
    From the function 'Meshgrid for marching squares':
    For stability of this algorithm: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours
    we have to set the outermost points to be outside the domain (i.e. for the SDF to be positive).
    To do this, we add pad with 1 point at each side of the domain.
    With this, e.g. case 3 (https://users.polytech.unice.fr/~lingrand/MarchingCubes/algo.html) will be at the boundary of the domain.
    '''
    ## now we have to set the outermost points to be outside the domain (i.e. for the SDF to be positive)
    Y[0, :] = 1.0
    Y[-1, :] = 1.0
    Y[:, 0] = 1.0
    Y[:, -1] = 1.0
    contours = measure.find_contours(Y, level=0.0)
    
    ## scale and translate
    scaled_and_translated_contours = []
    for contour in contours:
        if len(contour) < 3:
            print('WARNING: skipping contour with less than 3 points; fix this !!!')
            continue
        
        # rescale the contour to the domain
        # print('contour bounds: ', contour.min(axis=0), contour.max(axis=0))    
        contour[:, 0] = contour[:, 0] / X0.shape[1]
        contour[:, 1] = contour[:, 1] / X0.shape[0]
        new_contour = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * contour
        scaled_and_translated_contours.append(new_contour)
        # print(contour.shape)
        # print('bounds: ', bounds)
    
    ## TODO: fix HERE
    ## create multi-polygon
    # contour_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon(contour) for contour in scaled_and_translated_contours])
    polygons = [shapely.geometry.Polygon(contour) for contour in scaled_and_translated_contours]
    # sort by area descending
    polygons = sorted(polygons, key=lambda x: x.area, reverse=True)
    
    new_polygons = [] 
    already_added = []
    for i in range(len(polygons)):
        new_poly = polygons[i]
        if i in already_added:
            continue
        ## check if any of the other polygons are inside this one
        for j in range(i+1, len(polygons)):
            if j in already_added:
                continue
            if new_poly.contains(polygons[j]):
                ## since the contours are closed and one is inside the other
                ## we can use union - intersection
                new_poly = new_poly.union(polygons[j]).difference(new_poly.intersection(polygons[j]))
                already_added.append(j)

        new_polygons.append(new_poly)
    
    contour_shape = shapely.geometry.MultiPolygon(new_polygons)
    
    return contour_shape

def get_mesh(model,
        N=32,
        device='cpu',
        bbox_min=(-1,-1,-1),
        bbox_max=(1,1,1), 
        chunks=1, 
        flip_faces=False, 
        level=0, 
        return_normals=0,
        watertight=False,
        ):
    '''
    For marchihng cubes split the evaluation of the grid into chunks since for high resolutions memory can be limiting.
    NOTE: for chunks>1 there are duplicate vertices at the seams. Merge them.
    '''
    with torch.no_grad():
        verts_all = []
        faces_all = []
        if return_normals: normals_all = []
        x0, y0, z0 = bbox_min
        x1, y1, z1 = bbox_max
        dx, dy, dz = (x1-x0)/chunks, (y1-y0)/chunks, (z1-z0)/chunks
        nV = 0 # nof vertices
        for i in range(chunks):
            for j in range(chunks):
                for k in range(chunks):
                    xmin, ymin, zmin = x0+dx*i, y0+dy*j, z0+dz*k
                    xmax, ymax, zmax = xmin+dx, ymin+dy, zmin+dz
                    lsx = torch.linspace(xmin, xmax, N//chunks, device=device)
                    lsy = torch.linspace(ymin, ymax, N//chunks, device=device)
                    lsz = torch.linspace(zmin, zmax, N//chunks, device=device)
                    X, Y, Z = torch.meshgrid(lsx, lsy, lsz, indexing='ij')
                    pts = torch.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                    vs = model(pts.to(device=device))
                    V = vs.reshape(X.shape).cpu()
                    if watertight:
                        if chunks!=1:
                            raise NotImplementedError ## TODO
                        ## Pad
                        V = preprocess_for_watertight(V)
                        ## Adjust the coordinate system (TODO: vectorize)
                        xmin -= dx/N
                        xmax += dx/N
                        dx /= N*(N+2)
                        ymin -= dy/N
                        ymax += dy/N
                        dy /= N*(N+2)
                        zmin -= dz/N
                        zmax += dz/N
                        dz /= N*(N+2)
                    res = mc(V, level=level, return_normals=return_normals)
                    if return_normals: verts, faces, normals = res
                    else: verts, faces = res
                    verts[:,0] = (xmax-xmin).cpu()*verts[:,0]/(V.shape[0]-1) + xmin.cpu()
                    verts[:,1] = (ymax-ymin).cpu()*verts[:,1]/(V.shape[1]-1) + ymin.cpu()
                    verts[:,2] = (zmax-zmin).cpu()*verts[:,2]/(V.shape[2]-1) + zmin.cpu()
                    verts_all.append(verts)
                    faces_all.append(faces+nV)
                    if return_normals: normals_all.append(normals)
                    nV += len(verts)
    verts = np.vstack(verts_all)
    faces = np.vstack(faces_all)
    if return_normals: normals_all = np.vstack(normals_all)
    if flip_faces: faces = faces[:,::-1]
    if return_normals: return verts, faces, normals
    return verts, faces

def get_watertight_mesh_for_latent(f, params, z, bounds, mc_resolution=256, device='cpu', chunks=1, surpress_watertight=False):
    def f_fixed_z(x):
        """A wrapper for calling the model with a single fixed latent code"""
        with torch.no_grad():
            return f(params, *tensor_product_xz(x, z.unsqueeze(0))).squeeze(0)
    
    verts_, faces_ = get_mesh(f_fixed_z,
                                N=mc_resolution,
                                device=device,
                                bbox_min=bounds[:,0],
                                bbox_max=bounds[:,1],
                                chunks=chunks,
                                return_normals=0,
                                watertight=True)
    # print(f"Found a mesh with {len(verts_)} vertices and {len(faces_)} faces")
    
    if surpress_watertight: return verts_, faces_

    ## intersection with the bounding box to make watertight
    mesh_model = trimesh.Trimesh(vertices=verts_, faces=faces_)
    mesh_bbox = trimesh.primitives.Box(bounds=bounds.T.cpu().numpy() * 1.1)  ## try out if this makes the mesh watertight
    mesh_model = mesh_model.intersection(mesh_bbox)
    verts_, faces_ = mesh_model.vertices, mesh_model.faces
    
    ## last step to make watertight
    ## fix inversion; alternatively could do: https://trimesh.org/trimesh.repair.html#trimesh.repair.fix_inversion
    faces_ = faces_[:,::-1]
    
    return verts_, faces_

def compute_unit_sphere_transform(mesh):
    """
    returns translation and scale, which is applied to meshes before computing their SDF cloud
    ## Rescale the mesh. From https://github.com/marian42/mesh_to_sdf/issues/23
    """
    # the transformation applied by mesh_to_sdf.scale_to_unit_sphere(mesh)
    translation = -mesh.bounding_box.centroid
    scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))
    return translation, scale

def normalize_trimesh(mesh):
    """Normalizes trimesh vertices to unit sphere. Inplace."""
    translation, scale = compute_unit_sphere_transform(mesh)
    mesh.vertices = (mesh.vertices + translation)*scale

def get_mesh_bbox(mesh):
    bbox_min = mesh.vertices.min(0)
    bbox_max = mesh.vertices.max(0)
    return bbox_min, bbox_max

def get_meshgrid_in_domain(bounds_np, nx=400, ny=400):
        """Meshgrid the domain"""
        x0s = np.linspace(*bounds_np[0], nx)
        x1s = np.linspace(*bounds_np[1], ny)
        X0, X1 = np.meshgrid(x0s, x1s)
        xs = np.vstack([X0.ravel(), X1.ravel()]).T
        return X0, X1, xs
    
def get_meshgrid_for_marching_squares(bounds_np, nx=400, ny=400):
    '''
    Meshgrid for marching squares.
    For stability of this algorithm: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours
    we have to set the outermost points to be outside the domain (i.e. for the SDF to be positive).
    To do this, we add pad with 1 point at each side of the domain.
    With this, e.g. case 3 (https://users.polytech.unice.fr/~lingrand/MarchingCubes/algo.html) will be at the boundary of the domain.
    '''
    
    nx = nx + 2
    ny = ny + 2
    
    overhang_x_rel = 1.0 / (nx - 2) / 2
    start_x = bounds_np[0][0] - overhang_x_rel * (bounds_np[0][1] - bounds_np[0][0])
    stop_x = bounds_np[0][1] + overhang_x_rel * (bounds_np[0][1] - bounds_np[0][0])
    x0s = np.linspace(start=start_x, stop=stop_x, num=nx)
    
    overhang_y_rel = 1.0 / (ny - 2) / 2
    start_y = bounds_np[1][0] - overhang_y_rel * (bounds_np[1][1] - bounds_np[1][0])
    stop_y = bounds_np[1][1] + overhang_y_rel * (bounds_np[1][1] - bounds_np[1][0])        
    x1s = np.linspace(start=start_y, stop=stop_y, num=ny)
    
    X0, X1 = np.meshgrid(x0s, x1s)
    xs = np.vstack([X0.ravel(), X1.ravel()]).T
    return X0, X1, xs
    
def sample_pts_interior_and_exterior(mpoly:MultiPolygon, n_samples):
    def sample_from_line(line, num):
        interval = 1.0 / num
        return [line.interpolate(interval * i, normalized=True) for i in range(num)]
    
    total_length = mpoly.length
    if total_length == 0:
        return False, None
    
    points = []
    for poly in mpoly.geoms:
        ## sample exterior
        points_for_this_poly = int(math.ceil(poly.exterior.length / total_length * n_samples))
        points.extend(sample_from_line(poly.exterior, points_for_this_poly))
            
        # Sample each interior
        for interior in poly.interiors:
            points_for_this_poly = int(math.ceil(interior.length / total_length * n_samples))
            points.extend(sample_from_line(interior, points_for_this_poly))
        
    np_points = np.array([[pt.x, pt.y] for pt in points])
    
    ## remove some points if there are too many
    if len(np_points) > n_samples:
        np_points = np_points[np.random.choice(np_points.shape[0], n_samples, replace=False), :]
    return True, np_points