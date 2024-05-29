'''
Utils for meshing.
'''

import torch
import numpy as np
from skimage import measure
import shapely
from models.model_utils import tensor_product_xz

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

def get_2d_contour_for_latent(f, params, z, bounds, mc_resolution=256, device='cpu', chunks=1, watertight=True):
    def f_fixed_z(x):
        """A wrapper for calling the model with a single fixed latent code"""
        return f(params, *tensor_product_xz(x, z.unsqueeze(0))).squeeze(0)
    
    # evaluate at the meshgrid
    X0, X1, xs = get_meshgrid_in_domain(bounds, mc_resolution)
    xs = torch.tensor(xs, dtype=torch.float32, device=device)
    Y = f_fixed_z(xs).detach().cpu().numpy().reshape(X0.shape)
    
    Y = Y.T
    print('Y shape: ', Y.shape)
    
    ## find the contours
    contours = measure.find_contours(Y, level=0.0)
    
    ## scale and translate
    scaled_and_translated_contours = []
    for contour in contours:    
        # rescale the contour to the domain
        # print('contour bounds: ', contour.min(axis=0), contour.max(axis=0))    
        contour[:, 0] = contour[:, 0] / X0.shape[1]
        contour[:, 1] = contour[:, 1] / X0.shape[0]
        new_contour = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * contour
        scaled_and_translated_contours.append(new_contour)
        # print(contour.shape)
        # print('bounds: ', bounds)
    # create multi-polygon
    contour_shape = shapely.geometry.MultiPolygon([shapely.geometry.Polygon(contour) for contour in scaled_and_translated_contours])    

    if watertight:
        # bounds contour
        contour_bounds = shapely.geometry.box(*bounds.flatten())
        contour_shape = contour_shape.intersection(contour_bounds)
        raise ValueError('watertight=True not tested yet!')
    
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
    