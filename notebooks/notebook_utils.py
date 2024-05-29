import k3d
import trimesh
import shapely
from visualization.utils_mesh import get_mesh



class OptHelper:
    
    def __init__(self, f, params_attached, bounds, z, nx, device='cpu'):
        self.f = f
        self.params = {key: tensor.detach() for key, tensor in params_attached.items()}
        self.bounds = bounds
        self.z = z
        self.nx = nx
        self.device = device
        
    def flow_to_surface_points(self, n_pts=1000, n_steps=100, lr=1e-2, eps=1e-2):
        start_pts = torch.rand((n_pts, self.nx), device=self.device) * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]
        start_pts.requires_grad = True
        opt = torch.optim.Adam([start_pts], lr=lr)
        for i in range(n_steps):
            opt.zero_grad()
            y = self.f(self.params, *tensor_product_xz(start_pts, self.z.unsqueeze(0))).squeeze(1)
            loss = y.square().mean()
            loss.backward()
            opt.step()
            
        # remove those out of bounds
        start_pts = start_pts.detach()
        mask_out_of_bounds = ((start_pts < self.bounds[:,0]) | (start_pts > self.bounds[:,1])).any(dim=1)
        start_pts = start_pts[~mask_out_of_bounds]
        y = y[~mask_out_of_bounds]
        
        # remove those with |f(x)| > eps
        mask_out_of_eps = (y.abs() > eps)
        start_pts = start_pts[~mask_out_of_eps]
        y = y[~mask_out_of_eps]
            
        return start_pts
    
## 3D utils

    
def get_mesh_for_latent(f, params, z, bounds, mc_resolution=256, device='cpu', chunks=1, watertight=False):
    ## TODO: device is broken
    def f_fixed_z(x):
        """A wrapper for calling the model with a single fixed latent code"""
        return f(params, *tensor_product_xz(x, z.unsqueeze(0))).squeeze(0)
    
    verts_, faces_ = get_mesh(f_fixed_z,
                                N=mc_resolution,
                                device=device,
                                bbox_min=bounds[:,0],
                                bbox_max=bounds[:,1],
                                chunks=chunks,
                                return_normals=0,
                                watertight=watertight,)
    # print(f"Found a mesh with {len(verts_)} vertices and {len(faces_)} faces")
    
    if watertight:
        mesh_model = trimesh.Trimesh(vertices=verts_, faces=faces_)
        mesh_bbox = trimesh.primitives.Box(bounds=bounds.T)
        mesh_model = mesh_model.intersection(mesh_bbox)  # intersection with the bounding box to make watertight
        verts_, faces_ = mesh_model.vertices, mesh_model.faces
    
    # fix inversion; alternatively could do: https://trimesh.org/trimesh.repair.html#trimesh.repair.fix_inversion
    faces_ = faces_[:,::-1]
    
    return verts_, faces_

def display_meshes(meshes, bounds, attribute_list=[], pts_list=[], colors=0xff0000):
    shape_grid = 1.5*(bounds[:,1] - bounds[:,0]) ## distance between the shape grid for plotting
    
    fig = k3d.plot(height=800)#, grid_visible=False, camera_fov=1.0)
    n_cols = int(np.ceil(np.sqrt(len(meshes))))
    for i_shape in range(len(meshes)):
        i_col = (i_shape  % n_cols)
        i_row = (i_shape // n_cols)
        verts, faces = meshes[i_shape]
        group = f'Shape {i_shape}'
        if isinstance(colors, list):
            color = colors[i_shape]
        else:
            print(f'colors is of type {type(colors)} or instance of {isinstance(colors, list)}')
            color = colors
            
        if len(attribute_list) > 0:
            fig += k3d.mesh(verts, faces, attribute=attribute_list[i_shape], group=group, color=color, side='double',  opacity=0.8, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
        else:
            fig += k3d.mesh(verts, faces, group=group, color=color, side='double',  opacity=0.8, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
        
        if len(pts_list) > 0:
            pts = pts_list[i_shape]
            fig += k3d.points(pts, point_size=0.01, color=0x000000, opacity=0.8, shader='flat', translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row])
        
    fig.display()
    
    
    
    
    
## 2D utils

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from models.model_utils import tensor_product_xz

class GenPlotter():

    def __init__(self, config, model, bounds, fig_size=(16,16)) -> None:
        self.config = config
        self.model = model
        self.bounds = bounds
        self.fig_size = fig_size

        self.plot_scale = (self.bounds[:, 1] - self.bounds[:, 0]).prod().sqrt()  # scale for plotting

        self.X0, self.X1, self.xs = self.get_meshgrid_in_domain()
        self.y = None
        self.Y = None
        self.epoch = 0

        self.bounds = config['bounds'].cpu()
        self.plot_scale = (self.bounds[:, 1] - self.bounds[:, 0]).prod().sqrt()  # scale for plotting

        self.X0, self.X1, self.xs = self.get_meshgrid_in_domain()
        self.y = None
        self.Y = None

    def get_meshgrid_in_domain(self, nx=400, ny=400):
        """Meshgrid the domain"""
        x0s = np.linspace(*self.bounds[0], nx)
        x1s = np.linspace(*self.bounds[1], ny)
        X0, X1 = np.meshgrid(x0s, x1s)
        xs = np.vstack([X0.ravel(), X1.ravel()]).T
        xs = torch.tensor(xs).float()
        return X0, X1, xs

    def recalc_y(self, z_latents):
        """Compute the function on the grid.
        :param z_latents:
        """
        with torch.no_grad():
            self.y = self.model(*tensor_product_xz(self.xs, z_latents))
            self.Y = einops.rearrange(self.y.detach().cpu().numpy(), '(bz h w) 1 -> bz h w', bz=len(z_latents),
                                      h=self.X0.shape[0])

    def draw_base(self, ax, i_shape, draw_constraint_samples=False, show_colorbar=None):
        Y_i = self.Y[i_shape]
        im = ax.imshow(Y_i, origin='lower', extent=self.bounds.flatten())
        ax.contour(self.X0, self.X1, Y_i, levels=50, colors='gray', linewidths=1.0)
        ax.contour(self.X0, self.X1, Y_i, levels=[self.config['level_set']], colors='r')

        if draw_constraint_samples:
            for i, constraint in enumerate(self.constraint_list):
                color = 'k' if i == 0 else plt.cm.tab10(i)
                ax.scatter(*constraint.get_sampled_points().cpu().numpy().T, color=color, marker='.', zorder=10)

        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust the size and pad as needed
            plt.colorbar(im, cax=cax)

    def plot_shape_for_latents(self, z, fig_label):
        '''
        z: [(nc ni) nz]  # nc is the number of corners visited; ni is the number of intermediate points
        '''
        k, nz = z.shape

        n_cols = int(np.ceil(np.sqrt(k)))
        n_rows = int(np.ceil(k / n_cols))

        fig, axs = plt.subplots(n_rows, n_cols, figsize=self.fig_size)
        axs = axs.ravel()
        for i_shape in range(n_cols * n_rows):
            ax = axs[i_shape]
            if i_shape >= k:
                ax.axis('off')
                continue
            
            self.draw_base(ax, i_shape)
            ax.set_title(f'z:\n{self.beautify_z(z[i_shape])}')

        plt.suptitle(fig_label)
        plt.tight_layout()
        plt.show()

    def beautify_z(self, z):
        rounded_strings = [f'{x:.2f}' for x in z]
        max_col = 4
        # n_rows = (len(rounded_strings) - 1) // max_col + 1
        result_string = '['
        for i, z_str in enumerate(rounded_strings):
            if i > 0 and i % max_col == 0:
                result_string += '\n '
            result_string += z_str + ', '
        result_string = result_string[:-2]  # remove last ', '
        result_string += ']'
        return result_string
    
def plot_poly_or_multipolygon(ax, poly_or_multipoly, color='blue', alpha=0.5):
    if isinstance(poly_or_multipoly, shapely.geometry.Polygon):
        ax.plot(*poly_or_multipoly.exterior.xy, color=color, alpha=alpha)
    elif isinstance(poly_or_multipoly, shapely.geometry.MultiPolygon):
        for polyg in poly_or_multipoly.geoms:
            ax.plot(*polyg.exterior.xy, color=color, alpha=alpha)
    elif isinstance(poly_or_multipoly, shapely.geometry.MultiLineString):
        for line in poly_or_multipoly.geoms:
            ax.plot(*line.xy, color=color, alpha=alpha)
    elif isinstance(poly_or_multipoly, shapely.geometry.LineString):
        ax.plot(*poly_or_multipoly.xy, color=color, alpha=alpha)
    elif isinstance(poly_or_multipoly, shapely.geometry.GeometryCollection):
        for geom in poly_or_multipoly.geoms:
            plot_poly_or_multipolygon(ax, geom, color=color, alpha=alpha)
    else:
        raise ValueError('poly_or_multipoly must be of type Polygon or MultiPolygon')