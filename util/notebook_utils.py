import k3d
from matplotlib import patches
from matplotlib.collections import LineCollection
import trimesh
import shapely
from util.visualization.utils_mesh import get_mesh



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
from util.model_utils import tensor_product_xz

class GenPlotter():

    def __init__(self, config, model, bounds, fig_size=(16,16)) -> None:
        self.config = config
        self.model = model
        self.bounds = bounds
        self.fig_size = fig_size

        self.X0, self.X1, self.xs = self.get_meshgrid_in_domain()
        self.y = None
        self.Y = None
        self.epoch = 0
        self.bounds = config['bounds'].cpu()
        self.plot_scale = (self.bounds[:, 1] - self.bounds[:, 0]).prod().sqrt()  # scale for plotting

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

    def draw_base(self, ax, i_shape):
        Y_i = self.Y[i_shape]
        print(Y_i.min(), Y_i.max())
        # cm_out = cm.get_cmap('Oranges_r', 128)
        # cm_in = cm.get_cmap('Blues', 128)
        # colors = np.vstack((cm_in(np.linspace(0, 1, 128)), cm_out(np.linspace(0, 1, 128))))
        # colors = [lighten_color(color, .4) for color in colors]
        # cmap = ListedColormap(colors)

        # ax.imshow(Y_i, origin='lower', extent=self.bounds.flatten(), cmap='bwr', vmin=-.2, vmax=.2, alpha=.5)
        # ax.contourf(self.X0, self.X1, Y_i, cmap='bwr', levels=40, vmin=-.3, vmax=.3)
        # ax.contourf(self.X0, self.X1, Y_i, cmap=cmap, levels=40, vmin=-.1, vmax=.1)
        # ax.contour(self.X0, self.X1, Y_i, levels=50, colors='gray', linewidths=1.0)
        # ax.contour(self.X0, self.X1, Y_i, levels=[self.config['level_set']], colors='k')
        # ax.contourf(self.X0, self.X1, Y_i, levels=[-1e10, self.config['level_set']], colors='gray', antialiased=True)
        
        ## draw gray object until level set 0.0
        ## make color RGB(208, 208, 208)
        color = '#D0D0D0'
        # color = '#929591'
        ax.contourf(self.X0, self.X1, Y_i, levels=[-1e10, self.config['level_set']], colors=color, antialiased=True)
    
    def draw_constraints(self, ax):
        color_nondesign = 'gray'
        color_interface = 'darkgreen'
        # Domain
        x_domain = [-1, 1]
        y_domain = [-.5, .5]
        # Design region
        x_design = [-0.9, 0.9]
        y_design = [-0.4, 0.4]
        
        # kwargs_obstacle = {'fill':None, 'color':'gray', 'hatch':'////', 'linewidth':0, 'alpha':1.0}
        kwargs_obstacle = {'fill':None, 'color':'black', 'linewidth':1.0, 'alpha':1.0}
        # ax.add_patch(patches.Rectangle((x_domain[0], y_domain[0]), x_domain[1]-x_domain[0], y_domain[1]-y_domain[0], **kwargs_obstacle))
        ax.add_patch(patches.Rectangle((x_design[0], y_design[0]), x_design[1]-x_design[0], y_design[1]-y_design[0], **kwargs_obstacle))


        # ax.add_patch(patches.Rectangle((x_domain[0], y_design[1]), x_domain[1]-x_domain[0], y_domain[1]-y_design[1], **kwargs_obstacle))
        # ax.add_patch(patches.Rectangle((x_domain[0], y_domain[0]), x_domain[1]-x_domain[0], y_design[0]-y_domain[0], **kwargs_obstacle))
        # ax.add_patch(patches.Rectangle((x_domain[0], y_design[0]), x_design[0]-x_domain[0], y_design[1]-y_design[0], **kwargs_obstacle))
        # ax.add_patch(patches.Rectangle((x_design[1], y_design[0]), x_domain[1]-x_design[1], y_design[1]-y_design[0], **kwargs_obstacle))
        # pc = PatchCollection([rect1, rect2, rect3, rect4], facecolors=color_nondesign, alpha=.4)
        # ax.add_collection(pc)
        # disk = plt.Circle((0, 0), 0.1, facecolor=color_nondesign, alpha=.4)


        disk = plt.Circle((0, 0), 0.1, **kwargs_obstacle)
        ax.add_patch(disk)

        # Interface
        segments = ([(-.9, -.4), (-.9, .4)], [(.9, -.4), (.9, .4)])
        line_segments = LineCollection(segments, linewidths=2, colors=color_interface)
        ax.add_collection(line_segments)


    def plot_shape_for_latents(self, n_cols=4, n_rows=3, z_latents=None):
        '''
        z: [(nc ni) nz]  # nc is the number of corners visited; ni is the number of intermediate points
        '''

        fig, axs = plt.subplots(n_rows, n_cols, figsize=self.fig_size)
        for i_shape in range(n_cols * n_rows):
            if n_cols==1 and n_rows==1:
                ax = axs
            else:
                ax = axs.ravel()[i_shape]
            self.draw_base(ax, i_shape)
            self.draw_constraints(ax)
            ax.axis('off')
            ax.axis('scaled')
            ax.set_xlim(*self.bounds[0].T)
            ax.set_ylim(*self.bounds[1].T)
            if z_latents is not None:
                ax.set_title(f'z={z_latents[i_shape].item():.2f}')

        plt.tight_layout()
        # plt.show()
        return fig

def plot_poly_or_multipolygon(ax, poly_or_multipoly, color='blue', alpha=0.5):
    if isinstance(poly_or_multipoly, shapely.geometry.Polygon):
        ax.plot(*poly_or_multipoly.exterior.xy, color=color, alpha=alpha)
    elif isinstance(poly_or_multipoly, shapely.geometry.MultiPolygon):
        for polyg in poly_or_multipoly.geoms:
            ax.plot(*polyg.exterior.xy, color=color, alpha=alpha)
            for interior in polyg.interiors:
                ax.plot(*interior.xy, color=color, alpha=alpha, linestyle='--')
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