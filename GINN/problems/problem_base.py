import os
from turtle import circle
import einops
import torch
import numpy as np
import trimesh
import fast_simplification

from GINN.problems.constraints import BoundingBox2DConstraint, CircleObstacle2D, CompositeInterface2D, Envelope2D, LineInterface2D, CompositeConstraint, SampleConstraint, SampleConstraintWithNormals, SampleEnvelope
from util.model_utils import tensor_product_xz
from models.point_wrapper import PointWrapper
from util.visualization.utils_mesh import get_watertight_mesh_for_latent
from util.misc import get_is_out_mask
from util.visualization.utils_mesh import get_2d_contour_for_grid, get_meshgrid_for_marching_squares, get_meshgrid_in_domain, get_mesh

def t_(x):
    return torch.tensor(x, dtype=torch.float32)

class ProblemBase():
    
    def __init__(self, config) -> None:
        self.config = config
        
    
    def sample_from_envelope(self):
        pts_per_constraint = self.config['n_points_envelope'] // len(self._envelope_constr)
        return torch.cat([c.get_sampled_points(pts_per_constraint) for c in self._envelope_constr], dim=0)
    
    def sample_from_interface(self):
        pts_per_constraint = self.config['n_points_interfaces'] // len(self._interface_constraints)
        pts = []
        normals = []
        for c in self._interface_constraints:
            pts_i, normals_i = c.get_sampled_points(pts_per_constraint)
            pts.append(pts_i)
            normals.append(normals_i)
        return torch.cat(pts, dim=0), torch.cat(normals, dim=0)
    
    def sample_from_obstacles(self):
        pts_per_constraint = self.config['n_points_obstacles'] // len(self._obstacle_constraints)
        return torch.vstack([c.get_sampled_points(pts_per_constraint) for c in self._obstacle_constraints])
    
    def sample_from_domain(self):
        return self._domain.get_sampled_points(self.config['n_points_domain'])
        
    
    def get_mesh_or_contour(self, f, params, z_latents):
        try:
            
            if self.config['nx']==2:
                ## get contour on the marching squares grid for 2d obstacle
                contour_list = []
                with torch.no_grad():   
                    y_ms = f(params, *tensor_product_xz(self.xs_ms, z_latents)).detach().cpu().numpy()
                Y_ms = einops.rearrange(y_ms, '(bz h w) 1 -> bz h w', h=self.X0_ms.shape[0], w=self.X0_ms.shape[1])
                for i, _ in enumerate(z_latents):
                    contour = get_2d_contour_for_grid(self.X0_ms, Y_ms[i], self.bounds)
                    contour_list.append(contour)
                return contour_list
            
            elif self.config['nx']==3:
                ## get watertight verts, faces for simjeb
                verts_faces_list = []
                for z_ in z_latents: ## do marching cubes for every z
                    verts_, faces_ = get_watertight_mesh_for_latent(f, params, z_, bounds=self.config["bounds"],
                                                                mc_resolution=self.config["mc_resolution"], 
                                                                device=z_latents.device, chunks=self.config['mc_chunks'])
                    verts_, faces_ = fast_simplification.simplify(verts_, faces_, target_reduction=self.config['mesh_reduction']) ## target_reduction
                    verts_faces_list.append((verts_, faces_))
                return verts_faces_list
        except Exception as e:
            print(f'WARNING: Could not compute mesh_or_contour for plotting: {e}')
            return None
        
        
    def recalc_output(self, f, params, z_latents):
        """Compute the function on the grid.
        epoch: will be used to identify figures for wandb or saving
        :param z_latents:
        :get_contour: only for 2d; if True, will return the contour instead of the full grid
        """        
        if self.config['nx']==2:
            ## just return the function values on the standard grid (used for visualization)
            with torch.no_grad():
                y = f(params, *tensor_product_xz(self.xs, z_latents)).detach().cpu().numpy()
            Y = einops.rearrange(y, '(bz h w) 1 -> bz h w', h=self.X0.shape[0], w=self.X0.shape[1])
            return y, Y
                    
        elif self.config['nx']==3:
            verts_faces_list = []
            for z_ in z_latents: ## do marching cubes for every z
                
                def f_fixed_z(x):
                    with torch.no_grad():
                        """A wrapper for calling the model with a single fixed latent code"""
                        return f(params, *tensor_product_xz(x, z_.unsqueeze(0))).squeeze(0)
                
                verts_, faces_ = get_mesh(f_fixed_z,
                                            N=self.config["mc_resolution"],
                                            device=z_latents.device,
                                            bbox_min=self.config["bounds"][:,0],
                                            bbox_max=self.config["bounds"][:,1],
                                            chunks=1,
                                            return_normals=0)
                verts_, faces_ = fast_simplification.simplify(verts_, faces_, target_reduction=self.config['mesh_reduction']) ## target_reduction

                # print(f"Found a mesh with {len(verts_)} vertices and {len(faces_)} faces")
                verts_faces_list.append((verts_, faces_))
            return verts_faces_list
    
    def is_inside_envelope(self, p_np: PointWrapper):
        """Remove points that are outside the envelope"""
        if not self.config['problem'] == 'simjeb':
            raise NotImplementedError('This function is only implemented for the simjeb problem')
        
        is_inside_mask = self.mesh_env.contains(p_np.data)
        return is_inside_mask