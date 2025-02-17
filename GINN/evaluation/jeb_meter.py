import logging
import os
from re import S
import time
import numpy as np
import torch
import trimesh

from GINN import shape_boundary_helper
from GINN.speed.dummy_mp_manager import DummyMPManager
from GINN.shape_boundary_helper import ShapeBoundaryHelper
from GINN.plot.plotter_dummy import DummyPlotter
from GINN.evaluation.metrics_utils import two_sided_chamfer_divergence, diversity_metric
from util.model_utils import tensor_product_xz
from train.losses import strain_curvature_loss

class JebMeter:
    
    def _load_mesh_and_scale_and_center(self, mesh_path, center_for_translation, scale_factor):
        mesh = trimesh.load(mesh_path)
        mesh.apply_translation(-center_for_translation)
        mesh.apply_scale(1/scale_factor)
        return mesh
    
    def __init__(self, 
                 simjeb_root_dir, 
                 n_points_surface=8192,
                 metrics_diversity_n_samples=1000,
                 n_samples_at_if_for_chamfer=10000, 
                 debug_intermediate_shapes=False):
        
        self.n_points_surface = n_points_surface
        self.metrics_diversity_n_samples = metrics_diversity_n_samples
        
        self.logger = logging.getLogger('SimJEBMeter')
        self.debug_intermediate_shapes = debug_intermediate_shapes
        
        env_path = os.path.join(simjeb_root_dir, '411_for_envelope.obj')
        interface_path = os.path.join(simjeb_root_dir, 'interfaces.stl')
        center_for_translation_path = os.path.join(simjeb_root_dir, 'center_for_translation.npy')
        scale_factor_path = os.path.join(simjeb_root_dir, 'scale_factor.npy')
        bounds_path = os.path.join(simjeb_root_dir, 'bounds.npy')
        
        self.center_for_translation = np.load(center_for_translation_path)
        self.scale_factor = np.load(scale_factor_path)
        self.bounds = np.load(bounds_path)
        self.mesh_design_region = self._load_mesh_and_scale_and_center(env_path, self.center_for_translation, self.scale_factor)
        self.mesh_interface = self._load_mesh_and_scale_and_center(interface_path, self.center_for_translation, self.scale_factor)

        self.pts_at_if = self.mesh_interface.sample(n_samples_at_if_for_chamfer)

        self.mesh_bbox = trimesh.primitives.Box(bounds=self.bounds.T)
        self.mesh_design_region_neg = self.mesh_bbox.difference(self.mesh_design_region)
        self.volume_design_region_neg = self.mesh_design_region_neg.volume
        # print('Volume of negative envelope:', self.volume_design_region_neg)
        self.d_minus_1_volume_design_region_boundary = self.mesh_design_region.area
        # print('Area of envelope surface:', self.area_envelope_surface)
        
        self.if_components = trimesh.graph.split(self.mesh_interface, only_watertight=False)
        self.if_samples = []
        for i in range(len(self.if_components)):
            self.if_samples.append(self.if_components[i].sample(n_samples_at_if_for_chamfer // 6))
            
        self.domain_length = np.max(np.diff(self.bounds, axis=1))
        self.logger.debug(f'Domain length: {self.domain_length}')
        
    def get_average_metrics_as_dict(self, verts_faces_list, netp=None, z_latents=None, prefix=''):
        if z_latents is not None:
            assert len(verts_faces_list) == len(z_latents), 'Number of meshes and latent vectors must be the same'
        else:
            z_latents = [None] * len(verts_faces_list)
        
        shape_boundary_helper = None
        if netp is not None:
            shape_boundary_helper = \
                ShapeBoundaryHelper(nx=3, bounds=self.bounds, netp=netp, 
                                    mp_manager=DummyMPManager(), plotter=DummyPlotter(), 
                                    x_interface=self.pts_at_if, n_points_surface=self.n_points_surface)
        
        metrics_dicts_list = []
        mesh_models = []
        for i, (verts, faces) in enumerate(verts_faces_list):
            mesh_model = trimesh.Trimesh(vertices=verts, faces=faces)
            if not mesh_model.is_watertight:
                return {}
            
            mesh_models.append(mesh_model)
            metrics = self.get_all_metrics_as_dict(mesh_model, netp, z_latents[i], shape_boundary_helper=shape_boundary_helper)
            metrics_dicts_list.append(metrics)
        
        avg_metrics = {}
        for key in metrics_dicts_list[0].keys():
            avg_metrics['avg_' + key] = np.mean([m[key] for m in metrics_dicts_list])
            avg_metrics['std_' + key] = np.std([m[key] for m in metrics_dicts_list])
        
        if len(mesh_models) > 1:
            avg_metrics.update(self.diversity_chamfer(mesh_models, n_samples=self.metrics_diversity_n_samples))
        
        ## update keys with prefix
        for key in list(avg_metrics.keys()):
            avg_metrics[prefix + key] = avg_metrics.pop(key)
        
        self.logger.info(f'Finished computing all metrics')
        return avg_metrics
    
    def get_all_metrics_as_dict(self, mesh_model, netp=None, z=None, shape_boundary_helper:ShapeBoundaryHelper=None):
        
        metrics = {}
        metrics['d-volume'] = mesh_model.volume
        metrics['(d-1)-volume'] = mesh_model.area
        
        self.logger.debug(f'Computing one-sided chamfer distance')
        metrics.update(self.one_sided_chamfer_distance(mesh_model))  # Used: yes
        
        self.logger.debug(f'Computing zeroth betti number')
        metrics.update(self.zeroth_betti_number(mesh_model))  # Used: yes
        metrics.update(self.zeroth_betti_number_within_envelope(mesh_model))  # Used: yes
        
        self.logger.debug(f'Computing volume outside envelope')  # Used: yes
        metrics.update(self.volume_outside_env(mesh_model))
        
        self.logger.debug(f'Computing model intersect env surface')  # Used: yes
        metrics.update(self.model_intersect_env_surface(mesh_model))
        
        self.logger.debug(f'Computing n connected interfaces')
        metrics.update(self.n_connected_interfaces(mesh_model))  # Used: yes
        
        self.logger.debug(f'Computing d-volume of disconnected components')
        metrics.update(self.d_volume_of_disconnected_components_in_domain(mesh_model)) # Used: yes
        metrics.update(self.d_volume_of_disconnected_components_in_design_region(mesh_model)) # Used: yes
        
        if netp is not None:
            self.logger.debug(f'Computing curvature')
            surf_pts, _ = shape_boundary_helper.get_surface_pts(z[None, :])
            start_t = time.time()
            metrics.update(self.curvature(netp, z[None, :], mesh_model=mesh_model, surf_pts=surf_pts.data))
            print(f'needed {time.time() - start_t} seconds for curvature')
        
        return metrics
        
    def one_sided_chamfer_distance(self, mesh_model):
        dist_to_if = mesh_model.nearest.on_surface(self.pts_at_if)[1]
        one_sided_chamfer = dist_to_if.mean()
        return {'one_sided_chamfer distance to interface': one_sided_chamfer}
    
    def zeroth_betti_number(self, mesh_model):
        components = trimesh.graph.split(mesh_model, only_watertight=False)
        return {'betti_0': len(components)}
    
    def zeroth_betti_number_within_envelope(self, mesh_model):
        
        mesh_model_inside_env = self.mesh_design_region.intersection(mesh_model)
        components = trimesh.graph.split(mesh_model_inside_env, only_watertight=False)
        
        res_dict = {'betti_0 inside design region': len(components)}
        if self.debug_intermediate_shapes:
            res_dict['mesh_model_inside_env'] = mesh_model_inside_env
        return res_dict
        
    def volume_outside_env(self, mesh_model):
        mesh_model_outside_env = self.mesh_design_region_neg.intersection(mesh_model)
        outside_env_volume = mesh_model_outside_env.volume
        vol_share = outside_env_volume / self.volume_design_region_neg
        
        res_dict = {'d-volume outside design region share': vol_share}
        if self.debug_intermediate_shapes:
            res_dict['d-volume outside design region'] = outside_env_volume
            res_dict['mesh_model_outside_env'] = mesh_model_outside_env
        return res_dict
        
    def model_intersect_env_surface(self, mesh_model):
        mesh_model_design = self.mesh_design_region.intersection(mesh_model)
        mesh_model_env_surface = self.mesh_design_region_neg.intersection(mesh_model_design)
        
        mesh_model_env_surface_area = mesh_model_env_surface.area
        area_share = mesh_model_env_surface_area / self.d_minus_1_volume_design_region_boundary
        
        res_dict = {'(d-1)-volume model intersect design region share': area_share}
        if self.debug_intermediate_shapes:
            res_dict['(d-1)-volume model intersect design region'] = mesh_model_env_surface_area
            res_dict['mesh_model_env_surface'] = mesh_model_env_surface
        return res_dict
    
    def n_connected_interfaces(self, mesh_model, eps=0.01, only_within_design_region=True):
        '''
        Check how many interfaces are connected to the model by searching the connected component with the most interfaces in proximity.
        Proximity vertices are those that are eps-close to an interface.
        args:
        contour_model: shapely.geometry.Polygon
        eps: float, proximity threshold in percentage of domain length (i.e. longest side-length of the domain)
        only_within_design_region: bool, whether to consider only the part of the model inside the design region, default is True
        '''
        eps = eps * self.domain_length
        
        if only_within_design_region:
            mesh_model = self.mesh_design_region.intersection(mesh_model)

        # get vertices of mesh eps-close to the interface
        if_prox_vertices = []
        vertices = mesh_model.vertices
        for i in range(len(self.if_components)):
            ## distance via broadcasting is much faster than distance to mesh_if
            dist_vertices = np.linalg.norm(self.if_samples[i][:,None] - vertices, axis=2).min(axis=0)
            vertices_close_to_if = vertices[dist_vertices < eps]
            if_prox_vertices.append(vertices_close_to_if)
        
        # for each connected component, check how many interfaces it contains, by checking if it contains at least one proximity vertex from each interface
        n_connected_interfaces = 0
        for conn_comp in trimesh.graph.split(mesh_model, only_watertight=False):
            cur_n_connected_interfaces = 0
            for i in range(len(self.if_components)):
                if np.any(np.linalg.norm(if_prox_vertices[i][:, None] - conn_comp.vertices, axis=2) < 1e-8):
                    cur_n_connected_interfaces += 1
            n_connected_interfaces = max(n_connected_interfaces, cur_n_connected_interfaces)
            
        # return n_connected_interfaces, n_connected_interfaces * 1.0 / len(self.if_components), n_connected_interfaces == len(self.if_components)
        res_dict = {'share of connected interfaces': n_connected_interfaces / len(self.if_components)}
        if self.debug_intermediate_shapes:
            res_dict['number of connected interfaces'] = n_connected_interfaces
            res_dict['are all interfaces connected'] = n_connected_interfaces == len(self.if_components)
            
        return res_dict
        
    def diversity_chamfer(self, mesh_list, div_inner_agg_fns=['sum'], div_outer_agg_fns=['sum'], div_ps=[0.5], chamfer_norm_orders=[2], n_samples=10000):
        sampled_points = []
        for i in range(len(mesh_list)):
            sampled_points.append(mesh_list[i].sample(n_samples))
            
        res_dict = {}
        for dist_norm_order in chamfer_norm_orders:
            for div_inner_agg_fn in div_inner_agg_fns:
                for div_outer_agg_fn in div_outer_agg_fns:
                    for div_p in div_ps:
                        diversity = diversity_metric(sampled_points, 
                                                        lambda x, y: two_sided_chamfer_divergence(x, y, dist_norm_order=dist_norm_order),
                                                        div_p, inner_agg_fn=div_inner_agg_fn, outer_agg_fn=div_outer_agg_fn)
                        res_dict[f'diversity_chamfer-order_{dist_norm_order}-inner_agg_{div_inner_agg_fn}-outer_agg_{div_outer_agg_fn}-p_{div_p}'] = diversity
        
        return res_dict
    
    def d_volume_of_disconnected_components_in_domain(self, mesh_model):
        '''
        Compute d-volume of all but the largest connected component.
        Then normalize by the total d-volume of the design region.
        '''
        components = trimesh.graph.split(mesh_model, only_watertight=False)
        
        if len(components) == 1:
            return {'d-volume of disconnected components in domain share': 0.0}

        # remove largest component
        components = sorted(components, key=lambda x: x.volume, reverse=True)[1:]
        
        total_disconnected_d_vol = sum([comp.volume for comp in components])
        disconnected_d_vol_share = total_disconnected_d_vol / self.mesh_bbox.volume
        return {'d-volume of disconnected components in domain share': disconnected_d_vol_share}
    
    def d_volume_of_disconnected_components_in_design_region(self, mesh_model):
        '''
        Compute the d-volume of all but the largest connected component inside the design region.
        Then normalize by the total d-volume of the design region.
        '''
        mesh_model_in_desreg = self.mesh_design_region.intersection(mesh_model)
        components = trimesh.graph.split(mesh_model_in_desreg, only_watertight=False)
        if len(components) == 1:
            return {'d-volume of disconnected components in design region share': 0.0}
        
        # remove largest component
        components = sorted(components, key=lambda x: x.volume, reverse=True)[1:]
        
        total_disconnected_d_vol = sum([comp.volume for comp in components])
        disconnected_d_vol_share = total_disconnected_d_vol / self.mesh_design_region.volume
        return {'d-volume of disconnected components in design region share': disconnected_d_vol_share}
    
    def curvature(self, netp, z, mesh_model, surf_pts=None, n_points=10000, n_steps=50, clip_max_value=1e6):
        assert mesh_model is not None or initial_points is not None, 'Either mesh_model or initial_points must be provided'
        assert surf_pts is not None or n_points is not None, 'Either surf_pts or n_points must be provided'
        
        if surf_pts is None:
            print(f'WARNING: surf_pts is None, initialize from mesh_model')
            initial_points = mesh_model.sample(n_points)
            surf_pts = torch.tensor(initial_points, requires_grad=True, dtype=torch.float32)
            netp = netp.detach()

            print(f'z.shape: {z.shape}')
            z = z[None, :]  # [1, z_dim] mimic batch dimension
            
            ## refine by flowing to 0-values of netp
            opt = torch.optim.Adam([surf_pts], lr=0.001)
            for i in range(n_steps):
                opt.zero_grad()
                y = netp.f(*tensor_product_xz(surf_pts, z)).squeeze(1)
                loss = y.square().mean()
                loss.backward()
                opt.step()
            
            print(f'mean y: {y.mean()}')
        
        ## compute curvature
        y_x_surf = netp.vf_x(*tensor_product_xz(surf_pts, z)).squeeze(1)
        y_xx_surf = netp.vf_xx(*tensor_product_xz(surf_pts, z)).squeeze(1)
        mean_clipped_curv = strain_curvature_loss(y_x_surf, y_xx_surf, clip_max_value=clip_max_value)
        mean_clipped_curv = np.array(mean_clipped_curv[0].detach().cpu())
        return {f'mean_clipped_{clip_max_value}_curvature': mean_clipped_curv}
        
        

            
            