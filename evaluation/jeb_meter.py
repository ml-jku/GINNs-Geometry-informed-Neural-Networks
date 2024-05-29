import os
import numpy as np
import trimesh

class JebMeter:
    
    def _load_mesh_and_scale_and_center(self, mesh_path, center_for_translation, scale_factor):
        mesh = trimesh.load(mesh_path)
        mesh.apply_translation(-center_for_translation)
        mesh.apply_scale(1/scale_factor)
        return mesh
    
    def __init__(self, simjeb_root_dir, n_samples_at_if_for_chamfer=10000):
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
            self.if_samples.append(self.if_components[i].sample(1000))
            
        self.domain_length = np.max(np.diff(self.bounds, axis=1))
        print(f'Domain length: {self.domain_length}')
        
        
    
    def get_all_metrics_as_dict(self, mesh_model, eps_connected_interfaces=0.01):
        metrics = {}
        metrics['volume'] = mesh_model.volume
        metrics['surface_area'] = mesh_model.area
        print(f'Computing one-sided chamfer distance')
        metrics['one_sided_chamfer_distance'] = self.one_sided_chamfer_distance(mesh_model)
        
        print(f'Computing zeroth betti number')
        metrics['Betti-0'] = self.zeroth_betti_number(mesh_model)
        metrics['Betti-0 inside design region'], _ = self.zeroth_betti_number_within_envelope(mesh_model)
        
        print(f'Computing volume outside envelope')
        metrics['d-volume outside design region'], metrics['d-volume outside design region share'], _ = self.volume_outside_env(mesh_model)
        
        print(f'Computing model intersect env surface')
        metrics['(d-1)-volume model intersect design region'], metrics['(d-1)-volume model intersect design region share'], _ = self.model_intersect_env_surface(mesh_model)
        
        print(f'Computing n connected interfaces')
        metrics['n_connected_interfaces'], metrics['share_connected_interfaces'], metrics['all_interfaces_connected'] = self.n_connected_interfaces(mesh_model, eps_connected_interfaces)
        return metrics
        
    def one_sided_chamfer_distance(self, mesh_model):
        dist_to_if = mesh_model.nearest.on_surface(self.pts_at_if)[1]
        one_sided_chamfer = dist_to_if.mean()
        return one_sided_chamfer
    
    def zeroth_betti_number(self, mesh_model):
        components = trimesh.graph.split(mesh_model, only_watertight=False)
        return len(components)
    
    def zeroth_betti_number_within_envelope(self, mesh_model):
        assert mesh_model.is_watertight, 'Model must be watertight'
        
        mesh_model_inside_env = self.mesh_design_region.intersection(mesh_model)
        components = trimesh.graph.split(mesh_model_inside_env, only_watertight=False)
        return len(components), mesh_model_inside_env
    
    def volume_outside_env(self, mesh_model):
        assert mesh_model.is_watertight, 'Model must be watertight'
        mesh_model_outside_env = self.mesh_design_region_neg.intersection(mesh_model)
        outside_env_volume = mesh_model_outside_env.volume
        vol_share = outside_env_volume / self.volume_design_region_neg
        return outside_env_volume, vol_share, mesh_model_outside_env
    
    def model_intersect_env_surface(self, mesh_model):
        assert mesh_model.is_watertight, 'Model must be watertight'
        
        mesh_model_design = self.mesh_design_region.intersection(mesh_model)
        mesh_model_env_surface = self.mesh_design_region_neg.intersection(mesh_model_design)
        
        mesh_model_env_surface_area = mesh_model_env_surface.area
        area_share = mesh_model_env_surface_area / self.d_minus_1_volume_design_region_boundary
        return mesh_model_env_surface_area, area_share, mesh_model_env_surface
    
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
            
        return n_connected_interfaces, n_connected_interfaces * 1.0 / len(self.if_components), n_connected_interfaces == len(self.if_components)
    
    def diversity_chamfer(self, mesh_list, n_samples=1000):
        sampled_points = []
        for i in range(len(mesh_list)):
            sampled_points.append(mesh_list[i].sample(n_samples))
            
        chamfer_sum = 0
        for i in range(len(mesh_list)):
            for j in range(len(mesh_list)):
                if i != j:
                    chamfer_sum += np.linalg.norm(sampled_points[i][:, None] - sampled_points[j], axis=2).min(axis=0).mean()
                    
        return chamfer_sum / (len(mesh_list) * (len(mesh_list) - 1))