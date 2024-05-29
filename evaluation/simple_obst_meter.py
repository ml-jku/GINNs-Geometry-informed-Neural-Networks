from matplotlib.pylab import f
import numpy as np
import shapely

class SimpleObstacleMeter:
    
    def __init__(self, 
                 bounds =  np.array([[-1, 1],[-0.5, 0.5]]),
                 envelope = np.array([[-.9, 0.9], [-0.4, 0.4]]),
                 obst_center = [0, 0],
                 obst_radius = 0.1,
                 interface_left = [[-0.9, -0.4], [-0.9, 0.4]],
                 interface_right = [[0.9, -0.4], [0.9, 0.4]],
                 ):
        
        self.bounds = bounds
        self.contour_bounds = shapely.geometry.box(*bounds.T.flatten())
        self.obstacle = shapely.geometry.Point(obst_center).buffer(obst_radius)
        self.contour_design_region = shapely.geometry.box(*envelope.T.flatten())
        ## remove obstacle from design region
        self.contour_design_region = self.contour_design_region.difference(self.obstacle)
        ## negative of design region for some metrics
        self.contour_design_region_neg = self.contour_bounds.difference(self.contour_design_region)
        
        self.if_left = shapely.geometry.LineString(interface_left)
        self.if_right = shapely.geometry.LineString(interface_right)
        
        self.interface_pt_samples = []
        num_points = 1000
        for if_line in [self.if_left, self.if_right]:
            interval_length = if_line.length / (num_points - 1)
            if_samples_interpolated = [if_line.interpolate(i * interval_length) for i in range(num_points)]
            self.interface_pt_samples.append(if_samples_interpolated)

        self.interface_np_samples = [np.array([[pt.x, pt.y] for pt in if_samples]) for if_samples in self.interface_pt_samples]

        self.domain_length = np.max(np.diff(self.bounds, axis=1))
        print(f'Domain length: {self.domain_length}')

    def get_all_metrics_as_dict(self, contour_model):
        metrics = {}
        metrics['d-volume'] = contour_model.area
        metrics['(d-1)-volume'] = contour_model.length
        print(f'Computing one-sided chamfer distance')
        metrics['one_sided_chamfer_distance'] = self.one_sided_chamfer_distance(contour_model)
        
        print(f'Computing Betti-0')
        metrics['Betti-0'] = self.betti_0(contour_model)
        metrics['Betti-0 inside design region'], _ = self.betti_0_in_design_region(contour_model)
        
        print(f'Computing volume outside envelope')
        metrics['d-volume outside design region'], metrics['d-volume outside design region share'], _ = self.dvol_outside_design_region(contour_model)
        
        print(f'Computing model intersect env surface')
        metrics['(d-1)-volume model intersect design region'], metrics['(d-1)-volume model intersect design region share'], _ = self.d_minus_1_vol_model_intersect_design_region(contour_model)
        
        print(f'Computing n connected interfaces')
        metrics['n_connected_interfaces'], metrics['share_connected_interfaces'], metrics['all_interfaces_connected'] = self.n_connected_interfaces(contour_model)
        
        return metrics
    
    def one_sided_chamfer_distance(self, contour_model):
        dist_to_if = [if_line.distance(contour_model) for if_line in [self.if_left, self.if_right]]
        one_sided_chamfer = np.mean(dist_to_if)
        return one_sided_chamfer
    
    def _get_components(self, contour_model):
        if isinstance(contour_model, shapely.geometry.Polygon):
            return [contour_model]
        elif isinstance(contour_model, shapely.geometry.MultiPolygon):
            res = [polyg for polyg in contour_model.geoms]
            return res                
        raise ValueError('contour_model must be of type Polygon or MultiPolygon')
    
    def betti_0(self, contour_model):
        return len(self._get_components(contour_model))
    
    def betti_0_in_design_region(self, contour_model):
        contour_model_inside_desreg = self.contour_design_region.intersection(contour_model)
        return len(self._get_components(contour_model_inside_desreg)), contour_model_inside_desreg
    
    def dvol_outside_design_region(self, contour_model):
        contour_model_outside_env = self.contour_design_region_neg.intersection(contour_model)
        outside_env_dvol = contour_model_outside_env.area
        outside_env_dvol_share = outside_env_dvol / self.contour_design_region_neg.area
        return outside_env_dvol, outside_env_dvol_share, contour_model_outside_env
    
    def d_minus_1_vol_model_intersect_design_region(self, contour_model):
        contour_model_design = self.contour_design_region.intersection(contour_model)
        contour_model_env_surface = self.contour_design_region_neg.intersection(contour_model_design)
        
        contour_model_env_surface_len = contour_model_env_surface.length
        length_share = contour_model_env_surface_len / self.contour_design_region.length
        return contour_model_env_surface_len, length_share, contour_model_env_surface
    
    def n_connected_interfaces(self, contour_model, eps=0.01, only_within_design_region=True):
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
            contour_model = self.contour_design_region.intersection(contour_model)
        
        pts_list = []
        for comp in self._get_components(contour_model):
            pts_list.extend([[pt[0], pt[1]] for pt in comp.exterior.coords])
        model_pts = np.array(pts_list)

        # get vertices of mesh eps-close to the interface
        if_prox_vertices = []
        for if_samples in self.interface_np_samples:
            ## distance via broadcasting is much faster than distance to mesh_if
            dist_vertices = np.linalg.norm(np.array(if_samples)[:,None] - model_pts, axis=2).min(axis=0)
            vertices_close_to_if = model_pts[dist_vertices < eps]
            if_prox_vertices.append(vertices_close_to_if)
        
        # for each connected component, check how many interfaces it contains, by checking if it contains at least one proximity vertex from each interface
        n_connected_interfaces = 0
        for conn_comp in self._get_components(contour_model):
            points_of_comp = np.array([[pt[0], pt[1]] for pt in conn_comp.exterior.coords])
            cur_n_connected_interfaces = 0
            for i in range(len(self.interface_np_samples)):
                if np.any(np.linalg.norm(if_prox_vertices[i][:, None] - points_of_comp, axis=2) < eps):
                    cur_n_connected_interfaces += 1
            n_connected_interfaces = max(n_connected_interfaces, cur_n_connected_interfaces)
            
        return n_connected_interfaces, n_connected_interfaces * 1.0 / len(self.interface_np_samples), n_connected_interfaces == len(self.interface_np_samples)
    
    def diversity_chamfer(self, contour_list, n_samples=1000):
        sampled_points = []
        for i in range(len(contour_list)):
            # sample points on the contour
            # if_samples_interpolated = [if_line.interpolate(i * interval_length) for i in range(num_points)]
            sampled_points.append(self._sample_points_on_contour(contour_list[i], n_samples))
            
        chamfer_sum = 0
        for i in range(len(contour_list)):
            for j in range(len(contour_list)):
                if i != j:
                    chamfer_sum += np.linalg.norm(sampled_points[i][:, None] - sampled_points[j], axis=2).min(axis=0).mean()
                    
        diversity =  chamfer_sum / (len(contour_list) * (len(contour_list) - 1))
        return diversity
    
    def _sample_points_on_contour(self, geometry, n_samples):
        def sample_from_line(line, num):
            interval = 1.0 / num
            return [line.interpolate(interval * i, normalized=True) for i in range(num)]
        
        total_length = geometry.length
        
        points = []
        for poly in self._get_components(geometry):
            points_for_this_poly = int(poly.length / total_length * n_samples)
            if points_for_this_poly == 0:
                points_for_this_poly = 1
            points.extend(sample_from_line(poly.exterior, points_for_this_poly))
            
        np_points = np.array([[pt.x, pt.y] for pt in points])
        return np_points