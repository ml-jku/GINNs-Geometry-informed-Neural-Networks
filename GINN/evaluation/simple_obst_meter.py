import logging
import time
from matplotlib import contour
from matplotlib.pylab import f
import numpy as np
import shapely
from sympy import div

from GINN.problems.problem_base import ProblemBase
from GINN.evaluation.metrics_utils import two_sided_chamfer_divergence, chamfer_diversity_metric_vec, diversity_metric
from models.net_w_partials import NetWithPartials
from util.visualization.utils_mesh import get_2d_contour_for_latent, sample_pts_interior_and_exterior
from shapely.geometry import MultiPolygon

class SimpleObstacleMeter:
    
    @staticmethod
    def create_from_problem_sampler(ps: ProblemBase, **kwargs):
        return SimpleObstacleMeter(bounds=ps.bounds.cpu().numpy(),
                                   envelope=ps.envelope,
                                   obst_center=ps.obst_1_center,
                                   obst_radius=ps.obst_1_radius,
                                   interface_left=ps.interface_left,
                                   interface_right=ps.interface_right, 
                                   **kwargs)
    
    def __init__(self,
                 bounds =  np.array([[-1, 1],[-0.5, 0.5]]),
                 envelope = np.array([[-.9, 0.9], [-0.4, 0.4]]),
                 obst_center = [0, 0],
                 obst_radius = 0.1,
                 interface_left = np.array([[-0.9, -0.4], [-0.9, 0.4]]),
                 interface_right = np.array([[0.9, -0.4], [0.9, 0.4]]),
                 debug_intermediate_shapes=False,
                 metrics_diversity_inner_agg_fns=['sum'],
                 metrics_diversity_outer_agg_fns=['sum'],
                 metrics_diversity_ps=[0.5],
                 metrics_chamfer_orders=[2],
                 metrics_diversity_n_samples=1000,
                 chamfer_metrics_vectorize=True,
                 **kwargs):
        
        self.bounds = bounds
        self.debug_intermediate_shapes = debug_intermediate_shapes
        self.metrics_diversity_inner_agg_fns = metrics_diversity_inner_agg_fns
        self.metrics_diversity_outer_agg_fns = metrics_diversity_outer_agg_fns
        self.metrics_diversity_ps = metrics_diversity_ps
        self.metrics_chamfer_orders = metrics_chamfer_orders
        self.metrics_diversity_n_samples = metrics_diversity_n_samples
        self.chamfer_metrics_vectorize = chamfer_metrics_vectorize
        
        self.logger = logging.getLogger('SimpleObstacleMeter')
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
        self.logger.debug(f'Domain length: {self.domain_length}')

    def get_average_metrics_as_dict(self, contour_models, prefix=''):
        
        metrics_dicts_list = []
        for contour in contour_models:
            metrics = self.get_all_metrics_as_dict(contour)
            metrics_dicts_list.append(metrics)
        
        # print(f'(d-1)-volume model intersect design region share: {[m["(d-1)-volume model intersect design region share"] for m in metrics_dicts_list]}')
        # print(f'd-volume outside design region share: {[m["d-volume outside design region share"] for m in metrics_dicts_list]}')
        
        avg_metrics = {}
        for key in metrics_dicts_list[0].keys():
            avg_metrics['avg_' + key] = np.mean([m[key] for m in metrics_dicts_list])
            avg_metrics['std_' + key] = np.std([m[key] for m in metrics_dicts_list])
        
        # print(f'(d-1)-volume model intersect design region share_avg: {avg_metrics["(d-1)-volume model intersect design region share_avg"]}')
        # print(f'd-volume outside design region share_avg: {avg_metrics["d-volume outside design region share_avg"]}')
        
        ## diversity metrics
        if len(contour_models) > 1:
            # start_t = time.time()
            avg_metrics.update(self.diversity_chamfer(contour_models, 
                                                div_inner_agg_fns=self.metrics_diversity_inner_agg_fns,
                                                div_outer_agg_fns=self.metrics_diversity_outer_agg_fns,
                                                div_ps=self.metrics_diversity_ps,
                                                chamfer_norm_orders=self.metrics_chamfer_orders,
                                                n_samples=self.metrics_diversity_n_samples,
                                                vectorize=self.chamfer_metrics_vectorize))
            # print(f'needed {time.time() - start_t:0.2f} seconds for diversity')
            # print(f'avg_metrics: {avg_metrics}')
        
        ## update keys with prefix
        for key in list(avg_metrics.keys()):
            avg_metrics[prefix + key] = avg_metrics.pop(key)
        
        self.logger.info(f'Finished computing all metrics')
        return avg_metrics
        
    
    def get_all_metrics_as_dict(self, contour_model:MultiPolygon):
    
        metrics = {}
        metrics['d-volume'] = contour_model.area
        metrics['(d-1)-volume'] = contour_model.length
        
        self.logger.debug(f'Computing one-sided chamfer distance')
        metrics.update(self.one_sided_chamfer_distance(contour_model))
        
        self.logger.debug(f'Computing Betti-0') 
        metrics.update(self.betti_0(contour_model))
        metrics.update(self.betti_0_in_design_region(contour_model))
        
        self.logger.debug(f'Computing volume outside envelope')
        metrics.update(self.dvol_outside_design_region(contour_model))
        
        self.logger.debug(f'Computing model intersect env surface')
        metrics.update(self.d_minus_1_vol_model_intersect_design_region(contour_model))
        
        self.logger.debug(f'Computing n connected interfaces')
        metrics.update(self.n_connected_interfaces(contour_model))
        
        self.logger.debug(f'Computing size of disconnected components')
        metrics.update(self.d_volume_of_disconnected_components_in_domain(contour_model))
        metrics.update(self.d_volume_of_disconnected_components_in_design_region(contour_model))
        
        return metrics
    
    def _get_components(self, contour_model):
        if isinstance(contour_model, shapely.geometry.Polygon):
            return [contour_model]
        elif isinstance(contour_model, shapely.geometry.MultiPolygon):
            return contour_model.geoms
        raise ValueError('contour_model must be either a Polygon or a MultiPolygon')
        
    
    def one_sided_chamfer_distance(self, contour_model):
        dist_to_if = [if_line.distance(contour_model) for if_line in [self.if_left, self.if_right]]
        one_sided_chamfer = np.mean(dist_to_if)
        return {'one_sided_chamfer distance to interface': one_sided_chamfer}
    
    def betti_0(self, contour_model):
        n_components = len(self._get_components(contour_model))
        return {'betti_0': n_components}
    
    def betti_0_in_design_region(self, contour_model):
        contour_model_inside_desreg = self.contour_design_region.intersection(contour_model)
        n_components = len(self._get_components(contour_model_inside_desreg))
        res_dict = {'betti_0 inside design region': n_components}
        if self.debug_intermediate_shapes:
            res_dict['contour_model_inside_desreg'] = contour_model_inside_desreg
        return res_dict
    
    def dvol_outside_design_region(self, contour_model):
        contour_model_outside_env = self.contour_design_region_neg.intersection(contour_model)
        outside_env_dvol = contour_model_outside_env.area
        outside_env_dvol_share = outside_env_dvol / self.contour_design_region_neg.area
        
        res_dict = {'d-volume outside design region share': outside_env_dvol_share}
        if self.debug_intermediate_shapes:
            res_dict['d-volume outside design region'] = outside_env_dvol
            res_dict['contour_model_outside_env'] = contour_model_outside_env
        return res_dict
    
    def d_minus_1_vol_model_intersect_design_region(self, contour_model):
        contour_model_design = self.contour_design_region.intersection(contour_model)
        contour_model_env_surface = self.contour_design_region_neg.intersection(contour_model_design)
        
        contour_model_env_surface_len = contour_model_env_surface.length
        length_share = contour_model_env_surface_len / self.contour_design_region.length
        
        res_dict = {'(d-1)-volume model intersect design region share': length_share}
        if self.debug_intermediate_shapes:
            res_dict['(d-1)-volume model intersect design region'] = contour_model_env_surface_len
            res_dict['contour_model_env_surface'] = contour_model_env_surface
            
        return res_dict
    
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
        
        if contour_model.is_empty:
            return {'share of connected interfaces': 0.0}
        
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
        components = self._get_components(contour_model)
        for conn_comp in components:
            points_of_comp = np.array([[pt[0], pt[1]] for pt in conn_comp.exterior.coords])
            for interior_ring in conn_comp.interiors:
                points_of_comp = np.concatenate([points_of_comp, np.array([[pt[0], pt[1]] for pt in interior_ring.coords])], axis=0)
                
            cur_n_connected_interfaces = 0
            for i in range(len(self.interface_np_samples)):
                if np.any(np.linalg.norm(if_prox_vertices[i][:, None] - points_of_comp, axis=2) < eps):
                    cur_n_connected_interfaces += 1
            n_connected_interfaces = max(n_connected_interfaces, cur_n_connected_interfaces)
        
        res_dict = {'share of connected interfaces': n_connected_interfaces / len(components)}
        if self.debug_intermediate_shapes:
            res_dict['number of connected interfaces'] = n_connected_interfaces
            res_dict['are all interfaces connected'] = n_connected_interfaces == len(self.interface_np_samples)
        return res_dict        
    
    def d_volume_of_disconnected_components_in_domain(self, contour_model):
        '''
        Compute d-volume of all but the largest connected component.
        Then normalize by the total d-volume of the design region.
        '''
        components = self._get_components(contour_model)
        
        if len(components) == 1:
            return {'d-volume of disconnected components in domain share': 0.0}

        # remove largest component
        components = sorted(components, key=lambda x: x.area, reverse=True)[1:]
        
        total_disconnected_d_vol = sum([comp.area for comp in components])
        disconnected_d_vol_share = total_disconnected_d_vol / self.contour_bounds.area
        return {'d-volume of disconnected components in domain share': disconnected_d_vol_share}
    
    def d_volume_of_disconnected_components_in_design_region(self, contour_model):
        '''
        Compute the d-volume of all but the largest connected component inside the design region.
        Then normalize by the total d-volume of the design region.
        '''
        contour_model_in_desreg = self.contour_design_region.intersection(contour_model)
        components = self._get_components(contour_model_in_desreg)
        if len(components) in [0, 1]:
            return {'d-volume of disconnected components in design region share': 0.0}
        
        # remove largest component
        components = sorted(components, key=lambda x: x.area, reverse=True)[1:]
        
        total_disconnected_d_vol = sum([comp.area for comp in components])
        disconnected_d_vol_share = total_disconnected_d_vol / self.contour_design_region.area
        return {'d-volume of disconnected components in design region share': disconnected_d_vol_share}
        
    
    def diversity_chamfer(self, contour_list, div_inner_agg_fns=['sum'], div_outer_agg_fns=['sum'], div_ps=[0.5], chamfer_norm_orders=[2], n_samples=1000, vectorize=False):
        '''
        Compute the diversity metric based on chamfer distances between the contours.
        :param contour_list: list of shapely.geometry.MultiPolygon
        :param dist_norm_orders: list of int, orders of the norm used for the distance function
        :param div_inner_agg_fns: list of str, aggregation functions for the inner aggregation
        :param div_outer_agg_fns: list of str, aggregation functions for the outer aggregation
        :param div_ps: list of float, p-values for the diversity metric
        '''
        sampled_points = []
        for i in range(len(contour_list)):
            # sample points on the contour
            # if_samples_interpolated = [if_line.interpolate(i * interval_length) for i in range(num_points)]
            success, samples = sample_pts_interior_and_exterior(contour_list[i], n_samples)
            if not success:
                return {'diversity_chamfer': float('inf')}
            sampled_points.append(samples)
        
        if vectorize:
            pts_mat = np.stack(sampled_points, axis=0)  # [n_shapes, n_points, n_dims]
        
        res_dict = {}
        for dist_norm_order in chamfer_norm_orders:
            for div_inner_agg_fn in div_inner_agg_fns:
                for div_outer_agg_fn in div_outer_agg_fns:
                    for div_p in div_ps:
                        if vectorize:
                            diversity = chamfer_diversity_metric_vec(pts_mat, p=div_p, row_agg_fn=div_inner_agg_fn, col_agg_fn=div_outer_agg_fn, norm_order=dist_norm_order)
                        else:
                            diversity = diversity_metric(sampled_points, 
                                                        lambda x, y: two_sided_chamfer_divergence(x, y, dist_norm_order=dist_norm_order),
                                                        div_p, inner_agg_fn=div_inner_agg_fn, outer_agg_fn=div_outer_agg_fn)
                        res_dict[f'diversity_chamfer-order_{dist_norm_order}-inner_agg_{div_inner_agg_fn}-outer_agg_{div_outer_agg_fn}-p_{div_p}'] = diversity
        
        return res_dict
