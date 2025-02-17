import os
from turtle import circle
import einops
import torch
import numpy as np
import trimesh

from GINN.problems.constraints import BoundingBox2DConstraint, DiskConstraint, CompositeInterface, RectangleEnvelope, LineInterface2D, CompositeConstraint, SampleConstraint, SampleConstraintWithNormals, SampleEnvelope
from GINN.problems.problem_base import ProblemBase
from util.model_utils import tensor_product_xz
from models.point_wrapper import PointWrapper
from util.visualization.utils_mesh import get_watertight_mesh_for_latent
from util.misc import get_is_out_mask
from util.visualization.utils_mesh import get_2d_contour_for_grid, get_meshgrid_for_marching_squares, get_meshgrid_in_domain

def t_(x):
    return torch.tensor(x, dtype=torch.float32)

class ProblemObstacle(ProblemBase):
    
    def __init__(self, nx, 
                 n_points_envelope, 
                 n_points_interfaces, 
                 n_points_obstacles,
                 n_points_domain,
                 **kwargs) -> None:
        super().__init__(nx=nx)
        
        self.n_points_envelope = n_points_envelope
        self.n_points_interfaces = n_points_interfaces
        self.n_points_obstacles = n_points_obstacles
        self.n_points_domain = n_points_domain
        self._envelope_constr = None
        self._interface_constraints = []
        self._normal_constraints = []        
        self._obstacle_constraints = []
        
        self.bounds = t_([[-1, 1],[-0.5, 0.5]])  # [[x_min, x_max], [y_min, y_max]]
        self.envelope_np = np.array([[-.9, 0.9], [-0.4, 0.4]])
        self.envelope = t_(self.envelope_np)
        self.obst_1_center = [0, 0]
        self.obst_1_radius = 0.1
        self.interface_left = np.array([[-0.9, -0.4], [-0.9, 0.4]])
        self.interface_right = np.array([[0.9, -0.4], [0.9, 0.4]])
        
        outside_envelop = RectangleEnvelope(env_bbox=self.envelope)
        # sample only within envelope
        inside_envelope = BoundingBox2DConstraint(bbox=self.envelope)
        domain = BoundingBox2DConstraint(bbox=self.config['bounds'])
        
        # TODO: normals should be computed from the interface definition
        l_target_normal = t_([-1.0, 0.0])
        r_target_normal = t_([1.0, 0.0])
        l_bc = LineInterface2D(start=t_(self.interface_left[0]), 
                                end=t_(self.interface_left[1]), 
                                target_normal=l_target_normal)
        r_bc = LineInterface2D(start=t_(self.interface_right[0]),
                                    end=t_(self.interface_right[1]),
                                    target_normal=r_target_normal)
        all_interfaces = CompositeInterface([l_bc, r_bc])
        
        circle_obstacle_1 = DiskConstraint(center=t_(self.obst_1_center), radius=t_(self.obst_1_radius))
        
        # sample once and keep the points; these are used for plotting
        self.constr_pts_dict = {
            'outside_envelope': outside_envelop.get_sampled_points(N=self.config['n_points_envelope']).cpu().numpy().T,
            'interface': all_interfaces.get_sampled_points(N=self.config['n_points_interfaces'])[0].cpu().numpy().T,
            'obstacles': circle_obstacle_1.get_sampled_points(N=self.config['n_points_obstacles']).cpu().numpy().T,
            'inside_envelope': inside_envelope.get_sampled_points(N=self.config['n_points_domain']).cpu().numpy().T,
            'domain': inside_envelope.get_sampled_points(N=self.config['n_points_domain']).cpu().numpy().T,
        }
        
        # save the constraints
        self._envelope_constr = [outside_envelop]
        self._interface_constraints = [l_bc, r_bc]
        self._obstacle_constraints = [circle_obstacle_1]
        self._inside_envelope = inside_envelope
        self._domain = domain
        
        ##
        self.X0_ms, _, xs_ms = get_meshgrid_for_marching_squares(self.config['bounds'].cpu().numpy())
        self.xs_ms = torch.tensor(xs_ms, dtype=torch.float32, device=self.config['device'])
        
        ## For plotting
        self.X0, self.X1, self.xs = get_meshgrid_in_domain(self.bounds.cpu(), self.config['plot_2d_resolution'], self.config['plot_2d_resolution']) # inflate bounds for better visualization
        self.xs = torch.tensor(self.xs, dtype=torch.float32, device=torch.get_default_device())
        
    def is_inside_envelope(self, a: torch.Tensor):
        """Get mask for points which are inside the envelope"""
        is_inside_mask = (a[:, 0] >= self.envelope[0, 0]) & (a[:, 0] <= self.envelope[0, 1]) & \
            (a[:, 1] >= self.envelope[1, 0]) & (a[:, 1] <= self.envelope[1, 1])
        return is_inside_mask