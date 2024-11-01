import os
from turtle import circle
import einops
import torch
import numpy as np
import trimesh

from GINN.problems.constraints import BoundingBox2DConstraint, CircleObstacle2D, CompositeInterface2D, Envelope2D, LineInterface2D, CompositeConstraint, SampleConstraint, SampleConstraintWithNormals, SampleEnvelope
from GINN.problems.problem_base import ProblemBase
from util.model_utils import tensor_product_xz
from models.point_wrapper import PointWrapper
from util.visualization.utils_mesh import get_watertight_mesh_for_latent
from util.misc import get_is_out_mask
from util.visualization.utils_mesh import get_2d_contour_for_grid, get_meshgrid_for_marching_squares, get_meshgrid_in_domain, get_mesh

def t_(x):
    return torch.tensor(x, dtype=torch.float32)

class ProblemObstacle(ProblemBase):
    
    def __init__(self, config) -> None:
        super().__init__(config)
        device = self.config['device']
        
        self._envelope_constr = None
        self._interface_constraints = []
        self._normal_constraints = []        
        self._obstacle_constraints = []
        
        self.config['bounds'] = t_([[-1, 1],[-0.5, 0.5]])  # [[x_min, x_max], [y_min, y_max]]
        self.envelope = np.array([[-.9, 0.9], [-0.4, 0.4]])
        self.obst_1_center = [0, 0]
        self.obst_1_radius = 0.1
        self.interface_left = np.array([[-0.9, -0.4], [-0.9, 0.4]])
        self.interface_right = np.array([[0.9, -0.4], [0.9, 0.4]])
        
        envelope = Envelope2D(env_bbox=t_(self.envelope), bounds=self.config['bounds'], device=device, sample_from=self.config['envelope_sample_from'])
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
        all_interfaces = CompositeInterface2D([l_bc, r_bc])
        
        circle_obstacle_1 = CircleObstacle2D(center=t_(self.obst_1_center), radius=t_(self.obst_1_radius))
        
        # sample once and keep the points; these are used for plotting
        self.constr_pts_dict = {
            'envelope': envelope.get_sampled_points(N=self.config['n_points_envelope']).cpu().numpy().T,
            'interface': all_interfaces.get_sampled_points(N=self.config['n_points_interfaces'])[0].cpu().numpy().T,
            'obstacles': circle_obstacle_1.get_sampled_points(N=self.config['n_points_obstacles']).cpu().numpy().T,
            'domain': domain.get_sampled_points(N=self.config['n_points_domain']).cpu().numpy().T,
        }
        
        # save the constraints
        self._envelope_constr = [envelope]
        self._interface_constraints = [l_bc, r_bc]
        self._obstacle_constraints = [circle_obstacle_1]
        self._domain = domain
        
        ##
        self.X0_ms, _, xs_ms = get_meshgrid_for_marching_squares(self.config['bounds'].cpu().numpy())
        self.xs_ms = torch.tensor(xs_ms, dtype=torch.float32, device=self.config['device'])
        
        self.bounds = config['bounds'].cpu()
        ## For plotting
        self.X0, self.X1, self.xs = get_meshgrid_in_domain(self.bounds) # inflate bounds for better visualization
        self.xs = torch.tensor(self.xs, dtype=torch.float32, device=device)