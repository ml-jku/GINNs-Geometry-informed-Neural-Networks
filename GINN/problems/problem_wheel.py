import os
from turtle import circle
import einops
import torch
import numpy as np
import trimesh

from GINN.problems.constraints import BoundingBox2DConstraint, CircleConstraintWithNormals, DiskConstraint, CompositeInterface, RectangleEnvelope, LineInterface2D, CompositeConstraint, RingConstraint, RingEnvelope, RotationSymmetricRingConstraint, SampleConstraint, SampleConstraintWithNormals, SampleEnvelope
from GINN.problems.problem_base import ProblemBase
from util.model_utils import tensor_product_xz
from models.point_wrapper import PointWrapper
from util.visualization.utils_mesh import get_watertight_mesh_for_latent
from util.misc import get_is_out_mask
from util.visualization.utils_mesh import get_2d_contour_for_grid, get_meshgrid_for_marching_squares, get_meshgrid_in_domain

def t_(x):
    return torch.tensor(x, dtype=torch.float32)

class ProblemWheel(ProblemBase):
    
    def __init__(self, nx, n_points_envelope, n_points_interfaces, n_points_domain, rotation_n_cycles,
                 plot_2d_resolution,
                 n_points_rotation_symmetric=None, **kwargs) -> None:
        super().__init__(nx=nx)
        
        self.n_points_envelope = n_points_envelope
        self.n_points_interfaces = n_points_interfaces
        self.n_points_domain = n_points_domain
        self.n_points_rotation_symmetric = n_points_rotation_symmetric
        
        self._envelope_constr = None
        self._interface_constraints = []
        self._normal_constraints = []        
        self._obstacle_constraints = []
        
        self.bounds = t_([[-1, 1],[-1, 1]])  # [[x_min, x_max], [y_min, y_max]]
        self.envelope = np.array([[-.9, 0.9], [-0.4, 0.4]])
        self.center = [0, 0]
        self.r1 = 0.2
        self.r2 = 0.8
        
        envelope = RingEnvelope(center=t_(self.center), r1=t_(self.r1), r2=t_(self.r2), bounds=self.bounds, sample_from='exterior')
        domain = BoundingBox2DConstraint(bbox=self.bounds)
        inside_envelope = RingEnvelope(center=t_(self.center), r1=t_(self.r1), r2=t_(self.r2), bounds=self.bounds, sample_from='interior')
        
        inner_if = CircleConstraintWithNormals(position=t_(self.center), radius=t_(self.r1), normal_inwards=True)
        outer_if = CircleConstraintWithNormals(position=t_(self.center), radius=t_(self.r2), normal_inwards=False)
        all_interfaces = CompositeInterface([inner_if, outer_if])
        
        self.rotation_symmetric_constraint = RotationSymmetricRingConstraint(center=t_(self.center), r1=t_(self.r1), r2=t_(self.r2), n_cycles=rotation_n_cycles)
        
        # sample once and keep the points; these are used for plotting
        self.constr_pts_dict = {
            'outside_envelope': envelope.get_sampled_points(N=self.n_points_envelope).cpu().numpy().T,
            'interface': all_interfaces.get_sampled_points(N=self.n_points_interfaces)[0].cpu().numpy().T,
            'inside_envelope': inside_envelope.get_sampled_points(N=self.n_points_domain).cpu().numpy().T,
            'domain': domain.get_sampled_points(N=self.n_points_domain).cpu().numpy().T
        }
        
        # save the constraints
        self._envelope_constr = [envelope]
        self._interface_constraints = [inner_if, outer_if]
        self._obstacle_constraints = None
        self._inside_envelope = inside_envelope
        self._domain = domain
        
        ##
        self.X0_ms, _, xs_ms = get_meshgrid_for_marching_squares(self.bounds.cpu().numpy())
        self.xs_ms = torch.tensor(xs_ms, dtype=torch.float32)
        
        ## For plotting
        self.X0, self.X1, self.xs = get_meshgrid_in_domain(self.bounds.cpu().numpy(), plot_2d_resolution, plot_2d_resolution) # inflate bounds for better visualization
        self.xs = torch.tensor(self.xs, dtype=torch.float32)
        
    def is_inside_envelope(self, a: torch.Tensor):
        """Get mask for points which are inside the envelope"""
        dist_from_center = torch.norm(a - t_(self.center), dim=1)
        is_inside_mask = (dist_from_center >= self.r1) & (dist_from_center <= self.r2)
        return is_inside_mask