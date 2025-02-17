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
from util.visualization.utils_mesh import get_meshgrid_for_marching_squares, get_meshgrid_in_domain

def t_(x):
    return torch.tensor(x, dtype=torch.float32)

class ProblemCantilever2d(ProblemBase):
    
    def __init__(self, 
                 nx, 
                 width, 
                 height, 
                 n_points_envelope, 
                 n_points_interfaces, 
                 n_points_domain, 
                 plot_2d_resolution,
                 center_scale_coords,
                 nf_is_density=False,
                 **kwargs) -> None:
        super().__init__(nx=nx)
        
        self.n_points_envelope = n_points_envelope
        self.n_points_interfaces = n_points_interfaces
        self.n_points_domain = n_points_domain
        
        self._envelope_constr = None
        self._interface_constraints = []
        self._normal_constraints = []
        self._obstacle_constraints = []
        
        x_max = width
        x_min = 0
        y_max = height
        y_min = 0        
        expansion_factor = 1.00001  # 1 means no expansion

        self.envelope_unnormalized = t_([[x_min, x_max], [y_min, y_max]]) # [[x_min, x_max], [y_min, y_max]]
        self.bounds_unnormalized = self.envelope_unnormalized * expansion_factor
        self.interface_left = t_([[x_min, y_min], [x_min, y_max]])
        self.interface_right = t_([[x_max, y_min], [x_max, y_max]])
        
        if center_scale_coords:
            # make longest side go from -1 to 1
            normalize_factor = torch.max(torch.abs(self.bounds_unnormalized)) / 2
            normalize_center = self.bounds_unnormalized[:, 1] / 2
            self.envelope = (self.envelope_unnormalized - normalize_center.unsqueeze(1)) / normalize_factor
            self.bounds = (self.bounds_unnormalized - normalize_center.unsqueeze(1)) / normalize_factor
            self.interface_left = (self.interface_left - normalize_center) / normalize_factor
            self.interface_right = (self.interface_right - normalize_center) / normalize_factor
        else:
            self.bounds = self.bounds_unnormalized
            self.envelope = self.envelope_unnormalized
        
        envelope = RectangleEnvelope(env_bbox=self.envelope, bounds=self.bounds, sample_from='exterior')
        inside_envelope = RectangleEnvelope(env_bbox=self.envelope, bounds=self.bounds, sample_from='interior')
        domain = BoundingBox2DConstraint(bbox=self.bounds)
        
        l_target_normal = t_([-1.0, 0.0])
        r_target_normal = t_([1.0, 0.0])
        # the normals are flipped if nf_is_density is as opposed to SDF
        if nf_is_density:
            l_target_normal = -l_target_normal
            r_target_normal = -r_target_normal
        l_bc = LineInterface2D(start=t_(self.interface_left[0]), 
                                end=t_(self.interface_left[1]), 
                                target_normal=l_target_normal)
        r_bc = LineInterface2D(start=t_(self.interface_right[0]),
                                    end=t_(self.interface_right[1]),
                                    target_normal=r_target_normal)
        all_interfaces = CompositeInterface([l_bc, r_bc])
        
        # sample once and keep the points; these are used for plotting
        self.constr_pts_dict = {
            'outside_envelope': envelope.get_sampled_points(N=self.n_points_envelope).cpu().numpy().T,
            'interface': all_interfaces.get_sampled_points(N=self.n_points_interfaces)[0].cpu().numpy().T,
            'inside_envelope': inside_envelope.get_sampled_points(N=self.n_points_domain).cpu().numpy().T,
            'domain': domain.get_sampled_points(N=self.n_points_domain).cpu().numpy().T
        }
        
        # save the constraints
        self._envelope_constr = [envelope]
        self._interface_constraints = [l_bc, r_bc]
        self._obstacle_constraints = None
        self._inside_envelope = inside_envelope
        self._domain = domain
        
        ## for what?
        self.X0_ms, _, xs_ms = get_meshgrid_for_marching_squares(self.bounds.cpu().numpy())
        self.xs_ms = torch.tensor(xs_ms, dtype=torch.float32)
        
        ## For plotting
        self.X0, self.X1, self.xs = get_meshgrid_in_domain(self.bounds.cpu(), plot_2d_resolution, plot_2d_resolution) # inflate bounds for better visualization
        self.xs = torch.tensor(self.xs, dtype=torch.float32, device=torch.get_default_device())
        
    def is_inside_envelope(self, a: torch.Tensor):
        """Get mask for points which are inside the envelope"""
        is_inside_mask = (a[:, 0] >= self.envelope[0, 0]) & (a[:, 0] <= self.envelope[0, 1]) & \
            (a[:, 1] >= self.envelope[1, 0]) & (a[:, 1] <= self.envelope[1, 1])
        return is_inside_mask
    