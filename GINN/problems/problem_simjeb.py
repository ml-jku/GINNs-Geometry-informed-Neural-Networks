import os
from turtle import circle
import einops
import torch
import numpy as np
import trimesh

from GINN.problems.constraints import BoundingBox2DConstraint, CircleObstacle2D, CompositeInterface2D, Envelope2D, LineInterface2D, CompositeConstraint, SampleConstraint, SampleConstraintWithNormals, SampleEnvelope
from GINN.problems.problem_base import ProblemBase
from util.sample_utils import inflate_bounds
from util.misc import get_is_out_mask
from util.visualization.utils_mesh import get_meshgrid_in_domain



class ProblemSimjeb(ProblemBase):
    
    def __init__(self, config) -> None:
        super().__init__(config)
        device = self.config['device']
        
        self._envelope_constr = None
        self._interface_constraints = []
        self._normal_constraints = []        
        self._obstacle_constraints = []
        
       
        # see paper page 5 - https://arxiv.org/pdf/2105.03534.pdf
        # measurements given in 100s of millimeters
        bounds = torch.from_numpy(np.load(os.path.join(self.config['simjeb_root_dir'], 'bounds.npy'))).to(device).float()
        
        # scale_factor and translation_vector
        scale_factor = np.load(os.path.join(self.config['simjeb_root_dir'], 'scale_factor.npy'))
        center_for_translation = np.load(os.path.join(self.config['simjeb_root_dir'], 'center_for_translation.npy'))
        
        # load meshes
        self.mesh_if = trimesh.load(os.path.join(self.config['simjeb_root_dir'], 'interfaces.stl'))
        self.mesh_env = trimesh.load(os.path.join(self.config['simjeb_root_dir'], '411_for_envelope.obj'))
        
        # translate meshes
        self.mesh_if.apply_translation(-center_for_translation)
        self.mesh_env.apply_translation(-center_for_translation)
        
        # scale meshes
        self.mesh_if.apply_scale(1. / scale_factor)
        self.mesh_env.apply_scale(1. / scale_factor)
        
        ## load points
        pts_far_outside_env = torch.from_numpy(np.load(os.path.join(self.config['simjeb_root_dir'], 'pts_far_outside.npy'))).to(device).float()
        pts_on_envelope = torch.from_numpy(np.load(os.path.join(self.config['simjeb_root_dir'], 'pts_on_env.npy'))).to(device).float()
        pts_inside_envelope = torch.from_numpy(np.load(os.path.join(self.config['simjeb_root_dir'], 'pts_inside.npy'))).to(device).float()
        pts_outside_envelope = torch.from_numpy(np.load(os.path.join(self.config['simjeb_root_dir'], 'pts_outside.npy'))).to(device).float()
        interface_pts = torch.from_numpy(np.load(os.path.join(self.config['simjeb_root_dir'], 'interface_points.npy'))).to(device).float()
        interface_normals = torch.from_numpy(np.load(os.path.join(self.config['simjeb_root_dir'], 'interface_normals.npy'))).to(device).float()
        pts_around_interface = torch.from_numpy(np.load(os.path.join(self.config['simjeb_root_dir'], 'pts_around_interface_outside_env_10mm.npy'))).to(device).float()
                    
        # print(f'bounds: {bounds}')
        # print(f'pts_on_envelope: min x,y,z: {torch.min(pts_on_envelope, dim=0)[0]}, max x,y,z: {torch.max(pts_on_envelope, dim=0)[0]}')
        # print(f'pts_outside_envelope: min x,y,z: {torch.min(pts_outside_envelope, dim=0)[0]}, max x,y,z: {torch.max(pts_outside_envelope, dim=0)[0]}')
        # print(f'interface_pts: min x,y,z: {torch.min(interface_pts, dim=0)[0]}, max x,y,z: {torch.max(interface_pts, dim=0)[0]}')
        assert get_is_out_mask(pts_on_envelope, bounds).any() == False
        assert get_is_out_mask(interface_pts, bounds).any() == False
        
        self.config['bounds'] = bounds  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]        
        envelope = SampleEnvelope(pts_on_envelope=pts_on_envelope, pts_outside_envelope=pts_outside_envelope, sample_from=self.config['envelope_sample_from'])
        envelope_around_interface = SampleConstraint(sample_pts=pts_around_interface)
        pts_far_from_env_constraint = SampleConstraint(sample_pts=pts_far_outside_env)
        inside_envelope = SampleConstraint(sample_pts=pts_inside_envelope)
        domain = CompositeConstraint([inside_envelope])  ## TODO: test also with including outside envelope
        interface = SampleConstraintWithNormals(sample_pts=interface_pts, normals=interface_normals)

        self.constr_pts_dict = {
            # the envelope points are sampled uniformly from the 3 subsets
            'far_outside_envelope': pts_far_from_env_constraint.get_sampled_points(N=self.config['n_points_envelope'] // 3).cpu().numpy(),
            'envelope': envelope.get_sampled_points(N=self.config['n_points_envelope'] // 3).cpu().numpy(),
            'envelope_around_interface': envelope_around_interface.get_sampled_points(N=self.config['n_points_envelope'] // 3).cpu().numpy(),
            # other constraints
            'interface': interface.get_sampled_points(N=self.config['n_points_interfaces'])[0].cpu().numpy(),
            'domain': domain.get_sampled_points(N=self.config['n_points_domain']).cpu().numpy(),
        }
        
        self._envelope_constr = [envelope, envelope_around_interface, pts_far_from_env_constraint]
        self._interface_constraints = [interface]
        self._obstacle_constraints = None
        self._domain = domain
        
        self.bounds = config['bounds'].cpu()
        # ## For plotting
        # self.X0, self.X1, self.xs = get_meshgrid_in_domain(inflate_bounds(self.bounds, amount=0.2)) # inflate bounds for better visualization
        # self.xs = torch.tensor(self.xs, dtype=torch.float32, device=device)
            