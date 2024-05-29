import einops
import torch
import numpy as np
import trimesh

from GINN.geometry.constraints import BoundingBox2DConstraint, CircleObstacle2D, CompositeInterface2D, Envelope2D, LineInterface2D, CompositeConstraint, SampleConstraint, SampleConstraintWithNormals, SampleEnvelope
from models.model_utils import tensor_product_xz
from models.point_wrapper import PointWrapper
from utils import get_is_out_mask
from visualization.utils_mesh import get_meshgrid_in_domain, get_mesh

class ProblemSampler():
    
    def __init__(self, config) -> None:
        self.config = config
        device = self.config['device']
        
        self._envelope_constr = None
        self._interface_constraints = []
        self._normal_constraints = []        
        self._obstacle_constraints = []
        
        if self.config['problem'] == 'simple_2d':
            self.config['bounds'] = torch.tensor([[-1, 1],[-0.5, 0.5]])  # [[x_min, x_max], [y_min, y_max]]
            envelope = Envelope2D(env_bbox=torch.tensor([[-.9, 0.9], [-0.4, 0.4]]), bounds=self.config['bounds'], device=device, sample_from=self.config['envelope_sample_from'])
            domain = BoundingBox2DConstraint(bbox=self.config['bounds'])
            
            l_target_normal = torch.tensor([-1.0, 0.0])
            r_target_normal = torch.tensor([1.0, 0.0])
            l_bc = LineInterface2D(start=torch.tensor([-.9, -.4]), end=torch.tensor([-.9, .4]), target_normal=l_target_normal)
            r_bc = LineInterface2D(start=torch.tensor([.9, -.4]), end=torch.tensor([.9, .4]), target_normal=r_target_normal)
            all_interfaces = CompositeInterface2D([l_bc, r_bc])
            
            circle_obstacle = CircleObstacle2D(center=torch.tensor([0.0, 0.0]), radius=torch.tensor(0.1))
            
            # sample once and keep the points; these are used for plotting
            self.constr_pts_dict = {
                'envelope': envelope.get_sampled_points(N=self.config['n_points_envelope']).cpu().numpy().T,
                'interface': all_interfaces.get_sampled_points(N=self.config['n_points_interfaces'])[0].cpu().numpy().T,
                'obstacles': circle_obstacle.get_sampled_points(N=self.config['n_points_obstacles']).cpu().numpy().T,
                'domain': domain.get_sampled_points(N=self.config['n_points_domain']).cpu().numpy().T,
            }
            
            # save the constraints
            self._envelope_constr = [envelope]
            self._interface_constraints = [l_bc, r_bc]
            self._obstacle_constraints = [circle_obstacle]
            self._domain = domain
            
        elif self.config['problem'] == 'pipes':
            self.config['bounds'] = torch.tensor([[-0.1, 1.6],[-0.1, 1.1]])  # [[x_min, x_max], [y_min, y_max]]
            # see paper page 15 - https://arxiv.org/pdf/2004.11797.pdf
            envelope = Envelope2D(env_bbox=torch.tensor([[0, 1.5],[0, 1]]), bounds=self.config['bounds'], device=device, sample_from=self.config['envelope_sample_from'])
            domain = BoundingBox2DConstraint(bbox=self.config['bounds'])
            
            l_target_normal = torch.tensor([-1.0, 0.0])
            r_target_normal = torch.tensor([1.0, 0.0])
            l_bc_1 = LineInterface2D(start=torch.tensor([0, 0.25 - 1/12]), end=torch.tensor([0, 0.25 + 1/12]), target_normal=l_target_normal)
            l_bc_2 = LineInterface2D(start=torch.tensor([0, 0.75 - 1/12]), end=torch.tensor([0, 0.75 + 1/12]), target_normal=l_target_normal)
            r_bc_1 = LineInterface2D(start=torch.tensor([1.5, 0.25 - 1/12]), end=torch.tensor([1.5, 0.25 + 1/12]), target_normal=r_target_normal)
            r_bc_2 = LineInterface2D(start=torch.tensor([1.5, 0.75 - 1/12]), end=torch.tensor([1.5, 0.75 + 1/12]), target_normal=r_target_normal)
            
            edge_in = 0.05
            upper_target_normal = torch.tensor([0.0, 1.0])
            lower_target_normal = torch.tensor([0.0, -1.0])
            l_bc_1_upper = LineInterface2D(start=torch.tensor([0, 0.25 + 1/12]), end=torch.tensor([edge_in, 0.25 + 1/12]), target_normal=upper_target_normal)
            l_bc_1_lower = LineInterface2D(start=torch.tensor([0, 0.25 - 1/12]), end=torch.tensor([edge_in, 0.25 - 1/12]), target_normal=lower_target_normal)
            l_bc_2_upper = LineInterface2D(start=torch.tensor([0, 0.75 + 1/12]), end=torch.tensor([edge_in, 0.75 + 1/12]), target_normal=upper_target_normal)
            l_bc_2_lower = LineInterface2D(start=torch.tensor([0, 0.75 - 1/12]), end=torch.tensor([edge_in, 0.75 - 1/12]), target_normal=lower_target_normal)
            
            r_bc_1_upper = LineInterface2D(start=torch.tensor([1.5, 0.25 + 1/12]), end=torch.tensor([1.5 - edge_in, 0.25 + 1/12]), target_normal=upper_target_normal)
            r_bc_1_lower = LineInterface2D(start=torch.tensor([1.5, 0.25 - 1/12]), end=torch.tensor([1.5 - edge_in, 0.25 - 1/12]), target_normal=lower_target_normal)
            r_bc_2_upper = LineInterface2D(start=torch.tensor([1.5, 0.75 + 1/12]), end=torch.tensor([1.5 - edge_in, 0.75 + 1/12]), target_normal=upper_target_normal)
            r_bc_2_lower = LineInterface2D(start=torch.tensor([1.5, 0.75 - 1/12]), end=torch.tensor([1.5 - edge_in, 0.75 - 1/12]), target_normal=lower_target_normal)
            
            all_interfaces = CompositeInterface2D([l_bc_1, l_bc_2, r_bc_1, r_bc_2,
                                                    l_bc_1_upper, l_bc_1_lower, l_bc_2_upper, l_bc_2_lower,
                                                    r_bc_1_upper, r_bc_1_lower, r_bc_2_upper, r_bc_2_lower,
                                                    ])
            
            # TODO: the obstacles are Decagons, not circles; probably not worth the effort though
            # the holes are described in the paper page 19, - https://arxiv.org/pdf/2004.11797.pdf
            circle_obstacle_1 = CircleObstacle2D(center=torch.tensor([0.5, 1.0/3]), radius=torch.tensor(0.05))
            circle_obstacle_2 = CircleObstacle2D(center=torch.tensor([0.5, 2.0/3]), radius=torch.tensor(0.05))
            circle_obstacle_3 = CircleObstacle2D(center=torch.tensor([1.0, 1.0/4]), radius=torch.tensor(0.05))
            circle_obstacle_4 = CircleObstacle2D(center=torch.tensor([1.0, 2.0/4]), radius=torch.tensor(0.05))
            circle_obstacle_5 = CircleObstacle2D(center=torch.tensor([1.0, 3.0/4]), radius=torch.tensor(0.05))
            all_obstacles = CompositeConstraint([circle_obstacle_1, circle_obstacle_2, 
                                                   circle_obstacle_3, circle_obstacle_4, circle_obstacle_5])
            
            # sample once and keep the points; these are used for plotting
            self.constr_pts_dict = {
                'envelope': envelope.get_sampled_points(N=self.config['n_points_envelope']).cpu().numpy().T,
                'interface': all_interfaces.get_sampled_points(N=self.config['n_points_interfaces'])[0].cpu().numpy().T,
                'obstacles': all_obstacles.get_sampled_points(N=self.config['n_points_obstacles']).cpu().numpy().T,
                'domain': domain.get_sampled_points(N=self.config['n_points_domain']).cpu().numpy().T,
            }
            
            # save the constraints
            self._envelope_constr = [envelope]
            self._interface_constraints = [l_bc_1, l_bc_2, r_bc_1, r_bc_2]
            self._obstacle_constraints = [all_obstacles]
            self._domain = domain
    
        elif self.config['problem'] == 'simjeb':
            # see paper page 5 - https://arxiv.org/pdf/2105.03534.pdf
            # measurements given in 100s of millimeters
            bounds = torch.from_numpy(np.load('GINN/simJEB/derived/bounds.npy')).to(device).float()
            
            # scale_factor and translation_vector
            scale_factor = np.load('GINN/simJEB/derived/scale_factor.npy')
            center_for_translation = np.load('GINN/simJEB/derived/center_for_translation.npy')
            
            # load meshes
            self.mesh_if = trimesh.load("GINN/simJEB/orig/interfaces.stl")
            self.mesh_env = trimesh.load("GINN/simJEB/orig/411_for_envelope.obj")
            
            # translate meshes
            self.mesh_if.apply_translation(-center_for_translation)
            self.mesh_env.apply_translation(-center_for_translation)
            
            # scale meshes
            self.mesh_if.apply_scale(1. / scale_factor)
            self.mesh_env.apply_scale(1. / scale_factor)
            
            # load points
            pts_far_outside_env = torch.from_numpy(np.load('GINN/simJEB/derived/pts_far_outside.npy')).to(device).float()
            pts_on_envelope = torch.from_numpy(np.load('GINN/simJEB/derived/pts_on_env.npy')).to(device).float()
            pts_inside_envelope = torch.from_numpy(np.load('GINN/simJEB/derived/pts_inside.npy')).to(device).float()
            pts_outside_envelope = torch.from_numpy(np.load('GINN/simJEB/derived/pts_outside.npy')).to(device).float()
            interface_pts = torch.from_numpy(np.load('GINN/simJEB/derived/interface_points.npy')).to(device).float()
            interface_normals = torch.from_numpy(np.load('GINN/simJEB/derived/interface_normals.npy')).to(device).float()
            pts_around_interface = torch.from_numpy(np.load('GINN/simJEB/derived/pts_around_interface_outside_env_10mm.npy')).to(device).float()
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
            
            
        else:
            raise NotImplementedError(f'Problem {self.config["problem"]} not implemented')
    
        self.bounds = config['bounds'].cpu()
        ## For plotting
        self.X0, self.X1, self.xs = get_meshgrid_in_domain(self.bounds)
        self.xs = torch.tensor(self.xs).float()
    
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
        
    
    def recalc_output(self, f, params, z_latents):
        """Compute the function on the grid.
        epoch: will be used to identify figures for wandb or saving
        :param z_latents: 
        """
        with torch.no_grad():
            if self.config['nx']==2:
                y = f(params, *tensor_product_xz(self.xs, z_latents)).detach().cpu().numpy()
                Y = einops.rearrange(y, '(bz h w) 1 -> bz h w', bz=len(z_latents), h=self.X0.shape[0])
                return y, Y
            elif self.config['nx']==3:
                meshes = []
                for z_ in z_latents: ## do marching cubes for every z
                    
                    def f_fixed_z(x):
                        """A wrapper for calling the model with a single fixed latent code"""
                        return f(params, *tensor_product_xz(x, z_.unsqueeze(0))).squeeze(0)
                    
                    verts_, faces_ = get_mesh(f_fixed_z,
                                                N=self.config["mc_resolution"],
                                                device=z_latents.device,
                                                bbox_min=self.config["bounds"][:,0],
                                                bbox_max=self.config["bounds"][:,1],
                                                chunks=1,
                                                return_normals=0)
                    # print(f"Found a mesh with {len(verts_)} vertices and {len(faces_)} faces")
                    meshes.append((verts_, faces_))
                return meshes
        
    def is_inside_envelope(self, p_np: PointWrapper):
        """Remove points that are outside the envelope"""
        if not self.config['problem'] == 'simjeb':
            raise NotImplementedError('This function is only implemented for the simjeb problem')
        
        is_inside_mask = self.mesh_env.contains(p_np.data)
        return is_inside_mask