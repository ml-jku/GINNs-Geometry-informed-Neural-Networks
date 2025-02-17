import torch
from abc import ABC, abstractmethod
from typing import List

from GINN.problems.sampling_primitives import BBLineSampler, sample_axis_parallel_rectangle_in_3d, sample_bbox, sample_circle, sample_disk, sample_line_segment, sample_ring


class Constraint(ABC):

    # @abstractmethod
    # def sample_interior(self):
    #     pass
    #
    # @abstractmethod
    # def sample_exterior(self):
    #     pass
    #
    # @abstractmethod
    # def sample_envelope(self):
    #     pass

    @abstractmethod
    def get_sampled_points(self, N):
        '''
        :param N: number of points to sample
        :return: pts in the shape of N x nx, whereas N is the number of points and nx the x-dimensions
        '''
        pass


class LineInterface2D(Constraint):
    def __init__(self, start, end, target_normal):
        '''
        :param start: 1D tensor of length nx
        :param end: 1D tensor of length nx
        '''
        self.start = start
        self.end = end
        self.target_normal = target_normal

    def get_sampled_points(self, N):
        pts = sample_line_segment(self.start, self.end, N=N).T
        normals = self.target_normal.repeat(N, 1)
        return pts, normals

class CompositeInterface(Constraint):
    def __init__(self, interface_list: List[Constraint]):
        self.interfaces = interface_list

    def get_sampled_points(self, N):
        n_pts_per_interface = int(N / len(self.interfaces))
        pts = []
        normals = []
        for interface in self.interfaces:
            pts_i, normals_i = interface.get_sampled_points(N=n_pts_per_interface)
            pts.append(pts_i)
            normals.append(normals_i)
        return torch.concatenate(pts, dim=0), torch.concatenate(normals, dim=0)

class CompositeConstraint(Constraint):

    def __init__(self, constraint_list: List[Constraint]):
        self.constraints = constraint_list

    def get_sampled_points(self, N):
        n_pts_per_constraint = int(N / len(self.constraints))
        return torch.concatenate([c.get_sampled_points(N=n_pts_per_constraint) for c in self.constraints], dim=0)

class DiskConstraint(Constraint):
    '''
    Sample points from a disk.
    '''
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.sampled_points = None

    def get_sampled_points(self, N):
        return sample_disk(self.center, self.radius, N=N)

class CircleConstraint(Constraint):
    '''
    Sample points from a 2D circle.
    '''
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.sampled_points = None
        
    def get_sampled_points(self, N):
        return sample_circle(self.center, self.radius, N=N)
    
class CircleConstraintWithNormals(Constraint):
    def __init__(self, position, radius, normal_inwards):
        super().__init__()
        self.position = position
        self.radius = radius
        self.normal_inwards = normal_inwards
        
    def get_sampled_points(self, N):
        sampled_pts = sample_circle(self.position, self.radius, N=N)
        
        # compute normals
        if self.normal_inwards:
            sampled_normals = self.position - sampled_pts
        else:
            sampled_normals = sampled_pts - self.position
        # normalize normals            
        sampled_normals = sampled_normals / torch.norm(sampled_normals, dim=1).view(-1, 1)
        return sampled_pts, sampled_normals

class RingConstraint(Constraint):
    
    def __init__(self, position, r1, r2):
        self.position = position
        self.r1 = r1
        self.r2 = r2
    
    def get_sampled_points(self, N):
        return sample_ring(self.position, self.r1, self.r2, N=N)

class RingEnvelope(Constraint):
    def __init__(self, center, r1, r2, bounds, sample_from):
        assert sample_from in ['exterior', 'interior'], "sample_from must be either 'exterior' or 'interior'"
        assert r1 < r2, "Inner radius must be smaller than outer radius"
        assert bounds.shape[0] == 2, "Bounds must be 2D"
        assert torch.all(center-r2 > bounds[:, 0]) and torch.all(center+r2 < bounds[:, 1]), "Ring must be fully contained in bounds"
        
        self.env_center = center
        self.r1 = r1 # inner radius
        self.r2 = r2 # outer radius
        self.sample_from = sample_from
        self.bounds = bounds
        # compute upsample factor
        area_bbox = torch.prod(self.bounds[:,1] - self.bounds[:,0])
        area_ring = torch.pi * (self.r2 ** 2 - self.r1 ** 2)
        self.upsample_factor_exterior = 1 / (1 - area_ring / area_bbox)
        self.upsample_factor_interior = 1 / (area_ring / area_bbox)
        
    def get_sampled_points(self, N):
        # sample points from the bounding box, then filter the points to be outside the ring
        if self.sample_from == 'exterior':
            sampled_points = sample_bbox(self.bounds, N=int(N*self.upsample_factor_exterior))
            mask = ((torch.norm(sampled_points - self.env_center, dim=1) < self.r1) | (torch.norm(sampled_points - self.env_center, dim=1) > self.r2))
        else:
            sampled_points = sample_bbox(self.bounds, N=int(N*self.upsample_factor_interior))
            mask = ((torch.norm(sampled_points - self.env_center, dim=1) >= self.r1) & (torch.norm(sampled_points - self.env_center, dim=1) <= self.r2))
        return sampled_points[mask]
    
class RotationSymmetricRingConstraint(Constraint):
    def __init__(self, center, r1, r2, n_cycles: int):
        assert n_cycles > 0, "Number of cycles must be greater than 0"
        
        self.center = center
        self.r1 = r1
        self.r2 = r2
        self.n_cycles = n_cycles
        
    def get_sampled_points(self, N):
        
        pts_start = sample_ring(self.center, self.r1, self.r2, N=N//self.n_cycles)
        pi_t = torch.tensor(torch.pi)
        rot_matrix = torch.tensor([[torch.cos(2 * pi_t / self.n_cycles), -torch.sin(2 * pi_t / self.n_cycles)], [torch.sin(2 * pi_t / self.n_cycles), torch.cos(2 * pi_t / self.n_cycles)]])
        pts_all = [pts_start]
        for i in range(1, self.n_cycles):
            pts_all.append(torch.mm(pts_all[-1] - self.center, rot_matrix) + self.center)
        return torch.cat(pts_all, dim=0)
    

class RectangleEnvelope(Constraint):
    def __init__(self, env_bbox, bounds, sample_from):
        '''
        bounds[i] gives min, max along dimension i
        
        '''
        ## TODO: this might look completely different in simJEB.
        ## It might be easier to sample a lot of points using the mesh in a pre-processing step.
        ## It's slightly less precise than using true geometric primitives, but that's finnicky.
        self.bounds = bounds
        self.env_bbox = env_bbox
        self.sample_from = sample_from

        area_bbox = torch.prod(self.env_bbox[:,1] - self.env_bbox[:,0])
        area_domain = torch.prod(self.bounds[:,1] - self.bounds[:,0])
        self.area_ratio = (area_domain - area_bbox) / area_domain

    def get_sampled_points(self, N):
        
        if self.sample_from == 'exterior':
            ratio_adjusted_n_pts = int(N / self.area_ratio)
            sampled_points = sample_bbox(self.bounds, N=ratio_adjusted_n_pts)  # bnds[i] gives min, max along dimension i
            mask = ((sampled_points[:, 0] >= self.env_bbox[0, 0]) & (sampled_points[:, 0] <= self.env_bbox[0, 1]) &
                    (sampled_points[:, 1] >= self.env_bbox[1, 0]) & (sampled_points[:, 1] <= self.env_bbox[1, 1]))
            return sampled_points[~mask]
        elif self.sample_from == 'interior':
            sampled_points = sample_bbox(self.env_bbox, N=N)
            return sampled_points
        raise ValueError("sample_from must be either 'exterior' or 'interior'")
        
                
class BoundingBox2DConstraint(Constraint):
    
    def __init__(self, bbox):
        self.bbox = bbox
        
    def get_sampled_points(self, N):
        return sample_bbox(self.bbox, N)
                
                
class SampleEnvelope(Constraint):
    def __init__(self, pts_on_envelope, pts_outside_envelope):
        self.pts_on_envelope_constraint = SampleConstraint(pts_on_envelope)
        self.pts_outside_envelope_constraint = SampleConstraint(pts_outside_envelope)
        self.ratio_on_envelope = len(pts_on_envelope) / (len(pts_on_envelope) + len(pts_outside_envelope))

    def get_sampled_points(self, N):
        n_pts_on_envelope = int(N * self.ratio_on_envelope)
        n_pts_outside_envelope = N - n_pts_on_envelope
        return torch.cat([self.pts_on_envelope_constraint.get_sampled_points(n_pts_on_envelope),
                          self.pts_outside_envelope_constraint.get_sampled_points(n_pts_outside_envelope)], dim=0)
        

    
class SampleConstraint(Constraint):
    def __init__(self, sample_pts):
        self.sample_pts = sample_pts
        self.shuffled_idcs = torch.randperm(self.sample_pts.shape[0])
        self.current_idx = 0
        
    def get_sampled_points(self, N):
        if self.current_idx + N > self.sample_pts.shape[0]:
            self.current_idx = 0
            self.shuffled_idcs = torch.randperm(self.sample_pts.shape[0])
        sampled_pts = self.sample_pts[self.shuffled_idcs[self.current_idx:self.current_idx + N]]
        self.current_idx += N
        return sampled_pts
    
class SampleConstraintWithNormals(Constraint):
    def __init__(self, sample_pts, normals):
        super().__init__()
        assert sample_pts.shape[0] == normals.shape[0]
        self.sample_pts = sample_pts
        self.normals = normals
        self.shuffled_idcs = torch.randperm(self.sample_pts.shape[0])
        self.current_idx = 0
        
    def get_sampled_points(self, N):
        if self.current_idx + N > self.sample_pts.shape[0]:
            self.current_idx = 0
            self.shuffled_idcs = torch.randperm(self.sample_pts.shape[0])
        sampled_pts = self.sample_pts[self.shuffled_idcs[self.current_idx:self.current_idx + N]]
        sampled_normals = self.normals[self.shuffled_idcs[self.current_idx:self.current_idx + N]]
        self.current_idx += N
        return sampled_pts, sampled_normals
    
class CuboidEnvelope(Constraint):
    def __init__(self, env_bbox, bounds, sample_from):
        '''
        bounds[i] gives min, max along dimension i
        
        '''
        self.bounds = bounds
        self.env_bbox = env_bbox
        self.sample_from = sample_from
        
        vol_bbox = torch.prod(self.env_bbox[:,1] - self.env_bbox[:,0])
        vol_domain = torch.prod(self.bounds[:,1] - self.bounds[:,0])
        self.vol_ratio = (vol_domain - vol_bbox) / vol_domain
        
    def get_sampled_points(self, N):
        if self.sample_from == 'exterior':
            ratio_adjusted_n_pts = int(N / self.vol_ratio)
            sampled_points = sample_bbox(self.bounds, N=ratio_adjusted_n_pts)
            mask = ((sampled_points[:, 0] >= self.env_bbox[0, 0]) & (sampled_points[:, 0] <= self.env_bbox[0, 1]) &
                    (sampled_points[:, 1] >= self.env_bbox[1, 0]) & (sampled_points[:, 1] <= self.env_bbox[1, 1]) &
                    (sampled_points[:, 2] >= self.env_bbox[2, 0]) & (sampled_points[:, 2] <= self.env_bbox[2, 1]))
            return sampled_points[~mask]
        elif self.sample_from == 'interior':
            sampled_points = sample_bbox(self.env_bbox, N=N)
            return sampled_points
        
class BoundingBox3DConstraint(Constraint):
    
    def __init__(self, bbox):
        self.bbox = bbox
        
    def get_sampled_points(self, N):
        return sample_bbox(self.bbox, N)
    
    
class RectangleInterface3D(Constraint):
    '''
    This is rectangle (2D object) in 3D space.
    The rectangle is assumed to be axis-parallel.
    '''
    def __init__(self, start, end, target_normal):
        self.start = start
        self.end = end
        self.target_normal = target_normal
        
    def get_sampled_points(self, N):
        pts = sample_axis_parallel_rectangle_in_3d(self.start, self.end, N=N)
        normals = self.target_normal.repeat(N, 1)
        return pts, normals