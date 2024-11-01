import torch
from abc import ABC, abstractmethod
from typing import List

from GINN.problems.sampling_primitives import BBLineSampler, sample_bbox, sample_disk, sample_line_segment


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

class CompositeInterface2D(Constraint):
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

class CircleObstacle2D(Constraint):
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.sampled_points = None

    def get_sampled_points(self, N):
        return sample_disk(self.center, self.radius, N=N)


class Envelope2D(Constraint):
    def __init__(self, env_bbox, bounds, device, sample_from='on envelope'):
        '''
        bounds[i] gives min, max along dimension i
        
        '''
        ## TODO: this might look completely different in simJEB.
        ## It might be easier to sample a lot of points using the mesh in a pre-processing step.
        ## It's slightly less precise than using true geometric primitives, but that's finnicky.
        self.bounds = bounds.to(device)
        self.env_bbox = env_bbox
        self.sample_from = sample_from

        self.bbox_line_sampler = BBLineSampler(nx=2, bnds=self.env_bbox)
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
        elif self.sample_from == 'on envelope':
            return self.bbox_line_sampler.sample_on_bb(N)
        else:
            raise ValueError(f'Sample from not in valid values: {self.sample_from}')
                
class BoundingBox2DConstraint(Constraint):
    
    def __init__(self, bbox):
        self.bbox = bbox
        
    def get_sampled_points(self, N):
        return sample_bbox(self.bbox, N)
                
                
class SampleEnvelope(Constraint):
    def __init__(self, pts_on_envelope, pts_outside_envelope, sample_from='on envelope'):
        self.sample_from = sample_from
        self.pts_on_envelope_constraint = SampleConstraint(pts_on_envelope)
        self.pts_outside_envelope_constraint = SampleConstraint(pts_outside_envelope)
        self.ratio_on_envelope = len(pts_on_envelope) / (len(pts_on_envelope) + len(pts_outside_envelope))

    def get_sampled_points(self, N):
        if self.sample_from == 'on envelope':
            return self.pts_on_envelope_constraint.get_sampled_points(N)
        
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
    
