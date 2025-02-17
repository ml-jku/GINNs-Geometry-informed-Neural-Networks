import torch
import numpy as np

def sample_bbox(bnds, N=1000):
    """
    Sample N points in nx dimenions from a rectangular domain given by bnds.
    bnds[i] gives min, max along dimension i
    """
    return bnds[:,0] + (bnds[:,-1] - bnds[:,0])*torch.rand(N, bnds.shape[0]) ## Sample bx number of points within the specified domain

class BBLineSampler():

    def __init__(self, nx, bnds) -> None:
        self.nx = nx
        self.lines_list, self.ratio_per_line_list = self.get_lines_and_n_points_to_sample_uniformly(bnds, self.nx)

    def sample_on_bb(self, N):
        """
        Sample N points in nx dimenions on a rectangular domain given by bnds.
        bnds[i] gives min, max along dimension i
        """
        samples = []
        for i, (p1, p2) in enumerate(self.lines_list):
            # line_samples = p1 + (p2 - p1) * sample_unit_interval(n_points_list[i], method='random')
            samples.append(sample_line_segment(p1, p2, N=int(N * self.ratio_per_line_list[i])))
            
        samples = torch.concatenate(samples, dim=1).T
        return samples

    def get_lines_and_n_points_to_sample_uniformly(self, bnds, nx):
        bbox = bnds.cpu().numpy()
        if nx != 2:
            raise NotImplementedError('Sample on Bounding box not implemented for nx != 2')

        #        0       1
        # 0: [[x_min, x_max], 
        # 1:  [y_min, y_max]]
        #               x           y
        lower_left = torch.tensor([bbox[0, 0], bbox[1, 0]])
        upper_left = torch.tensor([bbox[0, 0], bbox[1, 1]])
        lower_right = torch.tensor([bbox[0, 1], bbox[1, 0]])
        upper_right = torch.tensor([bbox[0, 1], bbox[1, 1]])
            
        lines_list = [ 
                    (lower_left, lower_right),
                    (lower_right, upper_right),
                    (upper_right, upper_left),
                    (upper_left, lower_left),
                    ]
        line_lengths = ([torch.linalg.vector_norm(p1 - p2).item() for p1, p2 in lines_list])
        ratio_per_line_list = [line_len / np.sum(line_lengths) for line_len in line_lengths]
        return lines_list, ratio_per_line_list
        

def sample_unit_interval(N, method='random'):
    """
    Sample N points from the unit interval [0, 1].
    """
    if method=='random': ## sample random uniform distribution
        return torch.rand(N)
    if method=='grid': ## sample equidistant points
        return torch.linspace(0, 1, N)

def sample_line_segment(x_start, x_end, N=50):
    """
    Sample N points from a line segment from x_a to x_b.
    """
    x_start = x_start[:,None]
    x_end = x_end[:,None]
    ts = sample_unit_interval(N, method='grid')
    return x_start + (x_end-x_start)*ts ## linear interpolation


def inside_disk(xs, x, R, strict=True):
    """Returns mask describing if points xs are strictly inside the disk."""
    compare = torch.less if strict else torch.less_equal
    return compare(torch.norm(xs - x, dim=1), R)


def sample_disk(center, R, N=50):
    """Sample (approximately) N points from a 2D disk with radius R centered at x."""
    bbox = torch.vstack([center - R, center + R]).T
    xs_bbox = sample_bbox(bbox, N=int(N * 4 / torch.pi))

    # Reject points outside the disk
    inside_mask = inside_disk(xs_bbox, center, R)
    xs_disk = xs_bbox[inside_mask]
    return xs_disk

def sample_ring(x, r1, r2, N=50):
    """Sample (approximately) N points from a 2D ring with inner radius r1 and outer radius r2 centered at x."""
    bbox = torch.vstack([x - r2, x + r2]).T
    xs_bbox = sample_bbox(bbox, N=int(N * 4 * r2**2 / (r2**2 - r1**2) / torch.pi))

    # Reject points outside the ring
    inside_mask = inside_disk(xs_bbox, x, r2)
    outside_mask = ~inside_disk(xs_bbox, x, r1)
    ring_mask = inside_mask & outside_mask
    xs_ring = xs_bbox[ring_mask]
    return xs_ring

def sample_circle(x, r, N=50):
    """Sample (approximately) N points from a 2D circle with radius r centered at x."""
    # sample between 0 and 1 and multiply by 2pi to get the angle
    theta = 2 * np.pi * torch.rand(N)
    xs = x + r * torch.vstack([torch.cos(theta), torch.sin(theta)]).T
    return xs

def sample_axis_parallel_rectangle_in_3d(start_xyz, end_xyz, N=50):
    """Sample (approximately) N points from a 3D rectangle with corners at start_xyz and end_xyz."""
    bbox = torch.vstack([start_xyz, end_xyz]).T
    xs_bbox = sample_bbox(bbox, N=N)
    return xs_bbox