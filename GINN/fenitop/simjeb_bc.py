import numpy as np

class SimjebBC():
    
    def __init__(self, interface_pts, bounds, xyz_split_thresh=(0,0,0)):
        self.bounds = bounds
        self.split_interfaces(interface_pts, xyz_split_thresh)
        
    def split_interfaces(self, interface_pts, xyz_split_thresh):
        x_split_thresh, y_split_thresh, z_split_thresh = xyz_split_thresh
        top_interfaces = interface_pts[interface_pts[:,2] > z_split_thresh]
        self.top_left_if = top_interfaces[top_interfaces[:,1] < y_split_thresh]
        self.top_right_if = top_interfaces[top_interfaces[:,1] > y_split_thresh]

        bottom_interfaces = interface_pts[interface_pts[:,2] < z_split_thresh]
        self.bottom_left_back_if = bottom_interfaces[(bottom_interfaces[:,0] < x_split_thresh) & (bottom_interfaces[:,1] < y_split_thresh)]
        self.bottom_left_front_if = bottom_interfaces[(bottom_interfaces[:,0] > x_split_thresh) & (bottom_interfaces[:,1] < y_split_thresh)]
        self.bottom_right_back_if = bottom_interfaces[(bottom_interfaces[:,0] < x_split_thresh) & (bottom_interfaces[:,1] > y_split_thresh)]
        self.bottom_right_front_if = bottom_interfaces[(bottom_interfaces[:,0] > x_split_thresh) & (bottom_interfaces[:,1] > y_split_thresh)]
        

    def _project_back(self, pts):
        '''Project the points to the back wall of the domain'''
        pts_new = pts.copy()
        pts_new[0] = self.bounds[0,0]
        return pts_new

    def _project_down(self, pts):
        '''Project the points to the bottom wall of the domain'''
        pts_new = pts.copy()
        pts_new[2] = self.bounds[2,0]
        return pts_new

    def _project_up(self, pts):
        '''Project the points to the bottom wall of the domain'''
        pts_new = pts.copy()
        pts_new[2] = self.bounds[2,1]
        return pts_new

    # TOP INTERFACES

    def top_left_interface_back_bc_mask(self, x, radius):
        '''Select boundary points close to the top left interface projected to the back wall
            x: boundary mesh points (a*b + a*c + b*c) x 2 for each side
            x: (3, n)
        '''
        assert x.shape[0] == 3, f'x.shape {x.shape}'
        left_if_position = np.mean(self.top_left_if, axis=0)
        left_if_back = self._project_back(left_if_position)
        left_mask = np.linalg.norm(x - left_if_back[:, np.newaxis], axis=0) < radius
        print(f'INFO: mask.sum() for top left back: {left_mask.sum()}')
        return left_mask

    def top_left_interface_top_bc_mask(self, x, radius):
        '''Select boundary points close to the top left interface projected to the top wall
            x: boundary mesh points (a*b + a*c + b*c) x 2 for each side
            x: (3, n)
        '''
        assert x.shape[0] == 3, f'x.shape {x.shape}'
        left_if_position = np.mean(self.top_left_if, axis=0)
        left_if_top = self._project_up(left_if_position)
        left_mask = np.linalg.norm(x - left_if_top[:, np.newaxis], axis=0) < radius
        print(f'INFO: mask.sum() for top left top: {left_mask.sum()}')
        return left_mask

    def top_right_interface_back_bc_mask(self, x, radius):
        '''Select boundary points close to the top left interface projected to the top wall
            x: boundary mesh points (a*b + a*c + b*c) x 2 for each side
            x: (3, n)
        '''
        assert x.shape[0] == 3, f'x.shape {x.shape}'
        right_if_position = np.mean(self.top_right_if, axis=0)
        right_if_back = self._project_back(right_if_position)
        right_mask = np.linalg.norm(x - right_if_back[:, np.newaxis], axis=0) < radius
        print(f'INFO: mask.sum() for top right back: {right_mask.sum()}')
        return right_mask

    def top_right_interface_top_bc_mask(self, x, radius):
        '''Select boundary points close to the top left interface projected to the top wall
            x: boundary mesh points (a*b + a*c + b*c) x 2 for each side
            x: (3, n)
        '''
        assert x.shape[0] == 3, f'x.shape {x.shape}'
        right_if_position = np.mean(self.top_right_if, axis=0)
        right_if_top = self._project_up(right_if_position)
        right_mask = np.linalg.norm(x - right_if_top[:, np.newaxis], axis=0) < radius
        print(f'INFO: mask.sum() for top right top: {right_mask.sum()}')
        return right_mask

    # BOTTOM INTERFACES

    def bottom_left_back_interface_bc_mask(self, x, radius):
        '''Select boundary points close to the bottom left back interface projected to the bottom wall
            x: boundary mesh points (a*b + a*c + b*c) x 2 for each side
            x: (3, n)
        '''
        assert x.shape[0] == 3, f'x.shape {x.shape}'
        bottom_left_back_position = np.mean(self.bottom_left_back_if, axis=0)
        bottom_left_back_down = self._project_down(bottom_left_back_position)
        left_mask = np.linalg.norm(x - bottom_left_back_down[:, np.newaxis], axis=0) < radius
        print(f'INFO: mask.sum() for bottom left back: {left_mask.sum()}')
        return left_mask

    def bottom_left_front_interface_bc_mask(self, x, radius):
        '''Select boundary points close to the bottom left front interface projected to the bottom wall
            x: boundary mesh points (a*b + a*c + b*c) x 2 for each side
            x: (3, n)
        '''
        assert x.shape[0] == 3, f'x.shape {x.shape}'
        bottom_left_front_position = np.mean(self.bottom_left_front_if, axis=0)
        bottom_left_front_down = self._project_down(bottom_left_front_position)
        left_mask = np.linalg.norm(x - bottom_left_front_down[:, np.newaxis], axis=0) < radius
        print(f'INFO: mask.sum() for bottom left front: {left_mask.sum()}')
        return left_mask

    def bottom_right_back_interface_bc_mask(self, x, radius):
        '''Select boundary points close to the bottom right back interface projected to the bottom wall
            x: boundary mesh points (a*b + a*c + b*c) x 2 for each side
            x: (3, n)
        '''
        assert x.shape[0] == 3, f'x.shape {x.shape}'
        bottom_right_back_position = np.mean(self.bottom_right_back_if, axis=0)
        bottom_right_back_down = self._project_down(bottom_right_back_position)
        right_mask = np.linalg.norm(x - bottom_right_back_down[:, np.newaxis], axis=0) < radius
        print(f'INFO: mask.sum() for bottom right back: {right_mask.sum()}')
        return right_mask

    def bottom_right_front_interface_bc_mask(self, x, radius):
        '''Select boundary points close to the bottom right front interface projected to the bottom wall
            x: boundary mesh points (a*b + a*c + b*c) x 2 for each side
            x: (3, n)
        '''
        assert x.shape[0] == 3, f'x.shape {x.shape}'
        bottom_right_front_position = np.mean(self.bottom_right_front_if, axis=0)
        bottom_right_front_down = self._project_down(bottom_right_front_position)
        right_mask = np.linalg.norm(x - bottom_right_front_down[:, np.newaxis], axis=0) < radius
        print(f'INFO: mask.sum() for bottom right front: {right_mask.sum()}')
        return right_mask