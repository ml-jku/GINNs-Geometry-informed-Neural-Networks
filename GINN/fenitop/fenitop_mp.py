import multiprocessing
import os

import numpy as np

import torch

from GINN.fenitop.fenitop_single_process import FenitopSingleProcess

from util.misc import fuzzy_lexsort


class FenitopMP():
    def __init__(self,
                 bz,
                 resolution,
                 problem_str,
                 envelope_unnormalized,
                 bounds_unnormalized,
                 use_filters,
                 envelope_mesh=None, # for simjeb
                 interface_pts=None, # for simjeb
                 mp_pool=None,
                 normalize_compliance=False,
                 center_scale_coords=True,
                 np_func_inside_envelope=None,
                 simjeb_root_dir=None,
                 device='cpu',
                 **kwargs):
        """
        Constructor for the TO numerical problem setup and FEM solver. Only 2d and no MPI support. 
        """
        self.device = device
        self.bz = bz
        self.resolution = resolution
        self.nx = len(resolution)
        # if use_filters is False, then we take the resolution of the rho_phys_field
        self.chosen_resolution = resolution if use_filters else [r+1 for r in resolution]
        
        if bz > 1 and mp_pool is not None:
            def worker(input_queue, output_queue):
                feni = FenitopSingleProcess(resolution=resolution, problem_str=problem_str, 
                                            envelope_unnormalized=envelope_unnormalized, envelope_mesh=envelope_mesh, interface_pts=interface_pts, 
                                            use_filters=use_filters, normalize_compliance=normalize_compliance, **kwargs)
                while True:
                    idx, variables = input_queue.get()
                    if variables is None:  # Sentinel value to indicate the end of input
                        break
                    res = feni.calc_2d_TO_grad_field(**variables)
                    output_queue.put((idx, res))
                    
            self.input_queue, self.output_queue = mp_pool
            processes = []
            for _ in range(bz):
                p = multiprocessing.Process(target=worker, args=(self.input_queue, self.output_queue))
                p.start()
                processes.append(p)
        elif bz > 1:
            print(f'WARNING: bz > 1 but no mp_pool provided')
        
        # initialize here to get rho_field and compute sort_idcs
        self.feni_single = FenitopSingleProcess(resolution=resolution, problem_str=problem_str, 
                                            envelope_unnormalized=envelope_unnormalized, envelope_mesh=envelope_mesh, interface_pts=interface_pts, 
                                            use_filters=use_filters, normalize_compliance=normalize_compliance, **kwargs)
        # chose the rho_field or the rho_phys_field for the meshgrid
        self.chosen_rho_field = self.feni_single.rho_field if use_filters else self.feni_single.rho_phys_field  
        self.fem_xy_torch_device, self.sort_idcs, self.sort_indices_rev, self.grid_dist = self._get_torch_mesh_and_sort_idcs(self.nx, self.chosen_rho_field, center_scale_coords, bounds_unnormalized)
        self.batch_dCdrho = torch.zeros((bz, self.chosen_rho_field.x.petsc_vec.array.size), dtype=torch.float32, device=self.device)
        self.batch_dVdrho = torch.zeros((bz, self.chosen_rho_field.x.petsc_vec.array.size), dtype=torch.float32, device=self.device)
        self.batch_C = torch.zeros((bz, 1), dtype=torch.float32, device=self.device)
        self.batch_V = torch.zeros((bz, 1), dtype=torch.float32, device=self.device)
        
        self.constraint_to_pts_dict = self._get_filtered_bc_points()
        self.fem_xy_inside_mask = self._set_inside_envelope_mask(problem_str, resolution, np_func_inside_envelope, simjeb_root_dir, device)
    
    def _set_inside_envelope_mask(self, problem_str, resolution, np_func_inside_envelope, simjeb_root_dir, device) -> None:
        
        if problem_str == 'simjeb':
            # file name contains resolution and number of points (so rho and rho_phys fields are both distinguished)
            file_path = os.path.join(simjeb_root_dir, f'fem_pts_in_domain-{"_".join([str(r) for r in resolution])}-{self.fem_xy_torch_device.shape[0]}.pt')
            if os.path.exists(file_path):
                fem_xy_inside_mask = torch.load(file_path, map_location='cpu').to(device)
            else:
                print("Computing FEM inside envelope mask.")
                fem_xy_inside_mask_np = np_func_inside_envelope(self.fem_xy_torch_device.detach().cpu().numpy())
                fem_xy_inside_mask = torch.tensor(fem_xy_inside_mask_np, dtype=torch.bool, device=device)
                torch.save(fem_xy_inside_mask, file_path)
                print("Saved FEM inside envelope mask.")
        else:
            fem_xy_inside_mask = torch.ones(self.fem_xy_torch_device.shape[0], dtype=torch.bool, device=device)
                
        return fem_xy_inside_mask
    
    def _get_filtered_bc_points(self, epsilon=1e-3):
        
        if self.nx == 2:
            print(f'WARNING: bc points are not plotted for 2D problems')
            return {}
        
        pts = self.fem_xy_torch_device.detach().cpu().numpy()
        # get the boundary points
        mins = pts.min(axis=0)
        maxes = pts.max(axis=0)
        boundary_pts_list = []
        for i in range(pts.shape[1]):
            # select points which are epsilon close to the min or max of ith dimension
            min_i_boundary = pts[np.abs(pts[:, i] - mins[i]) < epsilon]
            boundary_pts_list.append(min_i_boundary)
            max_i_boundary = pts[np.abs(pts[:, i] - maxes[i]) < epsilon]
            boundary_pts_list.append(max_i_boundary)
        boundary_pts = np.concat(boundary_pts_list, axis=0)
        
        # input to these functions needs to be in format [d, n] where n is the number of points and d is the dimension
        boundary_pts_transposed = np.transpose(boundary_pts)
        
        # apply disp_bc and traction_bc filters
        constraint_to_pts_dict = {}
        disp_mask = self.feni_single.fem["disp_bc"](boundary_pts_transposed)
        constraint_to_pts_dict['disp_bc'] = boundary_pts[disp_mask]
        
        pts_traction_bc_list = []
        for F, traction_bc in self.feni_single.fem["traction_bcs"]:
            traction_mask = traction_bc(boundary_pts_transposed)
            pts_traction_bc_list.append(boundary_pts[traction_mask])
            
        pts_traction_bc_mask = np.concat(pts_traction_bc_list, axis=0)
        constraint_to_pts_dict['traction_bc'] = pts_traction_bc_mask
        return constraint_to_pts_dict
                    
    def _get_torch_mesh_and_sort_idcs(self, nx, chosen_rho_field, center_scale_coords, bounds_unnormalized):
        '''
        The sort_idcs shall be
        [[0, 0], [1, 0], [2, 0], ..., [n, 0], [0, 1], [1, 1], ... [n, 1] ..., [n, m]]
        '''
        #centers = chosen_rho_field.function_space.tabulate_dof_coordinates()[:, :nx].copy()
        centers = self.feni_single.fem['mesh'].geometry.x[:, :nx].copy()
        # NOTE: for same strange reason, (x_min, y_max) is (-x_min, y_max) and (x_max, y_min) is (x_max, -y_min)
        ## we correct for it by subtracting the min value
        centers_min = np.min(centers)
        centers_abs = centers - centers_min
        sort_indices = fuzzy_lexsort([centers_abs[:, i] for i in range(centers.shape[1])], tol=0.1 / max(self.resolution))
        sort_indices_rev = np.argsort(sort_indices)
        meshgrid = torch.tensor(centers, dtype=torch.float32, device=self.device)
        sort_indices = torch.tensor(sort_indices, dtype=torch.long, device=self.device)
        sort_indices_rev = torch.tensor(sort_indices_rev, dtype=torch.long, device=self.device)
        
        if center_scale_coords:
            # make longest side go from -1 to 1
            normalize_factor = torch.max(torch.abs(bounds_unnormalized)) / 2
            normalize_center = bounds_unnormalized[:, 1] / 2
            meshgrid = (meshgrid - normalize_center) / normalize_factor
        else:
            print(f'WARNING: center_scale_coords=False -> higher spatial resolution plotting likely fails if bounds are not close to range [-1, 1]')
        
        # get distance in each dimension between points of the meshgrid
        neighbor_diff = meshgrid[1:] - meshgrid[:-1]
        abs_diff = torch.abs(neighbor_diff)
        max_dist = torch.min(torch.where(abs_diff ==0, torch.max(abs_diff), abs_diff), dim=0).values
        
        
        meshgrid.requires_grad = True
        return meshgrid, sort_indices, sort_indices_rev, max_dist

    def calc_2d_TO_grad_field_batch(self, rho_batch, beta, **kwargs):
        """
        Calculate the gradient of the compliance and volume with respect to the density field
        """
        assert rho_batch.shape[0] == self.bz, "Batch size must match the number of processes; otherwise not implemented"
        # set rhos outside the envelope to 0
        # print(f'WARNING: Setting rhos outside the envelope to 0; 0.005 could be a better value')
        rho_batch = rho_batch * self.fem_xy_inside_mask.unsqueeze(0)
        
        # Calculate the compliance and volume for the batch
        rho_batch_np = rho_batch.cpu().detach().numpy().astype('float64')
        # Send the batch to the processes
        if self.bz > 1:
            for i in range(rho_batch.shape[0]):
                # the index is used to put the results back in the correct order
                self.input_queue.put((i, dict(rho_np=rho_batch_np[i], beta=beta, **kwargs)))
            
        for i in range(rho_batch.shape[0]):
            # the index is used to put the results back in the correct order
            if self.bz > 1:
                idx, (dCdrho, dVdrho, C_value, V_value) = self.output_queue.get()
            else:
                idx = i
                dCdrho, dVdrho, C_value, V_value = self.feni_single.calc_2d_TO_grad_field(rho_batch_np[i], beta, **kwargs)
            self.batch_dCdrho[idx] = torch.tensor(dCdrho, device=self.device)
            self.batch_dVdrho[idx] = torch.tensor(dVdrho, device=self.device)
            self.batch_C[idx] = torch.tensor(C_value, device=self.device)
            self.batch_V[idx] = torch.tensor(V_value, device=self.device)

        # print(f'WARNING: setting compliance gradient to 0 for points outside the envelope')
        self.batch_dCdrho = self.batch_dCdrho * self.fem_xy_inside_mask.unsqueeze(0)
        return (self.batch_C, self.batch_dCdrho), (self.batch_V, self.batch_dVdrho)



