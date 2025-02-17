import logging
import time

import numpy as np
from mpi4py import MPI

from GINN.fenitop.feni_configs import get_configs

from .fem import form_fem
from .fem2 import form_fem2
from .parameterize import DensityFilter, Heaviside
from .sensitivity import Sensitivity



class FenitopSingleProcess():
    def __init__(self,
                 resolution,
                 problem_str,
                 use_filters,
                 vol_frac,
                 envelope_unnormalized,
                 normalize_compliance,
                 p,
                 vol_loss,
                 filter_radius=None,
                 envelope_mesh=None,
                 interface_pts=None,
                 F_jeb=None,
                 **kwargs):
        """
        Constructor for the TO numerical problem setup and FEM solver. Only 2d and no MPI support. 
        """
        assert MPI.COMM_WORLD.size == 1, "Only serial execution is supported"
            
        self.resolution = resolution
        self.use_filters = use_filters
        self.vol_loss = vol_loss
        self.normalize_compliance = normalize_compliance
            
        self.logger = logging.getLogger('FenitopInterface')
            
        # FEM parameters
        self.fem, self.opt = get_configs(problem_str, envelope_unnormalized, resolution, envelope_mesh=envelope_mesh, interface_pts=interface_pts, F_jeb=F_jeb)
        self.opt['penalty'] = p  # SIMP penalty
        self.opt['vol_frac'] = vol_frac
        self.opt['filter_radius'] = filter_radius

        self.comm = MPI.COMM_WORLD
        if 'disp_bc1' in self.fem and 'disp_bc2' in self.fem:
            self.linear_problem, self.u_field, self.lambda_field, self.rho_field, self.rho_phys_field = form_fem2(self.fem, self.opt)
        else:
            self.linear_problem, self.u_field, self.lambda_field, self.rho_field, self.rho_phys_field = form_fem(self.fem, self.opt)
        
        self.density_filter = DensityFilter(self.comm, self.rho_field, self.rho_phys_field, self.opt["filter_radius"], self.fem["petsc_options"])
        self.heaviside = Heaviside(self.rho_phys_field)
        self.sens_problem = Sensitivity(self.comm, self.opt, self.linear_problem, self.u_field, self.lambda_field, self.rho_phys_field)
        # for loss of TOUNN paper, Eq. 5
        self.C_init = None
        self.V_star = self.opt['vol_frac']
    
    def calc_2d_TO_grad_field(self, rho_np, beta, **kwargs):
        # start_t = time.time()
        
        if self.use_filters:
            self.rho_field.x.petsc_vec.setArray(rho_np)
            # Density filter and Heaviside projection
            self.density_filter.forward()
            # self.logger.debug(f'density filter time: {time.time() - start_t}')
            self.heaviside.forward(beta)
            # self.logger.debug(f'heaviside filter time: {time.time() - start_t}')
        else:
            self.rho_phys_field.x.petsc_vec.setArray(rho_np)
            
        # Solve FEM
        self.linear_problem.solve_fem()
        # self.logger.debug(f'fem solve time: {time.time() - start_t}')
        
        #sensitivites (gradients) 
        [C_value, V_value, U_value], sensitivities = self.sens_problem.evaluate()
        # self.logger.debug(f'sensitivities time: {time.time() - start_t}')
        
        if self.use_filters:
            self.heaviside.backward(sensitivities)
            [dCdrho, dVdrho, dUdrho] = self.density_filter.backward(sensitivities)
        else:
            dCdrho, dVdrho, dUdrho = [s.getArray().copy() if s is not None else None for s in sensitivities ]
        # self.logger.debug(f'backward time: {time.time() - start_t}')

        # Compute actual losses and gradients
        C_loss = C_value
        if self.normalize_compliance:
            # C_init as the initial complience is according to the TOUNN paper; it may be useful for generalizing to other problems
            # TODO: this upscales the gradient; is this optimal?
            if self.C_init is None and (V_value / self.V_star < 1.1):
                print(f'INFO: Setting C_init to {C_value}')
                self.C_init = C_value
            
            if self.C_init is not None:
                # only normalize if C_init is available
                dCdrho = dCdrho / self.C_init
                C_loss = C_loss / self.C_init

        # print(f'WARNING: hardcoded refscaling of compliance gradient')
        # dCdrho = dCdrho / 3.910
        default_loss = 1e-6

        if self.vol_loss == 'mse':
            V_loss = (V_value / self.V_star - 1) ** 2
            dVdrho = (2/self.V_star) * (V_value / self.V_star - 1) * dVdrho
        elif self.vol_loss == 'mae':
            V_loss = np.abs(V_value / self.V_star - 1)
            dVdrho = np.sign(V_value / self.V_star - 1) / self.V_star * dVdrho
        elif self.vol_loss == 'mse_pushdown':
            # only if V > V_star, we push it down
            if V_value > self.V_star:
                # (V_value - self.V_star) ** 2
                V_loss = (V_value / self.V_star - 1) ** 2
                dVdrho = (2/self.V_star) * (V_value / self.V_star - 1) * dVdrho
            else:
                # setting to zero leads to NaNs in the gradients
                V_loss = default_loss
                dVdrho = dVdrho * default_loss
        elif self.vol_loss == 'mse_unscaled_pushdown':
            # only if V > V_star, we push it down
            if V_value > self.V_star:
                V_loss = (V_value - self.V_star) ** 2
                dVdrho = 2 * (V_value - self.V_star) * dVdrho
            else:
                V_loss = default_loss
                dVdrho = dVdrho * default_loss
        elif self.vol_loss == 'mae_unscaled_pushdown':
            # only if V > V_star, we push it down
            if V_value > self.V_star:
                V_loss = np.abs(V_value - self.V_star)
                dVdrho = np.sign(V_value - self.V_star) * dVdrho
            else:
                V_loss = default_loss
                dVdrho = dVdrho * default_loss
        else:
            raise ValueError(f'Unknown volume loss: {self.vol_loss}')

        return dCdrho, dVdrho, C_loss, V_loss
    