import einops
import numpy as np
import atexit
from psutil import TimeoutExpired
import torch
import cripser
import math

from zmq import device
from GINN.ph.ph_distributed import calc_3d_ph_4d_array, calc_3d_ph_4d_array_full
from GINN.speed.timer import Timer
from GINN.speed.dummy_async_res import DummyAsyncResult
from GINN.speed.mp_manager import MPManager
import multiprocessing as mp
import trimesh
import os

from util.model_utils import tensor_product_xz

# from topologylayer.util.construction import unique_simplices
# from topologylayer.nn import LevelSetLayer2D, TopKBarcodeLengths, LevelSetLayer



class PHManager():
    def __init__(self, 
                 mp_pool,
                 nx,
                 bounds, 
                 netp,
                 scc_n_grid_points, 
                 func_inside_envelope,
                 ph_1_hole_level,
                 ph_loss_target_betti,
                 iso_level,
                 ph_loss_sub0_points,
                 ph_loss_super0_points,
                 simjeb_root_dir,
                 problem_str,
                 ph_loss_maxdim,
                 is_density,
                 **kwargs) -> None:
        '''
        The MP Pool is passed, because it needs to be initialized BEFORE torch is initialized.
        '''
        self.netp = netp
        self.nx = nx
        self.bounds = bounds
        self.problem_str = problem_str
        self.simjeb_root_dir = simjeb_root_dir
        
        # PH specific parameters
        self.ISO = iso_level
        self.HOLE_LEVEL = ph_1_hole_level
        self.MAXDIM = ph_loss_maxdim
        self.TARGET_Betti = ph_loss_target_betti
        self.ph_loss_sub0 = ph_loss_sub0_points
        self.ph_loss_super0_points = ph_loss_super0_points
        self.n_grid_points = scc_n_grid_points
        self.loss_sign = -1 if is_density else 1

        self.is_mp_on = mp_pool is not None
        if self.is_mp_on:
            self.pool = mp_pool
            atexit.register(self._cleanup_pool)
        
        self.xs, self.xs_flat = self.__generate_x_grid(self.n_grid_points, nx, self.bounds.detach().cpu().numpy())
        self.xs_inside_envelope, self.xs_inside_envelope_mask, self.Y_inf = self.__generate_x_grid_inside_envelop(self.xs, self.xs_flat, func_inside_envelope)
        

    def _cleanup_pool(self):
        if self.is_mp_on:
            self.pool.close()
            self.pool.join()
            
    def apply_async(self, fn, args, kwdict={}):
        if self.is_mp_on:
            return self.pool.apply_async(fn, args, kwdict)
        return DummyAsyncResult(fn(*args, **kwdict))

    def __generate_x_grid(self, n_grid_points, nx, bounds) -> None:
        ### generate x_grid for ph loss
        x_mesh = np.meshgrid(*[np.linspace(*bound, n_grid_points) for bound in bounds])
        # x_mesh_flat = [x.flatten() for x in x_mesh]
        xs = np.vstack([x.ravel() for x in x_mesh]).T
        xs = torch.tensor(xs, dtype=torch.float)
        xs_flat = xs.clone().detach()
        xs = xs.reshape(*(n_grid_points,)*nx, nx)
        return xs, xs_flat

    def __generate_x_grid_inside_envelop(self, xs, xs_flat, func_inside_envelope) -> None:
        
        if self.problem_str == 'simjeb':
            xs_load = torch.load(os.path.join(self.simjeb_root_dir, 'x_domain_grid.pt'), map_location='cuda' if torch.cuda.is_available() else 'cpu')
            if torch.equal(xs_load, xs_flat):
                print("Loading PH manager grid classification from file")
                xs_inside_envelope_mask = torch.load(os.path.join(self.simjeb_root_dir, 'x_domain_inside_envelop.pt'), map_location='cuda' if torch.cuda.is_available() else 'cpu')
            else:
                print("Computing PH manager grid classification.")
                xs_inside_envelope_mask = torch.from_numpy(func_inside_envelope(xs_flat.cpu().numpy())).to(device=xs.device)
                torch.save(xs, os.path.join(self.simjeb_root_dir, 'x_domain_grid.pt'))
                torch.save(xs_inside_envelope_mask, os.path.join(self.simjeb_root_dir, 'x_domain_inside_envelop.pt'))
        else:
            xs_inside_envelope_mask = func_inside_envelope(xs_flat).to(device=xs.device)
    
        # # directly work with inside pts
        xs_inside_envelope = xs_flat[xs_inside_envelope_mask]
        y_shape_for_single = xs_flat.shape[:-1]
        Y_inf = torch.full(y_shape_for_single, float('inf'), device=xs.device)
        return xs_inside_envelope, xs_inside_envelope_mask, Y_inf


    def calc_ph(self, z: torch.tensor) -> list:
        """
        Calculate the ph of the current model output.

        Args:
            z: tensor[batch, nz] batched latent input

        Returns:
            list[batch] containing ndarray[N_ph, 9] PH results of batches
        """
        ph = []
        # forward only points inside envelope
        Y_flat = self.netp.grouped_no_grad_fwd('vf', *tensor_product_xz(self.xs_inside_envelope, z)).squeeze(dim=1)  # [n_grid_points, bz]
        # put them back to the full grid
        Y_flat = einops.rearrange(Y_flat, '(bz n) -> bz n', bz=z.shape[0])
        Y = einops.repeat(self.Y_inf, '... -> bz ...', bz=z.shape[0]).clone()
        Y[:, self.xs_inside_envelope_mask] = Y_flat
        Y = Y.reshape(z.shape[0], *(self.n_grid_points,)*self.nx)
        Y_cpu_np = Y.cpu().numpy()
        ph = [self.apply_async(calc_3d_ph_4d_array, (Y_cpu_np[i], self.MAXDIM)) for i in range(0, z.shape[0])]
        return ph    

   
    def calc_ph_loss_cripser(self, z: torch.tensor)-> tuple[bool, torch.tensor]:
        """
        Calculate the persistent homology loss using a mpm pool.

        Args:
            z: [batch, nz] batched latent input
        
        Returns:
            succuss: True if computation successfull
            loss: [1] ph loss averaged over batches
            
        """
        # forward only points inside envelope
        Y_flat = self.netp.grouped_no_grad_fwd('vf', *tensor_product_xz(self.xs_inside_envelope, z)).squeeze(dim=1)  # [n_grid_points, bz]
        Y_flat = einops.rearrange(Y_flat, '(bz n) -> bz n', bz=z.shape[0])
        # put them back to the full grid
        Y = einops.repeat(self.Y_inf, '... -> bz ...', bz=z.shape[0]).clone()
        Y[:, self.xs_inside_envelope_mask] = Y_flat
        # unflatten Y
        Y = Y.reshape(z.shape[0], *(self.n_grid_points,)*self.nx)
        Y_cpu_np = Y.cpu().numpy()
        
        penalty_tuples_list = []
        for i in range(z.shape[0]):
            penalty_tuples_list.append(self.apply_async(calc_3d_ph_4d_array_full, (Y_cpu_np[i], i, self.MAXDIM,
                                                                            z.cpu().numpy(), self.TARGET_Betti, self.ISO, self.nx,
                                                                        self.ph_loss_super0_points, self.ph_loss_sub0, self.HOLE_LEVEL)))
        
        with Timer.record('PH loss computation'):
            penalty_tuples_list = [res.get() for res in penalty_tuples_list]

        # unpack the tuples
        x_list = []
        z_list = []
        loss_key_list = []
        loss_func_list = []
        for i in range(0, z.shape[0]):
            list_tuples = penalty_tuples_list[i]
            x_list.extend(list_tuples[0])
            z_list.extend(list_tuples[1])
            loss_key_list.extend(list_tuples[2])
            loss_func_list.extend(list_tuples[3])
    
        x_idcs = torch.from_numpy(np.concatenate(x_list)).to(self.xs.device)
        if len(x_idcs) == 0:
            return False, torch.tensor(0.0, device=self.xs.device), torch.tensor(0.0, device=self.xs.device), torch.tensor(0.0, device=self.xs.device)
       
        # select with indices 
        x_in = self.xs[*x_idcs.T]
        z_in = torch.from_numpy(np.concatenate(z_list)).to(self.xs.device)
        
        list_loss_scc = []
        list_loss_super0 = []
        list_loss_sub0 = []
        
        # do single forward pass for all x's
        Y_grad = self.netp(x_in, z_in)
        # compute loss for each listitem
        cur_idx = 0
        for x_in, loss_key, loss_func in zip(x_list, loss_key_list, loss_func_list):
            loss = self.loss_sign * loss_func(Y_grad[cur_idx:cur_idx+x_in.shape[0]])
            cur_idx += x_in.shape[0]
            if loss_key == 'loss_scc':
                list_loss_scc.append(loss)
            elif loss_key == 'loss_super0':
                list_loss_super0.append(loss)
            elif loss_key == 'loss_sub0':
                list_loss_sub0.append(loss)
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")
            
        loss_scc = torch.stack(list_loss_scc).sum() if len(list_loss_scc) > 0 else torch.tensor(0.0, device=self.xs.device)
        loss_super0 = torch.stack(list_loss_super0).mean() if len(list_loss_super0) > 0 else torch.tensor(0.0, device=self.xs.device)
        loss_sub0 = torch.stack(list_loss_sub0).mean() if len(list_loss_sub0) > 0 else torch.tensor(0.0, device=self.xs.device)
            
        return True, loss_scc, loss_sub0, loss_super0
