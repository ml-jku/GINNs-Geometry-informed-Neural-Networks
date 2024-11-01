import numpy as np
import atexit
import torch
import cripser
import math
from GINN.speed.dummy_async_res import DummyAsyncResult
from GINN.speed.mp_manager import MPManager
import multiprocessing as mp
import trimesh
import os

from scipy.spatial import Delaunay
# from topologylayer.util.construction import unique_simplices
# from topologylayer.nn import LevelSetLayer2D, TopKBarcodeLengths, LevelSetLayer

from util.misc import set_and_true
from util.model_utils import tensor_product_xz


def calc_3d_ph_4d_array(array: np.ndarray, slice: int, maxdim: int=0)-> np.ndarray:
    return cripser.computePH(array[slice, :], maxdim=maxdim)

class PHManager():
    def __init__(self, config, model) -> None:
        self.config = config
        self.model = model
        self.ISO = self.config['iso_level']
        self.HOLE_LEVEL = self.config['ph_1_hole_level']
        self.MAXDIM = self.config['ph_loss_maxdim'] if 'ph_loss_maxdim' in self.config else 0
        self.TARGET_Betti = self.config['ph_loss_target_betti']
        self.ph_loss_sub0 = ('ph_loss_sub0_points' in self.config) and (self.config['ph_loss_sub0_points'])
        self.ph_loss_super0_points = ('ph_loss_super0_points' in self.config) and (self.config['ph_loss_super0_points'])
        self.device = config['device']
        self.n_grid_points = self.config['scc_n_grid_points']
        self.bounds = self.config['bounds'].cpu().detach().numpy()
        # self.mpm = MPManager(config)
        self.is_mp_on = config.get('ph_num_workers', 0) > 1
        if self.is_mp_on:
            self.pool = mp.Pool(processes=config['ginn_bsize'])
            atexit.register(self._cleanup_pool)

        if self.config['problem'] == 'simjeb':
            # scale_factor and translation_vector
            scale_factor = np.load(os.path.join(self.config['simjeb_root_dir'], 'scale_factor.npy'))
            center_for_translation = np.load(os.path.join(self.config['simjeb_root_dir'], 'center_for_translation.npy'))
            self.envelop = trimesh.load(os.path.join(self.config['simjeb_root_dir'], '411_for_envelope.obj'))
            self.envelop.apply_translation(-center_for_translation)
            self.envelop.apply_scale(1. / scale_factor)
        
        self.__generate_x_grid()
        
        self.xs_inside_envelop = None
        if self.config['problem'] == 'simjeb':
            self.__generate_x_grid_inside_envelop()
        # self.__init_topologylayer_levelset()

    def _cleanup_pool(self):
        if self.is_mp_on:
            self.pool.close()
            self.pool.join()
            
    def apply_async(self, fn, args, kwdict={}):
        if self.is_mp_on:
            return self.pool.apply_async(fn, args, kwdict)
        return DummyAsyncResult(fn(*args, **kwdict))

    def __generate_x_grid(self) -> None:
        ### generate x_grid for ph loss
        x_mesh = np.meshgrid(*[np.linspace(*bound, self.n_grid_points) for bound in self.bounds])
        x_mesh_flat = [x.flatten() for x in x_mesh]
        xs = np.vstack([x.ravel() for x in x_mesh]).T
        self.xs = torch.tensor(xs, device='cpu').float()
        self.xs_device = self.xs.clone().detach().to(self.device)

    def __generate_x_grid_inside_envelop(self) -> None:
        
        if isinstance(self.envelop, trimesh.Trimesh):

            xs_load = torch.load(os.path.join(self.config['simjeb_root_dir'], 'x_domain_grid.pt'), map_location=torch.device('cpu'))
            if torch.equal(xs_load, self.xs):
                print("Loading PH manager grid classification from file")
                self.xs_inside_envelop = torch.load(os.path.join(self.config['simjeb_root_dir'], 'x_domain_inside_envelop.pt'), map_location=torch.device('cuda'))
            else:
                print("Computing PH manager grid classification.")
                self.xs_inside_envelop = self.envelop.contains(self.xs)
                self.xs_inside_envelop = torch.tensor(self.xs_inside_envelop, device='cpu')

                torch.save(self.xs, os.path.join(self.config['simjeb_root_dir'], 'x_domain_grid.pt'))
                torch.save(self.xs_inside_envelop, os.path.join(self.config['simjeb_root_dir'], 'x_domain_inside_envelop.pt'))
    
    def calc_ph(self, z: torch.tensor) -> list:
        """
        Calculate the ph of the current model output.

        Args:
            z: tensor[batch, nz] batched latent input

        Returns:
            list[batch] containing ndarray[N_ph, 9] PH results of batches
        """
        if set_and_true('evaluate_ph_grid_per_shape', self.config):
            ph = []
            for i in range(z.shape[0]):
                with torch.no_grad():
                    Y_flat = self.model(*tensor_product_xz(self.xs_device, z[i:i+1]))
                Y_flat_cpu =  Y_flat
                Y_flat_cpu = Y_flat_cpu.reshape((1, -1))
                if self.xs_inside_envelop is not None:
                    Y_flat_cpu[:, ~self.xs_inside_envelop] = torch.inf
                Y_cpu = Y_flat_cpu.reshape((1, *(self.n_grid_points,)*self.config['nx']))
                Y_cpu_np = Y_cpu.cpu().numpy()
                ph.append(self.apply_async(calc_3d_ph_4d_array, (Y_cpu_np, 0, self.MAXDIM)))
            return ph

        else:
            Y_flat = self.model(*tensor_product_xz(self.xs_device, z))
            Y_flat_cpu =  Y_flat.detach().cpu()
            Y_flat_cpu = Y_flat_cpu.reshape(z.shape[0], -1)
            if self.xs_inside_envelop is not None:
                Y_flat_cpu[:, ~self.xs_inside_envelop] = torch.inf
            Y = Y_flat_cpu.reshape(z.shape[0], *(self.n_grid_points,)*self.config['nx'])
            Y_np = Y.numpy()
            ph = [self.apply_async(calc_3d_ph_4d_array, (Y_np, i, self.MAXDIM)) for i in range(0, z.shape[0])]
            return ph
    

    def calc_ph_loss_crisper_slice(self, Y: torch.tensor, z:torch.tensor, PHs: list[np.ndarray], slice: int)-> torch.tensor:
        """
        Calculate the ph loss of a slice given batched images Y and the corresponding PH results.

        Args:
            Y: tensor[batch, n_grid_points, n_grid_points] or 
               tensor[batch, n_grid_points, n_grid_points, n_grid_points] 
               batched model output image in 2d or 3d.
            z: tensor[batch, nz] batched latent input
            PHs: list[batch] containing ndarray[N_ph, 9] PH results of batches
            slice: batch index to compute loss
        
        Returns:
            loss: tensor[1] loss of slice 
        """

        #results computePH; 0: dim, 1: birth, 2: death, 3: x1, 4: y1, 5: z1, 6: x2, 7: y2, 8: z2
        loss_scc = torch.tensor(0.0, device=self.device)
        loss_super0 = torch.tensor(0.0, device=self.device)
        loss_sub0 = torch.tensor(0.0, device=self.device)

        for DIM in range(0, self.MAXDIM+1):

            PH = PHs[slice]
            TARGET_Betti = self.TARGET_Betti[DIM]
            IS0 = self.ISO
            PH_dim = PH[PH[:, 0] == DIM]
            lifetime = PH_dim[:, 2] - PH_dim[:, 1]
            sorted_ids = np.argsort(lifetime)[::-1]

            PH_dim = PH_dim[sorted_ids]
            lifetime = lifetime[sorted_ids]
            lifetime_tensor = torch.from_numpy(lifetime).to(device=self.device)

            # disconnected components
            # death or birth isos
            # dim 1-3: 
            
            if DIM == 0:
                PH_filter, lifetime_filter = self.filter_PH_iso(PH_dim, lifetime)
                death_isos = PH_filter[:, 6:6+self.config['nx']].astype(int)
                birth_isos = PH_filter[:, 3:3+self.config['nx']].astype(int)
                if Y.requires_grad:
                    loss_scc += torch.clamp(self.ISO - Y[slice, *death_isos[TARGET_Betti:, :].T], max=0.).pow(2).sum()
                else:
                    # TODO: batch this forward pass with other z's
                    # TODO: doublecheck for Eric
                    x_in = self.xs_device.reshape(*(self.n_grid_points,)*self.config['nx'], self.config['nx'])[*death_isos[TARGET_Betti:, :].T]
                    Y_grad = self.model(*tensor_product_xz(x_in, z[slice:slice+1]))
                    loss_scc += torch.clamp(self.ISO - Y_grad, max=0.).pow(2).sum()
            
                #### super 0 points
                if self.ph_loss_super0_points:
                    PH_filter, lifetime_filter = self.filter_PH_super_iso(PH_dim, lifetime)
                    death_isos = PH_filter[:, 6:6+self.config['nx']].astype(int)
                    birth_isos = PH_filter[:, 3:3+self.config['nx']].astype(int)
                    if Y.requires_grad:
                        loss_super0 += -(Y[slice, *birth_isos[1:, :].T] * torch.tensor(lifetime_filter[1:], device=self.device)).sum()
                    else:
                        x_in = self.xs_device.reshape(*(self.n_grid_points,)*self.config['nx'], self.config['nx'])[*birth_isos[1:, :].T]
                        Y_grad = self.model(*tensor_product_xz(x_in, z[slice:slice+1]))
                        loss_super0 += -(Y_grad * torch.tensor(lifetime_filter[1:], device=self.device)).sum()

                #### sub 0 points
                if self.ph_loss_sub0:
                    PH_filter, lifetime_filter = self.filter_PH_sub_iso(PH_dim, lifetime)
                    death_isos = PH_filter[:, 6:6+self.config['nx']].astype(int)
                    birth_isos = PH_filter[:, 3:3+self.config['nx']].astype(int)
                    #loss += (2*torch.sigmoid(-(self.ISO - Y[slice, *death_isos.T])*3*(1.01 + z[slice, 0]))).sum()
                    #loss += (2*torch.sigmoid(-(self.ISO - Y[slice, *birth_isos.T])*3*(1.01 + z[slice, 0]))).sum()
                    if Y.requires_grad:
                        loss_sub0 += (2*torch.sigmoid(-(self.ISO - Y[slice, *death_isos.T])*30*(1.01 + z[slice, 0])) * torch.tensor(lifetime_filter, device=self.device)).sum()
                    else:
                        x_in = self.xs_device.reshape(*(self.n_grid_points,)*self.config['nx'], self.config['nx'])[*death_isos.T]
                        Y_grad = self.model(*tensor_product_xz(x_in, z[slice:slice+1]))
                        loss_sub0 += (2*torch.sigmoid(-(self.ISO - Y_grad)*30*(1.01 + z[slice, 0])) * torch.tensor(lifetime_filter, device=self.device)).sum()

            elif DIM == 1:
                    PH_filter, lifetime_filter = self.filter_PH_inf_death(PH_dim, lifetime)
                    death_isos = PH_filter[:, 6:6+self.config['nx']].astype(int)
                    birth_isos = PH_filter[:, 3:3+self.config['nx']].astype(int)
                    if Y.requires_grad:
                        loss_scc += (-self.HOLE_LEVEL - Y[slice, *birth_isos[0:TARGET_Betti, :].T]).pow(2).sum()
                        loss_scc += (self.HOLE_LEVEL - Y[slice, *death_isos[0:TARGET_Betti, :].T]).pow(2).sum()
                    else:
                        x_in_1 = self.xs_device.reshape(*(self.n_grid_points,)*self.config['nx'], self.config['nx'])[*birth_isos[0:TARGET_Betti, :].T]
                        x_in_2 = self.xs_device.reshape(*(self.n_grid_points,)*self.config['nx'], self.config['nx'])[*death_isos[0:TARGET_Betti, :].T]
                        x_in = torch.cat((x_in_1, x_in_2), dim=0)
                        Y_grad = self.model(*tensor_product_xz(x_in, z[slice:slice+1]))
                        loss_scc += (-self.HOLE_LEVEL - Y_grad[:x_in_1.shape[0]]).pow(2).sum()
                        loss_scc += (self.HOLE_LEVEL - Y_grad[x_in_1.shape[0]:]).pow(2).sum()
                        # raise NotImplementedError("Double check this code works! It is not tested")
                        
                    PH_filter, lifetime_filter = self.filter_PH_iso(PH_filter[TARGET_Betti:, :], lifetime_filter[TARGET_Betti:])
                    death_isos = PH_filter[:, 6:6+self.config['nx']].astype(int)
                    birth_isos = PH_filter[:, 3:3+self.config['nx']].astype(int)
                    if Y.requires_grad:
                        loss_scc += torch.clamp(self.ISO - Y[slice, *death_isos[:, :].T], max=0.).pow(2).sum()
                    else:
                        x_in = self.xs_device.reshape(*(self.n_grid_points,)*self.config['nx'], self.config['nx'])[*death_isos[:, :].T]
                        Y_grad = self.model(*tensor_product_xz(x_in, z[slice:slice+1]))
                        loss_scc += torch.clamp(self.ISO - Y_grad, max=0.).pow(2).sum()

            elif DIM == 2:
                raise NotImplementedError("DIM 2 loss not implemented")

        return loss_scc, loss_sub0, loss_super0


    def calc_ph_loss_cripser(self, z: torch.tensor)-> tuple[bool, torch.tensor]:
        """
        Calculate the persistent homology loss using a mpm pool.

        Args:
            z: [batch, nz] batched latent input
        
        Returns:
            succuss: True if computation successfull
            loss: [1] ph loss averaged over batches
            
        """
        if set_and_true('evaluate_ph_grid_per_shape', self.config):
            multi_res = []
            Y = torch.zeros((z.shape[0], *(self.n_grid_points,)*self.config['nx']), device=self.device, requires_grad=False)
            for i in range(z.shape[0]):
                with torch.no_grad():
                    Y_flat = self.model(*tensor_product_xz(self.xs_device, z[i:i+1]))
                Y[i] = Y_flat.reshape(*(self.n_grid_points,)*self.config['nx'])
                Y_flat_cpu =  Y_flat
                Y_flat_cpu = Y_flat_cpu.reshape((1, -1))
                if self.xs_inside_envelop is not None:
                    Y_flat_cpu[:, ~self.xs_inside_envelop] = torch.inf
                Y_cpu = Y_flat_cpu.reshape((1, *(self.n_grid_points,)*self.config['nx']))
                Y_cpu_np = Y_cpu.cpu().numpy()
                multi_res.append(self.apply_async(calc_3d_ph_4d_array, (Y_cpu_np, 0, self.MAXDIM)))
            
        else:    
            Y_flat = self.model(*tensor_product_xz(self.xs_device, z))
            Y = Y_flat.reshape(z.shape[0], *(self.n_grid_points,)*self.config['nx'])
            Y_flat_cpu =  Y_flat.detach().cpu()
            Y_flat_cpu = Y_flat_cpu.reshape((z.shape[0], -1))
            if self.xs_inside_envelop is not None:
                Y_flat_cpu[:, ~self.xs_inside_envelop] = torch.inf
            Y_cpu = Y_flat_cpu.reshape(z.shape[0], *(self.n_grid_points,)*self.config['nx'])
            Y_cpu_np = Y_cpu.numpy()

            multi_res = [self.apply_async(calc_3d_ph_4d_array, (Y_cpu_np, i, self.MAXDIM)) for i in range(0, z.shape[0])]
        PHs = [res.get() for res in multi_res]

        list_loss_scc = []
        list_loss_super0 = []
        list_loss_sub0 = []
        for i in range(0, z.shape[0]):
            loss_scc, loss_super0, loss_sub0 = self.calc_ph_loss_crisper_slice(Y, z, PHs, i)
            list_loss_scc.append(loss_scc)
            list_loss_sub0.append(loss_sub0)
            list_loss_super0.append(loss_super0)
        return True, torch.stack(list_loss_scc).sum(), torch.stack(list_loss_sub0).mean(), torch.stack(list_loss_super0).mean()


    def filter_PH_iso(self, PHd: np.ndarray, lengthsd: np.ndarray):
        is_iso = (PHd[:,1] < -self.ISO) & (PHd[:,2] > self.ISO) #& (PHd[:,2] < 1.e100)
        return PHd[is_iso], lengthsd[is_iso]
    

    def filter_PH_super_iso(self, PHd: np.ndarray, lengthsd: np.ndarray):
        is_iso = (PHd[:,1] > self.ISO) & (PHd[:,2] > self.ISO) & (PHd[:,2] < 1.e100)
        return PHd[is_iso], lengthsd[is_iso]
    

    def filter_PH_sub_iso(self, PHd: np.ndarray, lengthsd: np.ndarray):
        is_iso = (PHd[:,1] < self.ISO) & (PHd[:,2] < self.ISO)
        return PHd[is_iso], lengthsd[is_iso]
    
    def filter_PH_inf_death(self, PHd: np.ndarray, lengthsd: np.ndarray):
        is_iso = (PHd[:,2] < 1.e100)
        return PHd[is_iso], lengthsd[is_iso]
