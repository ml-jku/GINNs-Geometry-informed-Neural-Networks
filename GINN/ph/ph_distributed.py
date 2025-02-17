import functools
from re import L
import torch 
import numpy as np
import cripser
from zmq import device

from util.model_utils import tensor_product_xz, tensor_product_xz_np

def calc_3d_ph_4d_array(Y_slice: np.ndarray, maxdim: int=0)-> np.ndarray:
    return cripser.computePH(Y_slice, maxdim=maxdim)

def calc_3d_ph_4d_array_full(Y_slice: np.ndarray, 
                        slice: int, 
                        maxdim: int,
                        z: np.ndarray, 
                        TARGET_Betti: int,
                        iso: float,
                        nx: int,
                        ph_loss_super0_points: bool,
                        ph_loss_sub0: bool,
                        HOLE_LEVEL: float,
                        )-> np.ndarray:
    ph = cripser.computePH(Y_slice, maxdim=maxdim)
    
    return calc_ph_loss_crisper_slice(z,
                               ph,
                               slice,
                               maxdim,
                               TARGET_Betti,
                               iso,
                               nx,
                               ph_loss_super0_points,
                               ph_loss_sub0,
                               HOLE_LEVEL,
    )
    

# TODO: cleanup: don't need slice if z is passed for current shape only
def calc_ph_loss_crisper_slice(z:np.ndarray, 
                               PH: np.ndarray,
                               slice: int,
                               maxdim: int,
                               target_Bettis: int,
                               iso: float,
                               nx: int,
                               ph_loss_super0_points: bool,
                               ph_loss_sub0: bool,
                               hole_level: float,
                               )-> list[np.ndarray]:
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
        

        x_list = []
        z_list = []
        loss_key_list = []
        loss_func_list = []

        for DIM in range(0, maxdim+1):

            TARGET_Betti = target_Bettis[DIM]
            PH_dim = PH[PH[:, 0] == DIM]
            lifetime = PH_dim[:, 2] - PH_dim[:, 1]
            sorted_ids = np.argsort(lifetime)[::-1]

            PH_dim = PH_dim[sorted_ids]
            lifetime = lifetime[sorted_ids]

            # disconnected components
            # death or birth isos
            # dim 1-3: 
            
            if DIM == 0:
                PH_filter, lifetime_filter = filter_PH_iso(PH_dim, lifetime, iso)
                death_isos = PH_filter[:, 6:6+nx].astype(int)
                birth_isos = PH_filter[:, 3:3+nx].astype(int)
                # if Y.requires_grad:
                #     loss_scc += torch.clamp(iso - Y[slice, *death_isos[TARGET_Betti:, :].T], max=0.).pow(2).sum()
                # else:
                x_idcs = death_isos[TARGET_Betti:, :]
                z_in = np.repeat(z[slice:slice+1], repeats=x_idcs.shape[0], axis=0)
                x_list.append(x_idcs)
                z_list.append(z_in)
                loss_key_list.append('loss_scc')
                loss_func_list.append(functools.partial(loss_scc_dim_0, iso=iso))
            
                #### super 0 points
                if ph_loss_super0_points:
                    raise NotImplementedError("Super loss outdated but kept for possible future use")
                    PH_filter, lifetime_filter = filter_PH_super_iso(PH_dim, lifetime, iso)
                    death_isos = PH_filter[:, 6:6+nx].astype(int)
                    birth_isos = PH_filter[:, 3:3+nx].astype(int)
                    # if Y.requires_grad:
                    #     loss_super0 += -(Y[slice, *birth_isos[1:, :].T] * torch.tensor(lifetime_filter[1:], device=self.device)).sum()
                    # else:
                    x_idcs = birth_isos[1:, :]
                    z_in = np.repeat(z[slice:slice+1], repeats=x_idcs.shape[0], axis=0)
                    x_list.append(x_idcs)
                    z_list.append(z_in)
                    loss_key_list.append('loss_super0')
                    loss_func_list.append(functools.partial(loss_super_dim_0, lifetime_filter=lifetime_filter))

                #### sub 0 points
                if ph_loss_sub0:
                    raise NotImplementedError("Sub loss outdated but kept for possible future use")
                    PH_filter, lifetime_filter = filter_PH_sub_iso(PH_dim, lifetime, iso)
                    death_isos = PH_filter[:, 6:6+nx].astype(int)
                    birth_isos = PH_filter[:, 3:3+nx].astype(int)
                    #loss += (2*torch.sigmoid(-(iso - Y[slice, *death_isos.T])*3*(1.01 + z[slice, 0]))).sum()
                    #loss += (2*torch.sigmoid(-(iso - Y[slice, *birth_isos.T])*3*(1.01 + z[slice, 0]))).sum()
                    # if Y.requires_grad:
                    #     loss_sub0 += (2*torch.sigmoid(-(iso - Y[slice, *death_isos.T])*30*(1.01 + z[slice, 0])) * torch.tensor(lifetime_filter, device=self.device)).sum()
                    # else:
                    x_idcs = death_isos
                    z_in = np.repeat(z[slice:slice+1], repeats=x_idcs.shape[0], axis=0)
                    x_list.append(x_idcs)
                    z_list.append(z_in)
                    loss_key_list.append('loss_sub0')
                    loss_func_list.append(functools.partial(loss_sub_dim_0, iso=iso, z=z, slice=slice, lifetime_filter=lifetime_filter))

            elif DIM == 1:
                PH_filter, lifetime_filter = filter_PH_inf_death(PH_dim, lifetime)
                death_isos = PH_filter[:, 6:6+nx].astype(int)
                birth_isos = PH_filter[:, 3:3+nx].astype(int)
                # if Y.requires_grad:
                #     loss_scc += (-HOLE_LEVEL - Y[slice, *birth_isos[0:TARGET_Betti, :].T]).pow(2).sum()
                #     loss_scc += (HOLE_LEVEL - Y[slice, *death_isos[0:TARGET_Betti, :].T]).pow(2).sum()
                # else:
                x_idcs = birth_isos[0:TARGET_Betti, :]
                z_in = np.repeat(z[slice:slice+1], repeats=x_idcs.shape[0], axis=0)
                x_list.append(x_idcs)
                z_list.append(z_in)
                loss_key_list.append('loss_scc')
                loss_func_list.append(functools.partial(loss_scc_dim_1_neg, hole_level=hole_level))
                
                x_idcs = death_isos[0:TARGET_Betti, :]
                z_in = np.repeat(z[slice:slice+1], repeats=x_idcs.shape[0], axis=0)
                x_list.append(x_idcs)
                z_list.append(z_in)
                loss_key_list.append('loss_scc')
                loss_func_list.append(functools.partial(loss_scc_dim_1_pos, hole_level=hole_level))
                    
                PH_filter, lifetime_filter = filter_PH_iso(PH_filter[TARGET_Betti:, :], lifetime_filter[TARGET_Betti:], iso)
                death_isos = PH_filter[:, 6:6+nx].astype(int)
                birth_isos = PH_filter[:, 3:3+nx].astype(int)
                # if Y.requires_grad:
                #     loss_scc += torch.clamp(iso - Y[slice, *death_isos[:, :].T], max=0.).pow(2).sum()
                # else:
                x_idcs = death_isos[:, :]
                z_in = np.repeat(z[slice:slice+1], repeats=x_idcs.shape[0], axis=0)
                x_list.append(x_idcs)
                z_list.append(z_in)
                loss_key_list.append('loss_scc')
                loss_func_list.append(functools.partial(loss_scc_dim_1, iso=iso))

            elif DIM == 2:
                raise NotImplementedError("DIM 2 loss not implemented")

        return x_list, z_list, loss_key_list, loss_func_list
    

## DIM 0
def loss_scc_dim_0(Y_grad: torch.Tensor, iso: float, **kwargs)-> torch.Tensor:
    return torch.clamp(iso - Y_grad, max=0.).pow(2).sum()

def loss_super_dim_0(Y_grad: torch.Tensor, lifetime_filter: np.ndarray, **kwargs)-> torch.Tensor:
    return -(Y_grad * torch.tensor(lifetime_filter[1:], device=Y_grad.device)).sum()
    
def loss_sub_dim_0(Y_grad: torch.Tensor, iso: float, z: torch.Tensor, slice: int, lifetime_filter: np.ndarray, **kwargs)-> torch.Tensor:
    # loss_func_list.append(lambda Y_grad: (2*torch.sigmoid(-(iso - Y_grad)*30*(1.01 + z[slice, 0])) * torch.tensor(lifetime_filter, device=Y_grad.device)).sum())
    return (2*torch.sigmoid(-(iso - Y_grad)*30*(1.01 + torch.tensor(z[slice, 0], device=Y_grad.device))) * torch.tensor(lifetime_filter, device=Y_grad.device)).sum() 

## DIM 1
def loss_scc_dim_1_neg(Y_grad: torch.Tensor, hole_level: float, **kwargs)-> torch.Tensor:
    return (-hole_level - Y_grad).pow(2).sum()

def loss_scc_dim_1_pos(Y_grad: torch.Tensor, hole_level: float, **kwargs)-> torch.Tensor:
    return (hole_level - Y_grad).pow(2).sum()

def loss_scc_dim_1(Y_grad: torch.Tensor, iso: float, **kwargs)-> torch.Tensor:
    return torch.clamp(iso - Y_grad, max=0.).pow(2).sum()
    

def filter_PH_iso(PHd: np.ndarray, lengthsd: np.ndarray, iso: float):
    is_iso = (PHd[:,1] < -iso) & (PHd[:,2] > iso) #& (PHd[:,2] < 1.e100)
    return PHd[is_iso], lengthsd[is_iso]


def filter_PH_super_iso(PHd: np.ndarray, lengthsd: np.ndarray, iso: float):
    is_iso = (PHd[:,1] > iso) & (PHd[:,2] > iso) & (PHd[:,2] < 1.e100)
    return PHd[is_iso], lengthsd[is_iso]


def filter_PH_sub_iso(PHd: np.ndarray, lengthsd: np.ndarray, iso: float):
    is_iso = (PHd[:,1] < iso) & (PHd[:,2] < iso)
    return PHd[is_iso], lengthsd[is_iso]

def filter_PH_inf_death(PHd: np.ndarray, lengthsd: np.ndarray):
    is_iso = (PHd[:,2] < 1.e100)
    return PHd[is_iso], lengthsd[is_iso]
