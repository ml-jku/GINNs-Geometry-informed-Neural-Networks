import os
from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset

class SimJebDataset(Dataset):
    def __init__(self, data_dir, simjeb_ids, seed, transform=None, **kwargs):
        self.device = torch.get_default_device()
        self.data_dir = data_dir
        self.pt_coords, self.sdf_vals, self.idcs = self._load_data(simjeb_ids)
        self.pt_coords, self.sdf_vals, self.idcs = torch.from_numpy(self.pt_coords).to(self.device), torch.from_numpy(self.sdf_vals).to(self.device), torch.from_numpy(self.idcs).to(self.device)
        print(f'n_data_points: {len(self.pt_coords)}')
        
    def _load_data(self, simjeb_ids):
        pts = []
        y = []
        idcs = []
        for i, simjeb_id in enumerate(simjeb_ids):
            
            pts_near_surf = np.load(join(self.data_dir, str(simjeb_id), f"points_near_surface.npz"))['data']
            pts_uniform = np.load(join(self.data_dir, str(simjeb_id), f"points_uniform.npz"))['data']
            sdf_vals_np = np.load(join(self.data_dir, str(simjeb_id), f"sdf_near.npz"))['data']
            sdf_uniform_np = np.load(join(self.data_dir, str(simjeb_id), f"sdf_uniform.npz"))['data']
            
            pts.extend([pts_near_surf, pts_uniform])
            y.extend([sdf_vals_np, sdf_uniform_np])
            idcs.append(np.ones(pts_near_surf.shape[0] + pts_uniform.shape[0], dtype=np.int64) * i)
        
        pts = np.concatenate(pts, axis=0)
        y = np.concatenate(y, axis=0)
        idcs = np.concatenate(idcs, axis=0)
        return pts, y, idcs

    def __len__(self):
        return len(self.sdf_vals)

    def __getitem__(self, idx):
        x = self.pt_coords[idx]
        y = self.sdf_vals[idx]
        shape_idx = self.idcs[idx]
        return x, y, shape_idx