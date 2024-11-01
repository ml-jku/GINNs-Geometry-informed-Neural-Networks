import os
from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset

class SimJebDataset(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        self.device = config['device']
        self.data_dir = config['data_dir']
        self.pt_coords, self.sdf_vals, self.idcs = self._load_data(config)
        self.pt_coords, self.sdf_vals, self.idcs = torch.from_numpy(self.pt_coords).to(self.device), torch.from_numpy(self.sdf_vals).to(self.device), torch.from_numpy(self.idcs).to(self.device)
        print(f'n_data_points: {len(self.pt_coords)}')
        
    def _load_data(self, config):
        pts = []
        y = []
        idcs = []
        for i, simjeb_id in enumerate(config['simjeb_ids']):
            # load from path or cleaned path if exists
            clean_pts_file_path = join(self.data_dir, f"{simjeb_id}.obj_525000_points_cleaned.npy")
            if os.path.exists(clean_pts_file_path):
                pts_file_path = clean_pts_file_path
                sdf_val_path = join(self.data_dir, f"{simjeb_id}.obj_525000_sdf_cleaned.npy")
            else:
                pts_file_path = join(self.data_dir, f"{simjeb_id}.obj_525000_points.npy")
                sdf_val_path = join(self.data_dir, f"{simjeb_id}.obj_525000_sdf.npy")    
            pts_npy = np.load(pts_file_path)
            sdf_vals_npy = np.load(sdf_val_path)
            
            pts.append(pts_npy)
            y.append(sdf_vals_npy)
            idcs.append(np.ones(pts_npy.shape[0], dtype=np.int64) * i)
        
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