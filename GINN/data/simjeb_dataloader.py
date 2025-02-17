
import torch

class SimJebDataloader:
    '''
    The dataloader for the SimJeb dataset loads the data in memory.
    '''
    
    def __init__(self, dataset, n_points) -> None:
        self.device = torch.get_default_device()
        self.data = dataset
        self.shuffle_idcs = torch.randperm(len(self.data), device=self.device)
        self.cur_data_idx = 0
        
        self.data.pt_coords = self.data.pt_coords.to(self.device)
        self.data.sdf_vals = self.data.sdf_vals.to(self.device)
        self.data.idcs = self.data.idcs.to(self.device)
        
        self.n_points = n_points
        
    def __len__(self):
        return len(self.dataset)
    
    def get_data_batch(self):
        # NOTE: the dataset has points of several shapes concatenated together (can't be distinguished anymore)
        n_points_per_batch = self.n_points if self.n_points > 0 else len(self.data)
        assert len(self.data) % n_points_per_batch == 0, f"len(data)={len(self.data)} must be divisible by ginn_bsize={n_points_per_batch}; otherwise the last batch will be smaller"
        
        start_idx = self.cur_data_idx
        end_idx = min(start_idx + n_points_per_batch, len(self.data))
        
        perm_idcs = self.shuffle_idcs[start_idx:end_idx]
        pts = self.data.pt_coords[perm_idcs]
        sdf = self.data.sdf_vals[perm_idcs]
        idcs = self.data.idcs[perm_idcs]
        
        if self.cur_data_idx + n_points_per_batch > len(self.data):
            start_idx = 0
            end_idx = n_points_per_batch - (len(self.data) - self.cur_data_idx)
            perm_idcs = self.shuffle_idcs[start_idx:end_idx]
            pts = torch.cat([pts, self.data.pt_coords[perm_idcs]])
            sdf = torch.cat([sdf, self.data.sdf_vals[perm_idcs]])
            idcs = torch.cat([idcs, self.data.idcs[perm_idcs]])
            self.shuffle_idcs = torch.randperm(len(self.data), device=self.device)
            
        self.cur_data_idx = end_idx % len(self.data)
        return pts, sdf, idcs
        
        
        
    
