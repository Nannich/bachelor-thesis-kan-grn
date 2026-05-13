import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils import *

class SingleCellDataset(Dataset):
    def __init__(self, trajectories, counts):
        self.inputs = torch.tensor(trajectories, dtype=torch.float32)
        self.targets = torch.tensor(counts, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def get_dataloaders(adata, pseudotime, weights, target_gene=None, batch_size=64, train_ratio=0.8):
    """ Creates train/test dataloaders from AnnData and Slingshot results. """
    n_cells = adata.n_obs
    indices = np.random.permutation(n_cells)
    split_idx = int(n_cells * train_ratio)
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    train_pt_values = pseudotime[train_indices]
    pt_min = train_pt_values.min(keepdims=True)
    pt_max = train_pt_values.max(keepdims=True)
    
    pt_scaled = (pseudotime - pt_min) / (pt_max - pt_min + 1e-8)
    trajectories = np.hstack((pt_scaled, weights))

    raw_counts = get_raw_counts(adata)

    if target_gene is not None:
        if isinstance(target_gene, str):
            gene_idx = adata.raw.var_names.get_loc(target_gene)
            count_values = np.round(raw_counts[:, [gene_idx]])
        else:
            count_values = np.round(raw_counts[:, [target_gene]])
    else:
        count_values = np.round(raw_counts)
    
    train_dataset = SingleCellDataset(trajectories[train_indices], count_values[train_indices])
    test_dataset = SingleCellDataset(trajectories[val_indices], count_values[val_indices])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = trajectories.shape[1]
    output_dim = count_values.shape[1] * 3 

    return train_dataloader, test_dataloader, input_dim, output_dim, pt_min, pt_max

def get_eval_dataloader(adata, pseudotime, weights, pt_min, pt_max, target_gene=None, batch_size=256):
    """ Creates an evaluation dataloader for the full dataset using existing scaling bounds. """
    pt_scaled = (pseudotime - pt_min) / (pt_max - pt_min + 1e-8)
    trajectories = np.hstack((pt_scaled, weights))

    raw_counts = get_raw_counts(adata)

    if target_gene is not None:
        if isinstance(target_gene, str):
            gene_idx = adata.raw.var_names.get_loc(target_gene)
            count_values = np.round(raw_counts[:, [gene_idx]])
        else:
            count_values = np.round(raw_counts[:, [target_gene]])
    else:
        count_values = np.round(raw_counts)
    
    full_dataset = SingleCellDataset(trajectories, count_values)
    full_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    return full_dataloader