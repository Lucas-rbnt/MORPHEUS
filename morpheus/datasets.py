from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from collections import OrderedDict
import os

class WSIDataset(Dataset):
    """
    Basis class for all experiments. Load extracted pre-trained embeddings
    """
    def __init__(self, dataframe, mode, data_dir, n_patches: int = 1024) -> None:
        super().__init__()
        self.dataframe = dataframe
        self.mode = mode
        self.n_patches = n_patches
        self.data_dir = data_dir
        
        
    def _select_random_patches(self, feats):
        idx_ = np.random.choice(feats.shape[1], self.n_patches, replace=False)
        idx_ = np.sort(idx_)
        return torch.from_numpy(feats[0, idx_])
    
    def _get_wsi_embedding(self, idx):
        sample = self.dataframe.iloc[idx]
        wsi_paths = [item.strip().strip("'").strip('"') for item in sample.WSI.strip("[]").split(',')]
        wsi_embeddings = []
        for wsi_path in wsi_paths:
            wsi_path = os.path.join(self.data_dir, wsi_path)
            with h5py.File(wsi_path, 'r') as f:
                wsi_feats = f['features'][:]
                wsi_embeddings.append(torch.from_numpy(wsi_feats))
        
        wsi_embeddings = torch.cat(wsi_embeddings, axis=1).squeeze(0)
        if self.mode != 'val':
            # if number of patches is less than n_patches, we need to oversample
            n_samples = min(wsi_embeddings.shape[0], self.n_patches)
            idx = np.sort(np.random.choice(wsi_embeddings.shape[0], n_samples, replace=False))
            wsi_embeddings = wsi_embeddings[idx, :]
            
            if n_samples < self.n_patches:
                remaining = self.n_patches - n_samples
                # resample again 
                idx = np.sort(np.random.choice(wsi_embeddings.shape[0], remaining, replace=True))
                extra = wsi_embeddings[idx, :]
                wsi_embeddings = torch.cat([wsi_embeddings, extra], dim=0)
            
            # shuffle the patches
            wsi_embeddings = wsi_embeddings[torch.randperm(wsi_embeddings.size(0))]
        
        return wsi_embeddings
                   
    def __getitem__(self, idx):
        wsi_embeddings = self._get_wsi_embedding(idx)
        return OrderedDict({'wsi': wsi_embeddings})
    
    def __len__(self):
        return len(self.dataframe)
            
            
class WSIRNADataset(WSIDataset):
    "Load tokenized data for WSI and RNA"
    def __init__(self, dataframe, rna_dataframe, rna_dict, mode, data_dir, n_patches: int = 1024) -> None:
        super().__init__(dataframe, mode, data_dir, n_patches)
        self.rna_dataframe = rna_dataframe
        self.rna_dict = rna_dict
        
    def _prepare_rna(self, submitter_id):
        patient_df = self.rna_dataframe.loc[submitter_id]
        rna = {k: torch.from_numpy(patient_df[self.rna_dict[k]].values).float().squeeze(0) for k in self.rna_dict.keys()}
        return rna
        
    def __getitem__(self, idx):
        submitter_id = self.dataframe.iloc[idx].submitter_id
        wsi_embeddings = self._get_wsi_embedding(idx)
        rna = self._prepare_rna(submitter_id)
        return OrderedDict({'wsi': wsi_embeddings, 'rna': rna})
    
    
class WSIRNADNAmDataset(WSIRNADataset):
    def __init__(self, dataframe, rna_dataframe, rna_dict, dnam_dataframe, dnam_dict, mode, data_dir, n_patches: int = 1024) -> None:
        super().__init__(dataframe, rna_dataframe, rna_dict, mode, data_dir, n_patches)
        self.dnam_dataframe = dnam_dataframe
        self.dnam_dict = dnam_dict
        
    def _prepare_dnam(self, submitter_id):
        patient_df = self.dnam_dataframe.loc[submitter_id]
        dnam = {k: torch.from_numpy(patient_df[self.dnam_dict[k]].values).float().squeeze(0) for k in self.dnam_dict.keys()}
        return dnam
    
    def __getitem__(self, idx):
        submitter_id = self.dataframe.iloc[idx].submitter_id
        wsi_embeddings = self._get_wsi_embedding(idx)
        rna = self._prepare_rna(submitter_id)
        dnam = self._prepare_dnam(submitter_id)
        
        return OrderedDict({'wsi': wsi_embeddings, 'rna': rna, 'dnam': dnam})
    
    
class WSIMultiOmicsDataset(WSIRNADNAmDataset):
    def __init__(self, dataframe, rna_dataframe, rna_dict, dnam_dataframe, dnam_dict, cnv_dataframe, cnv_dict, mode, data_dir, n_patches: int = 1024) -> None:
        super().__init__(dataframe, rna_dataframe, rna_dict, dnam_dataframe, dnam_dict, mode, data_dir, n_patches)
        self.cnv_dataframe = cnv_dataframe
        self.cnv_dict = cnv_dict
        
    def _prepare_cnv(self, submitter_id):
        patient_df = self.cnv_dataframe.loc[submitter_id]
        cnv = {k: torch.from_numpy(patient_df[self.cnv_dict[k]].values).float().squeeze(0) for k in self.cnv_dict.keys()}
        return cnv

    def __getitem__(self, idx):
        submitter_id = self.dataframe.iloc[idx].submitter_id
        wsi_embeddings = self._get_wsi_embedding(idx)
        rna = self._prepare_rna(submitter_id)
        dnam = self._prepare_dnam(submitter_id)
        cnv = self._prepare_cnv(submitter_id)
        
        return OrderedDict({'wsi': wsi_embeddings, 'rna': rna, 'dnam': dnam, 'cnv': cnv})
    
class WrapperDataset:
    """
    Wrapper class for datasets to add labels to the data.
    """
    def __init__(self, base_dataset, label_provider, task: str = 'survival'):
        self.base_dataset = base_dataset
        self.label_provider = label_provider
        self.task = task

    def __getitem__(self, index):
        item = self.base_dataset[index]
        if self.task == 'survival':
            item = {**item, **self.label_provider[index]}  
        else:
            item['label'] = self.label_provider[index]
        return item

    def __len__(self):
        return len(self.base_dataset)
    