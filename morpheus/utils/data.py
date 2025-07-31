import json
from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from morpheus.datasets import WrapperDataset
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

class SurvivalLabelProvider:
    def __init__(self, df, censorship_col, time_col, discrete_labels_col): 
        self.df = df
        self.censorship_col = censorship_col
        self.time_col = time_col
        self.discrete_labels_col = discrete_labels_col

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return {"time": row[self.time_col], "censorship": row[self.censorship_col], "label": row[self.discrete_labels_col]}

    def __len__(self):
        return len(self.df)
    
class ClassificationLabelProvider:
    def __init__(self, df, label_col):
        self.df = df
        self.label_col = label_col
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return {"label": row[self.label_col]}
    
    def __len__(self):
        return len(self.df)
    
    
def get_labelled_dataset(base_dataset, task=None, label_col=None, censorship_col=None, time_col=None) -> Dataset:
    if task == "survival":
        return WrapperDataset(base_dataset, SurvivalLabelProvider(base_dataset.dataframe, censorship_col=censorship_col, time_col=time_col, discrete_labels_col=label_col), task='survival')
    elif task == "classification":
        return WrapperDataset(base_dataset, ClassificationLabelProvider(base_dataset.dataframe, label_col=label_col))
    
def get_survival_labels(metadata, time_col, censorship_col, n_bins):
    metadata = metadata[metadata[time_col].notna()]
    from pycox.preprocessing.label_transforms import LabTransDiscreteTime
    labtrans = LabTransDiscreteTime(cuts=n_bins, scheme='quantiles')
    event = 1 - metadata[censorship_col].values.astype(int)
    labtrans.fit(metadata[time_col].values, event)
    trf = labtrans.transform(metadata[time_col].values, event)[0]
    metadata.insert(2, 'label', trf.astype(int))
    
    return metadata

def get_classification_metadata(classification_task):
    if classification_task == "brain":
        metadata = pd.read_csv("datasets_csv/metadata/tcga_gbmlgg.csv")
        metadata = metadata[metadata.split == "fine-tune"]
        metadata = metadata[metadata["IDH"].notna()]
        label_col = "IDH"

    if classification_task == "breast":
        metadata = pd.read_csv("datasets_csv/metadata/tcga_brca.csv")
        metadata = metadata[metadata.split == "fine-tune"]
        metadata = metadata[metadata["oncotree_code"].notna()]
        label_col = "oncotree_code"

    if classification_task == "lung":
        metadata = pd.read_csv("datasets_csv/metadata/tcga_pancan.csv")
        metadata = metadata[metadata.split == "fine-tune"]
        metadata = metadata[metadata["project_id"].isin(["TCGA-LUAD", "TCGA-LUSC"])]
        # create label based on if project-id is TCGA-LUAD or TCGA-LUSC
        metadata["label"] = metadata["project_id"].apply(
            lambda x: 0 if x == "TCGA-LUAD" else 1
        )
        label_mapping = {"luad": 0, "lusc": 1}

    
    if classification_task != "lung":
        label_mapping = {
            label: idx for idx, label in enumerate(metadata[label_col].unique())
        }
        metadata["label"] = metadata[label_col].map(label_mapping)

    return metadata, label_mapping

def get_omics(metadata, modalities=None, project_id=None):
    omics_df = OrderedDict()
    for modality in modalities:
        if modality == 'rna':
            rna_dataframe = pd.read_csv(f'datasets_csv/rna/hallmarks/tcga_{project_id}.csv', index_col=0)
            omics_df['rna'] = rna_dataframe[rna_dataframe.index.isin(metadata['submitter_id'].values)]
        elif modality == 'dnam':
            dnam_dataframe = pd.read_csv(f'datasets_csv/dnam/tcga_{project_id}.csv', index_col=0)
            dnam_dataframe = dnam_dataframe.fillna(0.)
            omics_df['dnam'] = dnam_dataframe[dnam_dataframe.index.isin(metadata['submitter_id'].values)]
        elif modality == 'cnv':
            cnv_dataframe = pd.read_csv(f'datasets_csv/cnv/tcga_{project_id}.csv', index_col=0)
            cnv_dataframe = cnv_dataframe.fillna(2.)
            cnv_dataframe = np.log10(cnv_dataframe/2 + 1)
            omics_df['cnv'] = cnv_dataframe[cnv_dataframe.index.isin(metadata['submitter_id'].values)]

    # ensure that all omics dataframes have the same shape as metadata
    for modality in omics_df.keys():
        if omics_df[modality].shape[0] != metadata.shape[0]:
            raise ValueError(f"Mismatch in number of samples for {modality}: {omics_df[modality].shape[0]} vs {metadata.shape[0]}")
            
    return omics_df
        
def prepare_omics_mapping(modalities):
    mapping_dicts = {}
    for modality in modalities:
        if modality == 'rna':
            with open('data/mappings/hallmarks_genes.json', 'r') as f:
                mapping_dict = json.load(f)
        elif modality == 'dnam':
            with open('data/mappings/dnam_chr_genes.json', 'r') as f:
                mapping_dict = json.load(f)
        elif modality == 'cnv':
            with open('data/mappings/cnv_chr_genes.json', 'r') as f:
                mapping_dict = json.load(f)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        mapping_dicts[modality.lower()] = mapping_dict
        
    return mapping_dicts


def compute_clf_metrics(labels, predictions):
    roc_auc = roc_auc_score(labels, predictions)
    balanced_accuracy = balanced_accuracy_score(labels, predictions > 0.5)
    
    return roc_auc, balanced_accuracy