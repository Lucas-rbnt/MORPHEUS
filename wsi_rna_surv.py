# Third-party libraries
import torch
import torch.optim.optimizer
from torch.utils.data import DataLoader
import pandas as pd
from monai.utils import set_determinism
import numpy as np
from sklearn.model_selection import KFold

# Local dependencies
from morpheus.utils.data import get_survival_labels, get_labelled_dataset, prepare_omics_mapping, get_omics
from morpheus.utils.parser import _init_parser
from morpheus.utils.training_setup import _init_wsi_rna_model, _init_optimizer_and_scheduler
from morpheus.loops import loop_survival
from morpheus.datasets import WSIRNADataset


mtd_choices = ['mome', 'survpath', 'motcat', 'mcat', 'morpheusabmil', 'morpheusproto']


if __name__ == '__main__':
    parser = _init_parser()
    parser.add_argument('--mtd', type=str, default='morpheusabmil', choices=mtd_choices)
    parser.add_argument("--project_id", type=str, default='gbmlgg', help="Project ID for the dataset")
    parser.add_argument("--n_outputs", type=int, default=4, help="Number of bins for survival analysis")
    parser.add_argument("--time_col", type=str, default='survival_months_dss', help="Time column for survival analysis")
    parser.add_argument("--censorship_col", type=str, default='censorship_dss', help="Censorship column for survival analysis")
    args = parser.parse_args()
    wandb_logging = True if args.entity is not None else False
    if wandb_logging:
        import wandb

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=f"wsi-rna-surv-{args.mtd}-{args.project_id}-{args.epochs_pretrained}",
            reinit=True,
            config=vars(args),
        )
    
    set_determinism(seed=args.seed)
    metadata = pd.read_csv(f'datasets_csv/metadata/tcga_{args.project_id.lower()}.csv')
    metadata = metadata.dropna(subset=['rna'])
    metadata = metadata[metadata['split'] == 'fine-tune']
    metadata = get_survival_labels(metadata, time_col=args.time_col, censorship_col=args.censorship_col, n_bins=args.n_outputs)   
    assert (metadata.split == 'fine-tune').all(), "Metadata should only contain fine-tune split"
    print("Number of samples:", metadata.shape[0])
    
    total_c_index = []
    for fold in range(5):
        print(f"Fold {fold}:")
        train_metadata = metadata[metadata.folds != fold]
        val_metadata = metadata[metadata.folds == fold]
        
        print(f"Train shape fold {fold}: ", train_metadata.shape)
        print(f"Val shape fold {fold}: ", val_metadata.shape)
        
        mapping_dicts = prepare_omics_mapping(['rna'])

        omics_train = get_omics(metadata=train_metadata, modalities=['rna'], project_id=args.project_id)
        omics_val = get_omics(metadata=val_metadata, modalities=['rna'], project_id=args.project_id)

        train_dataset = WSIRNADataset(train_metadata, rna_dataframe=omics_train['rna'], rna_dict=mapping_dicts['rna'], mode='train', data_dir=args.data_dir)
        val_dataset = WSIRNADataset(val_metadata,  rna_dataframe=omics_val['rna'], rna_dict=mapping_dicts['rna'], mode='val', data_dir=args.data_dir)
        
        train_dataset = get_labelled_dataset(train_dataset, task='survival', time_col=args.time_col, censorship_col=args.censorship_col, label_col='label')
        val_dataset = get_labelled_dataset(val_dataset, task='survival', time_col=args.time_col, censorship_col=args.censorship_col, label_col='label')
            
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpus, persistent_workers=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.n_cpus, persistent_workers=True, pin_memory=True)
        
        model = _init_wsi_rna_model(args, rna_dict=mapping_dicts['rna'])    
        # print number of trainable parameters
        print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)    
        
        optimizer, lr_scheduler = _init_optimizer_and_scheduler(model, args)
        if args.batch_size == 1:
            print("here")
            accum_iter = 32
        else:
            accum_iter = None
            
        model, metrics = loop_survival(model=model, omics_modalities=['rna'], train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optimizer, lr_scheduler=lr_scheduler, epochs=args.epochs, device=device, wandb_logging=wandb_logging, accum_iter=accum_iter)
        total_c_index.append(metrics["val/c_index"][-1])
        
    total_c_index = np.array(total_c_index)
    logs = {
        "average_c_index": total_c_index.mean(),
        "std_c_index": total_c_index.std(),

    }
    print(logs)
    if wandb_logging:
        wandb.log(logs)
        run.finish()
