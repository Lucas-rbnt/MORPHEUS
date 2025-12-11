# Third-party libraries
import torch
import torch.optim.optimizer
from torch.utils.data import DataLoader
from monai.utils import set_determinism
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Local dependencies
from morpheus.utils.data import get_classification_metadata, get_labelled_dataset
from morpheus.utils.parser import _init_parser_fewshot
from morpheus.utils.training_setup import _init_wsi_model
from morpheus.loops import loop_classification
from morpheus.datasets import WSIDataset


mtd_choices = ['abmil', 'tangle', 'transmil', 'deepsets', 'morpheusabmil', 'morpheusproto']
task_choices = ['brain', 'lung', 'breast', 'atrx', '1p19q', 'pancan', 'egfr', 'tp53']


if __name__ == '__main__':
    parser = _init_parser_fewshot()
    parser.add_argument('--mtd', type=str, default='transmil', choices=mtd_choices)
    parser.add_argument('--task', type=str, default='1p19q', choices=task_choices)
    args = parser.parse_args()
    args.batch_size = 32
    wandb_logging = True if args.entity is not None else False
    if wandb_logging:
        import wandb

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=f"wsi-clf-{args.mtd}-{args.task}-{args.epochs_pretrained}",
            reinit=True,
            config=vars(args),
        )
    
    set_determinism(seed=args.seed)
    metadata, mapping = get_classification_metadata(classification_task=args.task) 
    print("Number of samples:", metadata.shape[0])
    print('labels:', mapping) 
   
    if len(mapping) != 2:
        args.n_outputs = len(mapping)
    else:
        args.n_outputs = 1
  
    total_auc = []
    total_balanced_accuracy = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(metadata, metadata['label'])):
        print(f"Fold {fold}:")
        train_metadata = metadata.iloc[train_idx]
        val_metadata = metadata.iloc[val_idx]

        print("Train shape:", train_metadata.shape)
        print("Val shape:", val_metadata.shape)

        train_dataset = WSIDataset(train_metadata, mode='train', data_dir=args.data_dir)
        val_dataset = WSIDataset(val_metadata, mode='val', data_dir=args.data_dir)

        train_dataset = get_labelled_dataset(train_dataset, task='classification', label_col='label')
        val_dataset = get_labelled_dataset(val_dataset, task='classification', label_col='label')
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpus, persistent_workers=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.n_cpus, persistent_workers=True, pin_memory=True)
        
        model = _init_wsi_model(args)  
        # print number of trainable parameters
        print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)    
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        model, metrics = loop_classification(model=model, omics_modalities=[], train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optimizer, lr_scheduler=None, epochs=args.epochs, device=device, wandb_logging=wandb_logging, mapping=mapping)
        total_auc.append(metrics["val/auc"][-1])
        total_balanced_accuracy.append(metrics["val/balanced_accuracy"][-1])
        
    total_auc = np.array(total_auc)
    total_balanced_accuracy = np.array(total_balanced_accuracy)
    logs = {
        "average_auc_score": total_auc.mean(),
        "std_auc_score": total_auc.std(),
        "average_balanced_accuracy": total_balanced_accuracy.mean(),
        "std_balanced_accuracy": total_balanced_accuracy.std(),
    }
    print(logs)
    if wandb_logging:
        wandb.log(logs)
        run.finish()