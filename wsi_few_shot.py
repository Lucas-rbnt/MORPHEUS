# Third-party libraries
import torch
import torch.optim.optimizer
from torch.utils.data import DataLoader
from monai.utils import set_determinism
import numpy as np

# Local dependencies
from morpheus.utils.data import get_classification_metadata, get_labelled_dataset
from morpheus.utils.parser import _init_parser_fewshot
from morpheus.utils.training_setup import _init_wsi_model
from morpheus.loops import loop_classification
from morpheus.datasets import WSIDataset


mtd_choices = ['abmil', 'tangle', 'transmil', 'deepsets', 'morpheusabmil', 'morpheusproto']
task_choices = ['brain', 'lung', 'breast', 'egfr', 'tp53', 'mgmt', 'response']


if __name__ == '__main__':
    parser = _init_parser_fewshot()
    parser.add_argument('--mtd', type=str, default='morpheusproto', choices=mtd_choices)
    parser.add_argument('--task', type=str, default='brain', choices=task_choices)
    args = parser.parse_args()
    wandb_logging = True if args.entity is not None else False
    if wandb_logging:
        import wandb

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=f"wsi-few-shot-{args.mtd}-{args.task}-{args.epochs_pretrained}-{args.k}",
            reinit=True,
            config=vars(args),
        )
    
    set_determinism(seed=args.seed)
    metadata, mapping = get_classification_metadata(classification_task=args.task) 
    # ensure all metadata is fine-tune
    assert (metadata.split == 'fine-tune').all(), "Metadata should only contain fine-tune split"
    print("Number of samples:", metadata.shape[0])
    print('labels:', mapping) 
    args.n_outputs = 1
  
    total_auc = []
    total_balanced_accuracy = []
    for i in range(args.n_sampling):
        print(f"Sampling iteration {i+1}/{args.n_sampling}")
        
        train_metadata = (
        metadata.groupby('label', group_keys=False)
        .sample(n=args.k, random_state=args.seed+i)
            )
        
        val_metadata = metadata.drop(train_metadata.index)
        print(f"Train shape: {train_metadata.shape}")
        print(f"Val shape: {val_metadata.shape}")

        train_dataset = WSIDataset(train_metadata, mode='train', data_dir=args.data_dir)
        val_dataset = WSIDataset(val_metadata, mode='val', data_dir=args.data_dir)
        
        train_dataset = get_labelled_dataset(train_dataset, task='classification', label_col='label')
        val_dataset = get_labelled_dataset(val_dataset, task='classification', label_col='label')

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpus, persistent_workers=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.n_cpus, persistent_workers=True, pin_memory=True)

        model = _init_wsi_model(args)
        print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)   

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model, metrics = loop_classification(model=model, omics_modalities=[],train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optimizer, lr_scheduler=None, epochs=args.epochs, device=device, wandb_logging=wandb_logging) 
        total_auc.append(metrics["val/auc"][-1])
        total_balanced_accuracy.append(metrics["val/balanced_accuracy"][-1])

total_auc = np.array(total_auc)
total_balanced_accuracy = np.array(total_balanced_accuracy)
print(len(total_auc), len(total_balanced_accuracy))
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
    