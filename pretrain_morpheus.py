# Standard libraries
import argparse
from collections import OrderedDict

# Third-party libraries
import torch
import torch.optim.optimizer
from torch.utils.data import DataLoader
import pandas as pd
from monai.utils import set_determinism

# Local dependencies
from morpheus.morpheus import MorpheusProto, MorpheusABMIL
from morpheus.decoders import MorpheusDecoder
from morpheus.tokenizers import OmicsTokenizer
from morpheus.datasets import WSIMultiOmicsDataset
from morpheus.reverse_tokenizers import OmicsReverseTokenizer
from morpheus.utils.data import get_omics, prepare_omics_mapping
from morpheus.utils.training_setup import _init_optimizer_and_scheduler
from morpheus.loops import loop_pretrain


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument("--omics_modalities", nargs="+", default=["rna", "dnam", "cnv"], help="List of modalities to use for training")
    parser.add_argument("--project_id", type=str, default="pancan")
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_layers_decoders', type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--wsi_type", type=str, default="abmil", help="Type of WSI model to use")
    parser.add_argument("--n_cpus", type=int, default=16, help="Number of cpus to use for data loading")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for training")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads for the model")
    parser.add_argument("--mlp_dim", type=int, default=256, help="MLP dimension for the model")
    parser.add_argument("--decoder_num_heads", type=int, default=8, help="Number of heads for the decoder")
    parser.add_argument("--dropout_rate", type=float, default=0.15, help="Dropout rate for the model")
    parser.add_argument("--masking_ratio", type=float, default=0.75, help="Masking ratio for the model")
    parser.add_argument("--num_proto", type=int, default=32, help="Number of prototypes for the model")
    parser.add_argument("--entity", type=str, default=None, help="Entity for wandb logging")
    parser.add_argument("--project", type=str, default="morpheus", help="Project for wandb logging")
    parser.add_argument("--seed", type=int, default=1999, help="Seed for reproducibility")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs for the model")
    parser.add_argument("--scaling_factor", type=int, default=500, help="Scaling factor for the model")
    args = parser.parse_args()
    wandb_logging = True if args.entity is not None else False
    set_determinism(seed=args.seed)
    metadata = pd.read_csv(f'datasets_csv/metadata/tcga_{args.project_id}.csv')
    metadata = metadata.dropna(subset=args.omics_modalities)
    if wandb_logging:
        import wandb
        run = wandb.init(project=args.project, entity=args.entity, reinit=True, config=vars(args), name=f"morpheus{args.wsi_type.upper()}_{args.project_id}_mod={len(args.omics_modalities)}_lay={args.num_layers}")
    
    train_metadata = metadata[metadata['split'] == 'pretrain']
    val_metadata  = metadata[metadata['split'] == 'val-pretrain']
    omics_train = get_omics(metadata=train_metadata, modalities=args.omics_modalities, project_id=args.project_id)
    omics_val = get_omics(metadata=val_metadata, modalities=args.omics_modalities, project_id=args.project_id)
    # print shape of omics
    print("Metadata shapes: ", train_metadata.shape)
    print(f"Omics train shapes: {', '.join([f'{modality}: {omics_train[modality].shape}' for modality in args.omics_modalities])}")
    mapping_dicts = prepare_omics_mapping(args.omics_modalities)
    
    train_dataset = WSIMultiOmicsDataset(train_metadata, rna_dataframe=omics_train['rna'], rna_dict=mapping_dicts['rna'], dnam_dataframe=omics_train['dnam'], dnam_dict=mapping_dicts['dnam'], cnv_dataframe=omics_train['cnv'], cnv_dict=mapping_dicts['cnv'], mode='train', data_dir=args.data_dir)
    val_dataset = WSIMultiOmicsDataset(val_metadata, rna_dataframe=omics_val['rna'], rna_dict=mapping_dicts['rna'], dnam_dataframe=omics_val['dnam'], dnam_dict=mapping_dicts['dnam'], cnv_dataframe=omics_val['cnv'], cnv_dict=mapping_dicts['cnv'], mode='train', data_dir=args.data_dir)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpus, persistent_workers=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpus, persistent_workers=True, pin_memory=True)

    omics_tokenizers = OrderedDict((modality, OmicsTokenizer(mapping_dicts[modality], output_dim=args.hidden_size, hidden=[256])) for modality in args.omics_modalities)
        
    reverse_tokenizers = OrderedDict((modality, OmicsReverseTokenizer(mapping_dicts[modality], input_dim=args.hidden_size, hidden=[256])) for modality in args.omics_modalities)
        
    omics_decoders = OrderedDict((modality, MorpheusDecoder(reverse_tokenizers[modality], modality=modality, hidden_size=args.hidden_size, num_layers=args.num_layers_decoders, num_heads=args.decoder_num_heads)) for modality in args.omics_modalities)
        
    if args.wsi_type == 'proto':
        from morpheus.tokenizers import WSITokenizer
        wsi_tokenizer = WSITokenizer(embed_dim=args.hidden_size, num_proto=args.num_proto)
        model = MorpheusProto(wsi_tokenizer, omics_tokenizers=omics_tokenizers, omics_decoders=omics_decoders, omics_modalities=args.omics_modalities, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout_rate=args.dropout_rate, masking_ratio=args.masking_ratio, mlp_dim=args.mlp_dim, num_heads=args.num_heads)
    else:
        from morpheus.models.abmil import ABMIL
        wsi_tokenizer = ABMIL(hidden_dim=args.hidden_size)
        model = MorpheusABMIL(wsi_tokenizer, omics_tokenizers=omics_tokenizers, omics_decoders=omics_decoders, omics_modalities=args.omics_modalities, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout_rate=args.dropout_rate, mlp_dim=args.mlp_dim, num_heads=args.num_heads)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer, lr_scheduler = _init_optimizer_and_scheduler(model, args)
        
    base_path = f"pretrained_models/morpheus{args.wsi_type.upper()}_{args.project_id}_mod={len(args.omics_modalities)}_layers={args.num_layers}_epochs="
    model, metrics = loop_pretrain(model=model, omics_modalities=args.omics_modalities, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optimizer, lr_scheduler=lr_scheduler, epochs=args.epochs, device=device, base_path_save=base_path, wandb_logging=wandb_logging)
        
    if wandb_logging:
        run.finish()