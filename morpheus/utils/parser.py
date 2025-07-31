import argparse

def _common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument("--n_cpus", type=int, default=16, help="Number of cpus to use for data loading")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for training")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads for the model")
    parser.add_argument("--mlp_dim", type=int, default=256, help="MLP dimension for the model")
    parser.add_argument("--dropout_rate", type=float, default=0.35, help="Dropout rate for the model")
    parser.add_argument("--num_proto", type=int, default=32, help="Number of prototypes for the model")
    parser.add_argument("--entity", type=str, default=None, help="Entity for wandb logging")
    parser.add_argument("--project", type=str, default="morpheus", help="Project for wandb logging")
    parser.add_argument("--seed", type=int, default=1999, help="Seed for reproducibility")
    
    return parser

def _init_parser_fewshot():
    parser = _common_parser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--k", type=int, default=5, help="Number of support samples per class")
    parser.add_argument("--n_sampling", type=int, default=10, help="Number of sampling")
    
    return parser

def _init_parser():
    parser = _common_parser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs for the model")
    parser.add_argument("--scaling_factor", type=int, default=25, help="Scaling factor for the model")
    
    return parser