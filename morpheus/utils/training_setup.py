from collections import OrderedDict
import torch
import math


def _init_wsi_model(args):
    from morpheus.tokenizers import WSITokenizer
    from morpheus.morpheus import MorpheusProtoEncoder

    wsi_tokenizer = WSITokenizer(
        embed_dim=args.hidden_size, num_proto=args.num_proto
    )
    model = MorpheusProtoEncoder(
        n_outputs=args.n_outputs,
        wsi_tokenizer=wsi_tokenizer,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        mlp_dim=args.mlp_dim,
        num_heads=args.num_heads,
    )

    print("Using pretrained model")
    state_dict = torch.load(
        f"pretrained_models/morpheus.pth"
    )
    state_dict = {k: v for k, v in state_dict.items() if "decoders" not in k}
    state_dict = {
        k: v for k, v in state_dict.items() if "omics_tokenizers" not in k
    }
    model.load_state_dict(state_dict, strict=False)

    return model


def _init_wsi_rna_model(args, rna_dict):
    from morpheus.morpheus import MorpheusProtoEncoder
    from morpheus.tokenizers import OmicsTokenizer, WSITokenizer

    wsi_tokenizer = WSITokenizer(
        embed_dim=args.hidden_size, num_proto=args.num_proto
    )
    omics_tokenizer = {
        "rna": OmicsTokenizer(rna_dict, output_dim=args.hidden_size, hidden=[256])
    }
    model = MorpheusProtoEncoder(
        n_outputs=args.n_outputs,
        wsi_tokenizer=wsi_tokenizer,
        omics_tokenizers=omics_tokenizer,
        omics_modalities=["rna"],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        mlp_dim=args.mlp_dim,
        num_heads=args.num_heads,
    )
    
    print("Using pretrained model")
    state_dict = torch.load(
        f"pretrained_models/morpheus.pth"
    )
    # remove decoder from state dict
    state_dict = {k: v for k, v in state_dict.items() if "decoders" not in k}
    model.load_state_dict(state_dict, strict=False)

    return model


def _init_full_model(args, dataset, mapping_dicts):
    from morpheus.morpheus import MorpheusProtoEncoder
    from morpheus.tokenizers import OmicsTokenizer, WSITokenizer

    wsi_tokenizer = WSITokenizer(
        embed_dim=args.hidden_size, num_proto=args.num_proto
    )
    omics_tokenizers = OrderedDict(
        (
            modality,
            OmicsTokenizer(
                mapping_dicts[modality], output_dim=args.hidden_size, hidden=[256]
            ),
        )
        for modality in mapping_dicts.keys()
    )
    model = MorpheusProtoEncoder(
        n_outputs=args.n_outputs,
        wsi_tokenizer=wsi_tokenizer,
        omics_tokenizers=omics_tokenizers,
        omics_modalities=["rna", "dnam", "cnv"],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        mlp_dim=args.mlp_dim,
        num_heads=args.num_heads,
    )
    print("Using pretrained model")
    state_dict = torch.load(
        f"pretrained_models/morpheus.pth"
    )
    model.load_state_dict(state_dict, strict=False)

    return model


def _init_optimizer_and_scheduler(model, args):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_func = lambda epoch: min(
        (epoch + 1) / (args.warmup_epochs + 1e-8),
        0.5 * (math.cos(epoch / args.scaling_factor * math.pi) + 1),
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return optimizer, lr_scheduler
