from collections import OrderedDict
import torch
import math
    
def _init_wsi_model(args):
    if args.mtd == 'morpheusabmil':
        from morpheus.models.abmil import ABMIL
        model = ABMIL(hidden_dim=args.hidden_size, classification=True, n_outputs=args.n_outputs, dropout=args.dropout_rate)
        if args.epochs_pretrained in [100, 200, 300, 400]:
            print('Using pretrained model')
            state_dict = torch.load(f'pretrained_models/morpheusABMIL.pth')
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'wsi_abmil' in k:
                    new_state_dict[k.replace('wsi_abmil.', '')] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict, strict=False)
                    
    if args.mtd == 'morpheusproto':
        from morpheus.tokenizers import WSITokenizer
        from morpheus.morpheus import MorpheusProtoEncoder
        wsi_tokenizer = WSITokenizer(embed_dim=args.hidden_size, num_proto=args.num_proto)
        model = MorpheusProtoEncoder(n_outputs=args.n_outputs, wsi_tokenizer=wsi_tokenizer, 
                                     hidden_size=args.hidden_size, num_layers=args.num_layers, dropout_rate=args.dropout_rate,
                                     mlp_dim=args.mlp_dim, num_heads=args.num_heads)
        
        if args.epochs_pretrained in [100, 200, 300, 400]:
            print('Using pretrained model')
            state_dict = torch.load(f'pretrained_models/morpheusPROTO.pth') 
            model.load_state_dict(state_dict, strict=False)                
                                     
    return model

def _init_wsi_rna_model(args, rna_dict):
    if args.mtd == 'morpheusabmil':
        from morpheus.morpheus import MorpheusABMILEncoder
        from morpheus.models.abmil import ABMIL
        from morpheus.tokenizers import OmicsTokenizer
        wsi_abmil = ABMIL(hidden_dim=args.hidden_size, dropout=args.dropout_rate)
        omics_tokenizer = {'rna': OmicsTokenizer(rna_dict, output_dim=args.hidden_size, hidden=[256])}
        model = MorpheusABMILEncoder(wsi_abmil=wsi_abmil, n_outputs=args.n_outputs, hidden_size=args.hidden_size, omics_tokenizers=omics_tokenizer, omics_modalities=['rna'], num_layers=args.num_layers, dropout_rate=args.dropout_rate)
        if args.epochs_pretrained in [100, 200, 300, 400]:
            print('Using pretrained model')
            state_dict = torch.load(f'pretrained_models/morpheusABMIL.pth')
            model.load_state_dict(state_dict, strict=False)
            
    if args.mtd == 'morpheusproto':
        from morpheus.morpheus import MorpheusProtoEncoder
        from morpheus.tokenizers import OmicsTokenizer, WSITokenizer
        wsi_tokenizer = WSITokenizer(embed_dim=args.hidden_size, num_proto=args.num_proto)
        omics_tokenizer = {'rna': OmicsTokenizer(rna_dict, output_dim=args.hidden_size, hidden=[256])}
        model = MorpheusProtoEncoder(n_outputs=args.n_outputs, wsi_tokenizer=wsi_tokenizer, omics_tokenizers=omics_tokenizer, omics_modalities=['rna'], hidden_size=args.hidden_size, num_layers=args.num_layers, dropout_rate=args.dropout_rate, mlp_dim=args.mlp_dim, num_heads=args.num_heads)
        if args.epochs_pretrained in [100, 200, 300, 400]:
            print('Using pretrained model')
            state_dict = torch.load(f'pretrained_models/morpheusPROTO.pth')
            model.load_state_dict(state_dict, strict=False)
                
    return model

def _init_full_model(args, dataset, mapping_dicts):
    if args.mtd == 'morpheusabmil':
        from morpheus.morpheus import MorpheusABMILEncoder
        from morpheus.models.abmil import ABMIL
        from morpheus.tokenizers import OmicsTokenizer
        wsi_abmil = ABMIL(hidden_dim=args.hidden_size, dropout=args.dropout_rate)
        omics_tokenizers = OrderedDict((modality, OmicsTokenizer(mapping_dicts[modality], output_dim=args.hidden_size, hidden=[256])) for modality in mapping_dicts.keys())
        model = MorpheusABMILEncoder(wsi_abmil=wsi_abmil, n_outputs=args.n_outputs, hidden_size=args.hidden_size, omics_tokenizers=omics_tokenizers, omics_modalities=['rna', 'dnam', 'cnv'], num_layers=args.num_layers, dropout_rate=args.dropout_rate)
        if args.epochs_pretrained in [100, 200, 300, 400]:
            print('Using pretrained model')
            state_dict = torch.load(f'pretrained_models/morpheusABMIL.pth')
            model.load_state_dict(state_dict, strict=False)
            
    if args.mtd == 'morpheusproto':
        from morpheus.morpheus import MorpheusProtoEncoder
        from morpheus.tokenizers import OmicsTokenizer, WSITokenizer
        wsi_tokenizer = WSITokenizer(embed_dim=args.hidden_size, num_proto=args.num_proto)
        omics_tokenizers = OrderedDict((modality, OmicsTokenizer(mapping_dicts[modality], output_dim=args.hidden_size, hidden=[256])) for modality in mapping_dicts.keys())
        model = MorpheusProtoEncoder(n_outputs=args.n_outputs, wsi_tokenizer=wsi_tokenizer, omics_tokenizers=omics_tokenizers, omics_modalities=['rna', 'dnam', 'cnv'], hidden_size=args.hidden_size, num_layers=args.num_layers, dropout_rate=args.dropout_rate, mlp_dim=args.mlp_dim, num_heads=args.num_heads)
        if args.epochs_pretrained in [100, 200, 300, 400]:
            print('Using pretrained model')
            state_dict = torch.load(f'pretrained_models/morpheusPROTO.pth')
            model.load_state_dict(state_dict, strict=False)
            
    return model
        
def _init_optimizer_and_scheduler(model, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_func = lambda epoch: min(
        (epoch + 1) / (args.warmup_epochs + 1e-8),
            0.5 * (math.cos(epoch / args.scaling_factor * math.pi) + 1),
        )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    
    return optimizer, lr_scheduler
