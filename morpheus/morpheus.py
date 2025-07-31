import torch.nn as nn
import torch
from .utils.ml_utils import TransformerBlock
from torch.distributions import Dirichlet
from monai.networks.layers import trunc_normal_

from collections import OrderedDict


def masking_func(omics_tokens, number_of_tokens_to_keep):
    """
    Create a mask for the omics tokens to randomly select a subset of tokens.
    Inspired by MultiMAE and 4M -> https://github.com/apple/ml-4m

    Args:
        omics_tokens (dict): Dictionary of omics tokens.
        number_of_tokens_to_keep (int): Number of tokens to keep.

    Returns:
        dict: Masked omics tokens.
    """
    bs = list(omics_tokens.values())[0].shape[0]
    device = list(omics_tokens.values())[0].device
    alphas = [1.] * len(omics_tokens)
    encoded_tokens_per_omics = OrderedDict((key, values.shape[1]) for key, values in omics_tokens.items())
    omics_sampling_dist = Dirichlet(torch.tensor(alphas)).sample((bs,)).to(device) # first sample
    while True:
        # ensure that the sampling can be done
        samples_per_omics = (omics_sampling_dist * number_of_tokens_to_keep).round().long()
        invalid_idx = samples_per_omics.sum(dim=1) != number_of_tokens_to_keep # check if the sampling is valid
        prop_constraint = (samples_per_omics > torch.tensor(list(encoded_tokens_per_omics.values())).to(device)).sum(1).bool()
        # ensure that the sampling for a given modality do not exceed the number of tokens of this modality, useful when using lower mask ratios
        invalid_idx = invalid_idx | prop_constraint
        if not invalid_idx.any():
            break
        resample = Dirichlet(torch.tensor(alphas)).sample((invalid_idx.sum(),)).to(device) # resample erroneous indices
        omics_sampling_dist[invalid_idx] = resample
        
    samples_per_omics = (omics_sampling_dist * number_of_tokens_to_keep).round().long() # final sampling
    omics_masks = OrderedDict()
    for i, (key, n_t) in enumerate(encoded_tokens_per_omics.items()):
        # sample randomly the given number of tokens for each modality
        noise = torch.rand(bs, n_t, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.arange(n_t, device=device).unsqueeze(0).expand(bs, -1)
        mask = torch.gather(mask, dim=1, index=ids_shuffle) # this shuffles indices
        mask = torch.where(mask < samples_per_omics[:, i].unsqueeze(1), 0, 1) # for element with value < samples_per_omics, set to 0 (keep it)
        omics_masks[key] = mask
            
    return omics_masks

def generate_omics_info(omics_masks):
    """
    Generate the omics information that will be given to the decoder to work properly

    Args:
        omics_masks (dict): contains the masks for each modality

    Returns:
        dict: Information about each modality containing the number of tokens, 
              start and end indices of the whole sequence for the decoder.
    """
    omics_info = OrderedDict()
    i = 0
    for modality, tensor in omics_masks.items():
        n_tokens = tensor.shape[1]
        omics_info[modality] = {
            'n_t': n_tokens,
            'start': i,
            'end': i + n_tokens,
        }
        i += n_tokens
    
    return omics_info
    

class MorpheusProto(nn.Module):
    def __init__(self, wsi_tokenizer, omics_tokenizers, omics_decoders, omics_modalities, hidden_size=256, masking_ratio=0.75, mlp_dim=256, num_heads=8, num_layers=2, dropout_rate=0.1):
        super(MorpheusProto, self).__init__()
        self.omics_modalities = omics_modalities
        self.wsi_tokenizer = wsi_tokenizer
        self.omics_tokenizers = nn.ModuleDict(omics_tokenizers)
        self.omics_decoders = nn.ModuleDict(omics_decoders)
        self.masking_ratio = masking_ratio
        self.norm = nn.LayerNorm(hidden_size)
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                qkv_bias=True,
            )
            for _ in range(num_layers)
        ])
        
        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        trunc_normal_(self.global_token, std=.02)
    
    
    def forward(self, wsi, omics, omics_masks=None):
        # ensure proper order
        omics_tokens = OrderedDict((modality, self.omics_tokenizers[modality](omics[modality])) for modality in self.omics_modalities)
        
        wsi_tokens = self.wsi_tokenizer(wsi) # [bs, n_proto, hidden_size]
        total_omics_token = sum([value.shape[1] for value in omics_tokens.values()]) # If you use default e.g. with 50 groups for RNA, 51 for DNAm and 45 for CNV then -> 146
        number_of_tokens_to_keep = int(total_omics_token * (1 - self.masking_ratio)) # number of tokens to keep for each modality
        if omics_masks is None:
            omics_masks = masking_func(omics_tokens, number_of_tokens_to_keep)
            # add wsi mask through a list of zeros, because we keep all WSI tokens and ensure proper order
            wsi_omics_masks = [torch.zeros(wsi_tokens.shape[0], wsi_tokens.shape[1], device=wsi_tokens.device)] + [omics_masks[modality] for modality in self.omics_modalities]
            mask_all = torch.cat(wsi_omics_masks, dim=1)
            # similar to MAE, make the full modal masking (in priority element set to 1)
            ids_shuffle = torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            # keep the number of tokens to keep for each modality + the number of WSI tokens
            ids_keep = ids_shuffle[:, :number_of_tokens_to_keep+wsi_tokens.shape[1]]
        
        else:
            # this is if you want to give a custom mask (i.e to check reconstructions quickly)
            # ensure correctly ordered
            omics_masks = OrderedDict((modality, omics_masks[modality]) for modality in self.omics_modalities)
            # again add the wsi mask through a list of zeros, because we keep all WSI tokens and ensure proper order
            wsi_omic_masks = [torch.zeros(wsi_tokens.shape[0], wsi_tokens.shape[1], device=wsi_tokens.device)] +  [omics_masks[modality] for modality in self.omics_modalities]
            mask_all = torch.cat(wsi_omic_masks, dim=1)
            ids_shuffle = torch.argsort(mask_all, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            # For the IDs to keep, the only change is that we now retain only the unmasked tokens.
            # Note: if you're applying this operation batch-wise, ensure that each element in the batch
            # retains the same number of tokens to work
            ids_keep = ids_shuffle[:, :int((mask_all == 0).sum() / wsi_tokens.shape[0])]
        
        omics_info = generate_omics_info(omics_masks)
        # The following two lines concatenate the whole sequence, with WSI tokens and omics tokens
        input_tokens = [omics_tokens[modality] for modality in self.omics_modalities]
        input_tokens = torch.cat([wsi_tokens] + input_tokens, dim=1) # shape [bs, n_proto + n_omics, hidden_size]
        input_tokens = torch.gather(input_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, input_tokens.shape[-1])) # keep only unmasked tokens
        global_tokens = self.global_token.expand(input_tokens.shape[0], -1, -1) 
        input_tokens = torch.cat([global_tokens, input_tokens], dim=1) # add the global token to the sequence
        
        # go through transformer blocks
        for blk in self.encoder_blocks:
            input_tokens = blk(input_tokens)
        encoded_tokens = self.norm(input_tokens)
        
        # then use the decoder, note that there is a decoder per modality and the outputs will be outputs[modality] = {pathway1: prediction, pathway2: prediction....}
        outputs = {modality: self.omics_decoders[modality](encoded_tokens, ids_restore, omics_info, omics_start=wsi_tokens.shape[1]) for modality in self.omics_modalities}
        
        return outputs, omics_masks
    
    def forward_encoder(self, wsi, omics):
        omics_tokens = OrderedDict((modality, self.omics_tokenizers[modality](omics[modality])) for modality in self.omics_modalities if modality in omics.keys())
        wsi_tokens = self.wsi_tokenizer(wsi)
        
        input_tokens = [omics_tokens[modality] for modality in self.omics_modalities if modality in omics.keys()]
        input_tokens = torch.cat([wsi_tokens] + input_tokens, dim=1) # shape [bs, n_proto + n_omics, hidden_size]
        global_tokens = self.global_token.expand(input_tokens.shape[0], -1, -1) 
        input_tokens = torch.cat([global_tokens, input_tokens], dim=1) # add the global token to the sequence
        
        # go through transformer blocks
        for blk in self.encoder_blocks:
            input_tokens = blk(input_tokens)
        encoded_tokens = self.norm(input_tokens)
        return encoded_tokens
    
        
            
class MorpheusABMIL(nn.Module):
    """
    In this variant of Morpheus, there is no WSI proto or global token, patches are passed through an ABMIL module
    The resulting tensor acts as both the WSI representation and the global token for the transformer.
    """
    def __init__(self, wsi_abmil, omics_tokenizers, omics_decoders, omics_modalities, hidden_size, masking_ratio=0.75, mlp_dim=256, num_heads=8, num_layers=2, dropout_rate=0.1):
        super(MorpheusABMIL, self).__init__()
        self.omics_modalities = omics_modalities
        self.wsi_abmil = wsi_abmil
        self.omics_tokenizers = nn.ModuleDict(omics_tokenizers)
        self.omics_decoders = nn.ModuleDict(omics_decoders)
        self.masking_ratio = masking_ratio
        self.norm = nn.LayerNorm(hidden_size)
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                qkv_bias=True,
            )
            for _ in range(num_layers)
        ])
            
    def forward(self, wsi, omics, omics_masks=None):
        # ensure proper order
        omics_tokens = OrderedDict((modality, self.omics_tokenizers[modality](omics[modality])) for modality in self.omics_modalities)
        wsi_token = self.wsi_abmil(wsi) # wsi_token is [bs, hidden_size] because of ABMIL
        total_omics_token = sum([value.shape[1] for value in omics_tokens.values()]) 
        number_of_tokens_to_keep = int(total_omics_token * (1 - self.masking_ratio))
        if omics_masks is None:
            omics_masks = masking_func(omics_tokens, number_of_tokens_to_keep)
            # here no WSI we stay only with omics masks
            list_omics_masks = [omics_masks[modality] for modality in self.omics_modalities]
            mask_all = torch.cat(list_omics_masks, dim=1)
            ids_shuffle = torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :number_of_tokens_to_keep]
        
        else:
            # ensure correctly ordered
            omics_masks = OrderedDict((modality, omics_masks[modality]) for modality in self.omics_modalities)
            list_omics_masks = [omics_masks[modality] for modality in self.omics_modalities]
            mask_all = torch.cat(list_omics_masks, dim=1)
            ids_shuffle = torch.argsort(mask_all, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :int((mask_all == 0).sum() / wsi_token.shape[0])] # for a given batch, masks must be the same as in MorpheusProto
            
        omics_info = generate_omics_info(omics_masks)
        # the input sequences is created with omics tokens only and then same as MAE
        input_tokens = [omics_tokens[modality] for modality in self.omics_modalities] 
        input_tokens = torch.cat(input_tokens, dim=1)
        input_tokens = torch.gather(input_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, input_tokens.shape[-1]))
        # here the wsi token is added to the sequence, similar to the <global> token in MorpheusProto (or in ViT)
        input_tokens = torch.cat([wsi_token.unsqueeze(1), input_tokens], dim=1)
        
        for blk in self.encoder_blocks:
            input_tokens = blk(input_tokens)
        encoded_tokens = self.norm(input_tokens)
        
        # here same as MorpheusProto, but we use omics_start=0 because there is no WSI tokens, the only one act as the global one
        outputs = {modality: self.omics_decoders[modality](encoded_tokens, ids_restore, omics_info, omics_start=0) for modality in self.omics_modalities}
        
        return outputs, omics_masks
    
    def forward_encoder(self, wsi, omics):
        # ensure proper order
        omics_tokens = OrderedDict((modality, self.omics_tokenizers[modality](omics[modality])) for modality in self.omics_modalities if modality in omics.keys())
        wsi_token = self.wsi_abmil(wsi)
        
        input_tokens = [omics_tokens[modality] for modality in self.omics_modalities if modality in omics.keys()]
        input_tokens = torch.cat(input_tokens, dim=1)
        input_tokens = torch.cat([wsi_token.unsqueeze(1), input_tokens], dim=1)
        
        for blk in self.encoder_blocks:
            input_tokens = blk(input_tokens)
        encoded_tokens = self.norm(input_tokens)
        return encoded_tokens


class MorpheusABMILEncoder(MorpheusABMIL, nn.Module):
    def __init__(self, n_outputs, wsi_abmil, omics_tokenizers, omics_modalities, hidden_size, mlp_dim=256, num_heads=8, num_layers=2, dropout_rate=0.1):
        nn.Module.__init__(self)
        self.omics_modalities = omics_modalities
        self.wsi_abmil = wsi_abmil
        self.omics_tokenizers = nn.ModuleDict(omics_tokenizers)
        self.norm = nn.LayerNorm(hidden_size)
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                qkv_bias=True,
            )
            for _ in range(num_layers)
        ])
        self.final_layer = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(hidden_size, n_outputs))
        
    def forward(self, wsi, omics={}):
        encoded_tokens = self.forward_encoder(wsi, omics)
        return self.final_layer(encoded_tokens[:, 0, :])
    
class MorpheusProtoEncoder(MorpheusProto, nn.Module):
    def __init__(self, n_outputs, wsi_tokenizer, omics_tokenizers={}, omics_modalities=[], hidden_size=256, mlp_dim=256, num_heads=8, num_layers=2, dropout_rate=0.1):
        nn.Module.__init__(self)
        self.omics_modalities = omics_modalities
        self.wsi_tokenizer = wsi_tokenizer
        if len(omics_modalities) != 0:
            self.omics_tokenizers = nn.ModuleDict(omics_tokenizers)
        self.norm = nn.LayerNorm(hidden_size)
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                qkv_bias=True,
            )
            for _ in range(num_layers)
        ])
        
        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        trunc_normal_(self.global_token, std=.02)
        self.final_layer = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(hidden_size, n_outputs))
        
    def forward(self, wsi, omics={}, return_embedding=None):
        encoded_tokens = self.forward_encoder(wsi, omics)
        if return_embedding:
            return encoded_tokens[:, 0, :]
        return self.final_layer(encoded_tokens[:, 0, :])
    