
import torch.nn as nn
import torch
from .utils.ml_utils import TransformerBlock
from monai.networks.layers import trunc_normal_


class MorpheusDecoder(nn.Module):
    def __init__(self, reverse_tokenizer, modality, hidden_size: int = 256, num_layers: int = 2, mlp_dim: int = 256, dropout_rate: float = 0.0, num_heads: int = 8) -> None:
        super().__init__()
        self.modality = modality
        self.hidden_size = hidden_size
        self.reverse_tokenizer = reverse_tokenizer
        self.list_pathways = list(self.reverse_tokenizer.mapping_dict.keys())
        self.mask_tokens = nn.Parameter(torch.zeros(1, 1, hidden_size))
        trunc_normal_(self.mask_tokens, std=0.02)
        self.decoder_embed = nn.Linear(hidden_size, hidden_size)
        self.decoder_norm = nn.LayerNorm(hidden_size)
        
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    qkv_bias=True,
                    with_cross_attention=True
                )
                for _ in range(num_layers)
            ]
        )

        self.id_embeddings = nn.Parameter(torch.zeros(1, len(self.list_pathways), self.hidden_size))
        trunc_normal_(self.id_embeddings, std=0.02)
    
    # def get_query_context(self, encoded_tokens, ids_restore, ids_keep, omics_info, omics_start):
    #     encoded_tokens_without_cls = encoded_tokens[:, 1:, :]

    #     mask_tokens = self.mask_tokens.repeat(encoded_tokens.shape[0], ids_restore.shape[1] + 1 - encoded_tokens.shape[1], 1).to(encoded_tokens.device)
    #     x_ = torch.cat([encoded_tokens_without_cls, mask_tokens], dim=1)
    #     x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))
    #     modalities_embedding = []
    #     for modality in self.modalities:
    #         if modality == 'wsi':
    #             modality_embedding = self.modalities_embeddings['wsi'].repeat(encoded_tokens.shape[0], omics_start, 1)
    #         else:
    #             modality_embedding = self.modalities_embeddings[modality].repeat(encoded_tokens.shape[0], omics_info[modality]['n_t'], 1)
            
    #         modalities_embedding.append(modality_embedding)

    #     modalities_embedding = torch.cat(modalities_embedding, dim=1)
    #     x_ = x_ + modalities_embedding
    #     query = x_[:, omics_start + omics_info[self.modality]['start']: omics_start + omics_info[self.modality]['end'], :]
    #     query = query + self.id_embeddings.repeat(encoded_tokens.shape[0], 1, 1)
    #     assert query.shape[1] == omics_info[self.modality]['n_t']

    #     context = torch.gather(x_, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, encoded_tokens.shape[2]))
    #     context = torch.cat([encoded_tokens[:, :1, :], context], dim=1)  # add cls token

    #     return query, context

    def get_query(self, encoded_tokens, ids_restore, omics_info, omics_start):

        mask_tokens = self.mask_tokens.repeat(encoded_tokens.shape[0], ids_restore.shape[1] + 1 - encoded_tokens.shape[1], 1)
        # cat with mask tokens but without the first token which is either the global tokens or the WSI token
        x = torch.cat([encoded_tokens[:, 1:, :], mask_tokens], dim=1)
        # unshuffle
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        # keep only tokens relevant to the modality
        x = x[:, omics_start + omics_info[self.modality]['start']: omics_start + omics_info[self.modality]['end'], :]
        x = x + self.id_embeddings.repeat(encoded_tokens.shape[0], 1, 1)
        assert x.shape[1] == omics_info[self.modality]['n_t']
        
        return x
        
    def forward(self, encoded_tokens, ids_restore, omics_info, omics_start):
        encoded_tokens = self.decoder_embed(encoded_tokens)
        x = self.get_query(encoded_tokens, ids_restore, omics_info, omics_start)

        for blk in self.decoder_blocks:
            x = blk(x, encoded_tokens)
        x = self.decoder_norm(x)
        x = {k: x[:, i, :] for i, k in enumerate(self.list_pathways)}
        x = self.reverse_tokenizer(x)

        return x

    def decode(self, encoded_tokens):
        encoded_tokens = self.decoder_embed(encoded_tokens)
       
        x = self.mask_tokens.repeat(encoded_tokens.shape[0], len(self.reverse_tokenizer.mapping_dict), 1)
        x = x + self.id_embeddings.repeat(encoded_tokens.shape[0], 1, 1)
        
        for blk in self.decoder_blocks:
            x = blk(x, encoded_tokens)  
        x = self.decoder_norm(x)
        
        x = {k: x[:, i, :] for i, k in enumerate(self.list_pathways)}
        x = self.reverse_tokenizer(x)
        
        return x