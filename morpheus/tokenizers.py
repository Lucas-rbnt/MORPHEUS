from typing import Optional, Union, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.ml_utils import SNN_Block

class WSITokenizer(nn.Module):
    def __init__(self, embed_dim: int=512, num_proto: int=32, input_dim: int=1536) -> None:
        super(WSITokenizer, self).__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim**-0.5  # Scaling factor for dot-product attention
        self.num_proto = num_proto
        
        # Learnable query prototypes (shared)
        self.prototypes = nn.Parameter(
            torch.zeros(num_proto, embed_dim)
        )  # (num_proto, embed_dim)

        
        # Projection from input_dim to embed_dim for key/value
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)

        self.kv_norm = nn.LayerNorm(input_dim)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, kv: torch.tensor, key_padding_mask: Optional[torch.tensor] = None) -> Union[torch.tensor, torch.tensor]:
        """
        key_value: (bs, N, input_dim) -> Projected to (B, N, embed_dim)
        key_padding_mask: (bs, N) - Boolean mask where True indicates padding
        Returns: (bs, num_proto, embed_dim) attended output
        """

        bs = kv.shape[0] # Batch size, sequence length (≤ N), feature dim

        # Project key and value from input_dim to embed_dim
        kv = self.kv_norm(kv)
        K = self.k_proj(kv)  # (bs, seq_len, embed_dim)
        V = self.v_proj(kv)  # (bs, seq_len, embed_dim)

        # Expand learnable prototypes for each sample in the batch
        Q = self.prototypes.unsqueeze(0).expand(bs, -1, -1)  # (bs, num_proto, embed_dim)
        Q = self.q_norm(Q)
        # Compute scaled dot-product attention
        attn_weights = (
            torch.matmul(Q, K.transpose(1, 2)) * self.scale
        )  # (bs, num_protos, seq_len)

        # Apply padding mask 
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1), float("-inf")
            )  # (bs, num_proto, seq_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attended_output = torch.matmul(attn_weights, V)  # (bs, num_proto, embed_dim)

        output = self.output_proj(attended_output)  # (bs, num_proto, embed_dim)
        output = self.out_norm(output)

        return output


class OmicsTokenizer(nn.Module):
    """
    Class for different types of omics data tokenization
    """
    def __init__(self, mapping_dict: Dict[str, List], output_dim: int, hidden: List[int] = []):
        super(OmicsTokenizer, self).__init__()
        self.mapping_dict = mapping_dict
        layers_size = {k: [len(v)] + hidden + [output_dim] for k, v in mapping_dict.items()}
        self.token_mlp_dict = nn.ModuleDict(
            {token: nn.Sequential(*[SNN_Block(layers_size[token][i], layers_size[token][i+1]) for i in range(len(layers_size[token])-1)]) for token in mapping_dict.keys()}
        )
    
    def forward(self, x: Dict[str, torch.tensor]) -> torch.tensor:
        return torch.stack([self.token_mlp_dict[k](x[k]) for k in x.keys()], dim=1)


