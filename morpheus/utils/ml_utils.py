"""
Correcting MONAI implementation to define cross-attention layer only when necessary.
Also adding SNN Block
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


from monai.networks.blocks import MLPBlock, SABlock, CrossAttentionBlock


class TransformerBlock(nn.Module):
    """
    ### NOTE: This is a modified version of the MONAI implementation of the TransformerBlock. This allows not to define cross-attention layer when only self-attention is required.
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
        causal: bool = False,
        sequence_length: int | None = None,
        with_cross_attention: bool = False,
        use_flash_attention: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = True,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            mlp_dim (int): dimension of feedforward layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias(bool, optional): apply bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.
            use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
                (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
            include_fc: whether to include the final linear layer. Default to True.
            use_combined_linear: whether to use a single linear layer for qkv projection, default to True.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(
            hidden_size,
            num_heads,
            dropout_rate,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
            causal=causal,
            sequence_length=sequence_length,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.with_cross_attention = with_cross_attention
        if self.with_cross_attention:
            self.norm_cross_attn = nn.LayerNorm(hidden_size)
            self.cross_attn = CrossAttentionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                qkv_bias=qkv_bias,
                causal=False,
                use_flash_attention=use_flash_attention,
            )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        if self.with_cross_attention:
            x = x + self.cross_attn(self.norm_cross_attn(x), context=context)
        x = x + self.mlp(self.norm2(x))
        return x
    

def SNN_Block(dim1, dim2, dropout=0.):
    r"""
    Code adapted from https://github.com/mahmoodlab/MMP/blob/main/src/mil_models/components.py
    """
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))
