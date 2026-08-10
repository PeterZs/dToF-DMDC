import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
# print(XFORMERS_ENABLED)
# from xformers.ops import memory_efficient_attention, unbind
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind, fmha

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        from torch import unbind as unbind
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Attention)")


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv_bias=qkv_bias

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        # self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def _init_weight_identity(self):
        for layer in [self.q, self.k, self.v]:
            nn.init.eye_(layer.weight)    
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.eye_(self.proj.weight)
        nn.init.ones_(self.proj.bias)


    def forward(self, x: Tensor, y:Tensor, attn_bias=None) -> Tensor:
        assert torch.__version__ >= '2.0', "SDPA requires PyTorch 2.0 or later"
        B, N, C = x.shape   # img tokens, Q
        q = self.q(x)
        k = self.k(y)       # depth token, K, V
        v = self.v(y)

        q = q.reshape(B, N, self.num_heads, C //self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C //self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C //self.num_heads).permute(0, 2, 1, 3)

        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=self.attn_drop if self.training else 0)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C) 

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttention(CrossAttention):
    def forward(self, x: Tensor, y: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, y)

        B, N, C = x.shape
        q = self.q(x)
        k = self.k(y)       # depth token, K, V
        v = self.v(y)
        q = q.reshape(B, N, self.num_heads, C //self.num_heads)
        k = k.reshape(B, N, self.num_heads, C //self.num_heads)
        v = v.reshape(B, N, self.num_heads, C //self.num_heads)
        
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x