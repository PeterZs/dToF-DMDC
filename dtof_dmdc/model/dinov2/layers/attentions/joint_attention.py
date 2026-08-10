import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import *


logger = logging.getLogger("dinov2")

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
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


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale


class JointAttention(nn.Module):
    def __init__(
            self,
            dims: List[int],
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            qk_rmsnorm=False
        ):
        super().__init__()
        self.dims = dims
        self.num_inputs = len(dims)
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        
        # self.scale = head_dim**-0.5

        for in_idx, dim in enumerate(dims):
            qkv_name = "qkv" if in_idx == 0 else f"qkv{in_idx}"
            setattr(self, qkv_name, nn.Linear(dim, dim * 3, bias = qkv_bias))
            proj_name = "proj" if in_idx == 0 else f"proj{in_idx}"
            setattr(self, proj_name, nn.Linear(dim, dim, bias = False))
        
        self.proj_drop = nn.Dropout(proj_drop)
        head_dims = [dim // num_heads for dim in dims]
        self.qk_rmsnorm = qk_rmsnorm
        if self.qk_rmsnorm:
            self.q_rmsnorms = nn.ModuleList([MultiHeadRMSNorm(head_dim, heads = num_heads) for head_dim in head_dims])
            self.k_rmsnorms = nn.ModuleList([MultiHeadRMSNorm(head_dim, heads = num_heads) for head_dim in head_dims])
        else:
            self.q_rmsnorms = (None,) * num_inputs
            self.k_rmsnorms = (None,) * num_inputs
    
    def _init_weights(self):
        qkv_state_dict = self.qkv.state_dict()
        proj_state_dict = self.proj.state_dict()
        for i in range(1, self.num_inputs):
            qkv = getattr(self, f"qkv{i}")
            proj = getattr(self, f"proj{i}")
            qkv.load_state_dict(qkv_state_dict)
            proj.load_state_dict(proj_state_dict)
    
    def forward(self,
                inputs: Tuple[Tensor],
                attn_bias: Tensor = None
                # attn_bias: Tuple[Tensor | None] | None = None
                ):
        assert torch.__version__ >= '2.0', "SDPA requires PyTorch 2.0 or later"
        assert self.num_inputs == len(inputs)

        # attn_bias = (None, ) * self.num_inputs if attn_bias is None else attn_bias

        fcs = []
        projs = []
        for idx in range(self.num_inputs):
            fc_name = "qkv" if idx == 0 else f"qkv{idx}"
            proj_name = "proj" if idx == 0 else f"proj{idx}"
            fcs.append(getattr(self, fc_name))
            projs.append(getattr(self, proj_name))
        
        all_qkvs = []
        all_seqlen = []
        for x, to_qkv, q_rmsnorm, k_rmsnorm in zip(inputs, fcs, self.q_rmsnorms, self.k_rmsnorms):
            B, N, C = x.shape
            qkv = to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, H, N, C // H)
            
            if self.qk_rmsnorm:
                q, k, v = unbind(qkv, 0)
                q = q_rmsnorm(q)
                k = k_rmsnorm(k)
                qkv = torch.stack([q, k, v])
            
            all_seqlen.append(N)
            all_qkvs.append(qkv)

        # (3, B, H, N, C // H)
        all_qkvs = torch.cat(all_qkvs, dim=-2)
        q, k, v = unbind(all_qkvs, 0)
        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=self.attn_drop if self.training else 0)
        x = x.permute(0, 2, 1, 3).reshape(B, -1, C)

        outs = x.split(all_seqlen, dim=-2)

        all_outs = []
        for x, proj in zip(outs, projs):
            out = self.proj_drop(proj(x))
            all_outs.append(out)
        return tuple(all_outs)
