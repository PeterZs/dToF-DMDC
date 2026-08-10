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
# print(XFORMERS_ENABLED)
# from xformers.ops import memory_efficient_attention, unbind
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind, fmha

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads=8, axis=-1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x, dim=-1):
        return F.normalize(x, dim=dim) * self.gamma * self.scale


class MaskedJointAttention(nn.Module):
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
        num_inputs = 2
        self.attn_drop = attn_drop
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj1 = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.proj_drop2 = nn.Dropout(proj_drop)

    def _init_weights(self):
        self.qkv1.load_state_dict(self.qkv.state_dict())
        self.proj1.load_state_dict(self.proj.state_dict())
        # self.proj_drop2.load_state_dict(self.proj_drop2.state_dict())
        
    # def forward(self, x:Tensor, y:Tensor, attn_bias:Tensor=None):
    def forward(self, inputs: Tuple[Tensor], attn_bias: Tensor=None):
        "input is (image_tokens, depth_tokens)"
        assert len(inputs) == 2, "inputs should be a tuple of two tensors"
        assert torch.__version__ >= '2.0', "SDPA requires PyTorch 2.0 or later"
        
        fcs = [self.qkv, self.qkv1]
        projs = [self.proj, self.proj1]

        all_qkvs = []
        all_seqlen = []
        # inputs = [x, y]
        for x, to_qkv in zip(inputs, fcs):
            B, N, C = x.shape
            qkv = to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)   # (3, B, H, N, C // H)
            
            all_qkvs.append(qkv)
            all_seqlen.append(N)

        # (3, B, H, N, C // H)
        all_qkvs = torch.cat(all_qkvs, dim=-2)
        q, k, v = torch.unbind(all_qkvs, 0)

        N_image, N_depth = all_seqlen

        if attn_bias is None: # if not designated,
            L, S = q.size(-2), k.size(-2)
            attn_bias = torch.zeros(L, S, dtype=torch.float32, device=x.device)
            attn_bias[N_image:, :N_image] = float("-inf")
            
        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=self.attn_drop if self.training else 0)
        x = x.permute(0, 2, 1, 3).reshape(B, -1, C)

        outs = x.split(all_seqlen, dim=-2)

        all_outs = []
        for x, proj in zip(outs, projs):
            out = self.proj_drop(proj(x))
            all_outs.append(out)
        
        return tuple(all_outs)
    

        """
        qkv_x = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)            # (3, B, H, N, C // H)
        qkv_y = self.qkv2(y).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)            # (3, B, H, N, C // H)
        
        qkv_xy = torch.cat([qkv_x, qkv_y], dim=-2)  # (3, B, H, 2N, C // H)
        q, k, v = unbind(qkv_xy, 0)

        if attn_bias is None:
            L, S = q.size(-2), k.size(-2)
            attn_bias = torch.zeros(L, S, dtype=torch.float32, device=x.device)
            attn_bias[N:, :N] = float("-inf")

        xy = F.scaled_dot_product_attention(q, k, v, attn_bias) #, dropout_p=self.attn_drop) # (B, H, 2N, C//H)
        
        x = xy[..., :N, :].permute(0, 2, 1, 3).reshape(B, N, C)
        y = xy[..., N:, :].permute(0, 2, 1, 3).reshape(B, N, C)
        
        x = self.proj_drop(self.proj(x))
        y = self.proj_drop2(self.proj2(y))

        return x, y
        """

class MemEffMaskedJointAttention(MaskedJointAttention):
    # def forward(self, x:Tensor, y:Tensor, attn_bias:Tensor=None):
    def forward(self, inputs: Tuple[Tensor], attn_bias: Tensor=None):
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(inputs, attn_bias)
        
        fcs = [self.qkv, self.qkv1]
        projs = [self.proj, self.proj1]

        all_qkvs = []
        all_seqlen = []
        # inputs = [x, y]
        for token, to_qkv, q_rmsnorm, k_rmsnorm in zip(inputs, fcs, self.q_rmsnorms, self.k_rmsnorms):
            B, N, C = x.shape
            qkv = to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

            if self.qk_rmsnorm:
                q, k, v = unbind(qkv, 2)
                q = q_rmsnorm(q, dim=1)
                k = k_rmsnorm(k, dim=1)
                qkv = torch.stack([q, k, v], dim=2)
            
            all_qkvs.append(qkv)
            all_seqlen.append(N)
        
        # qkvs = [B, N, 3, H, C//H]
        all_qkvs = torch.cat(all_qkvs, dim=1)
        q, k, v = unbind(all_qkvs, 2)

        if attn_bias is None:
            L, S = q.size(1), k.size(1)
            attn_bias = torch.zeros(1, L, S, dtype=torch.float32, device=x.device)
            attn_bias[:, N:, :N] = float("-inf")

        x = memory_efficient_attention(q, k, v, attn_bias) # (B, 2N, H, C//H)        
        x.reshape(B, -1, C)
        outs = x.split(all_seqlen, dim=1)
        
        all_outs = []
        for x, proj in zip(outs, projs):
            out = self.proj_drop(proj(x))
            all_outs.append(out)
        
        return tuple(all_outs)
    
        """
        B, N, C = x.shape
        qkv_x = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads)
        qkv_y = self.qkv2(y).reshape(
            B, N, 3, self.num_heads, C // self.num_heads)
        
        qkv_xy = torch.cat([qkv_x, qkv_y], dim=1)   # (B, 2N, 3, H, C//H)
        q, k, v = unbind(qkv_xy, 2)                 # (B, 2N, H, C//H)
        
        if attn_bias is None:
            L, S = q.size(1), k.size(1) # N, N
            attn_bias = torch.zeros(L, S, dtype=torch.float32, device=x.device)
            attn_bias[N:, :N] = float("-inf")
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, 2*N, 2*N)
        else:
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, 2*N, 2*N)
                    
        xy = memory_efficient_attention(q, k, v, attn_bias) # (B, 2N, H, C//H)
        x = xy[:, :N, ...].reshape(B, N, C)
        y = xy[:, N:, ...].reshape(B, N, C)
        
        x = self.proj_drop(self.proj(x))
        y = self.proj_drop2(self.proj2(y))

        return x, y"""