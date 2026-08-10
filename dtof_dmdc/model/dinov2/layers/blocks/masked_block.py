import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

import sys
import os

from block import *
from attentions import MaskedJointAttention, MemEffMaskedJointAttention
from drop_path import DropPath
from layer_scale import LayerScale
from mlp import Mlp


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat, unbind

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Block)")
    else:
        # warnings.warn("xFormers is disabled (Block)")
        from torch import unbind as unbind
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Block)")


class MaskedJointBlock(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = MaskedJointAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_rmsnorm=True
    ) -> None:
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, proj_bias, ffn_bias, drop, attn_drop, init_values, drop_path, act_layer, norm_layer, attn_class, ffn_layer)

        # Redefine attention layer with masked cross attention
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Depth backbone:
        self.depth_norm1 = norm_layer(dim)
        self.depth_ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.depth_drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.depth_norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.depth_mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.depth_ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.depth_drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def _init_weights(self):
        self.attn._init_weights()
        self.depth_norm1.load_state_dict(self.norm1.state_dict())
        self.depth_ls1.load_state_dict(self.ls1.state_dict())
        self.depth_drop_path1.load_state_dict(self.drop_path1.state_dict())
        self.depth_norm2.load_state_dict(self.norm2.state_dict())
        self.depth_mlp.load_state_dict(self.mlp.state_dict())
        self.depth_ls2.load_state_dict(self.ls2.state_dict())
        self.depth_drop_path2.load_state_dict(self.drop_path2.state_dict())
        
    def forward(self, x:Tensor, y:Tensor)-> Tuple[Tensor]:
        def attn_residual_func(x: Tensor, y: Tensor) -> Tensor:
            x, y = self.attn((self.norm1(x), self.depth_norm1(y)))
            x = self.ls1(x)
            y = self.depth_ls1(y)
            return x, y

        def img_ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        
        def depth_ffn_residual_func(x: Tensor) -> Tensor:
            return self.depth_ls2(self.depth_mlp(self.depth_norm2(x)))
        
        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x, y = joint_drop_add_residual_stochastic_depth(
                x, y,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=img_ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            y = drop_add_residual_stochastic_depth(
                y,
                residual_func=depth_ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )

        elif self.training and self.sample_drop_ratio > 0.0:
            x_, y_ = attn_residual_func(x, y)
            x = x + self.drop_path1(x_)
            y = y + self.depth_drop_path1(y_)
            x = x + self.drop_path2(img_ffn_residual_func(x))
            y = y + self.depth_drop_path2(depth_ffn_residual_func(y))
        else:
            x_, y_ = attn_residual_func(x, y)
            x, y = x + x_, y + y_
            x = x + img_ffn_residual_func(x)
            y = y + depth_ffn_residual_func(y)

        return x, y

# Implement Later !!!! 
# class NestedMaskedJointBlock(MaskedJointBlock):
#     def forward_nested(self, x_list: List[Tensor], y_list: List[Tensor]):
#         assert isinstance(self.attn, MemEffMaskedJointAttention)

#         if self.training and self.sample_drop_ratio > 0.0:
#             def attn_residual_func(x: Tensor, y: Tensor, attn_bias=None) -> Tensor:
#                 x, y = self.attn(self.norm1(x), self.depth_norm1(y), attn_bias)
#                 # x = self.ls1(x)
#                 # y = self.depth_ls1(y)
#                 return x, y
#             def ffn_residual_func(x: Tensor) -> Tensor:
#                 return self.mlp(self.norm2(x))
#             def depth_ffn_residual_func(x: Tensor) -> Tensor:
#                 return self.depth_ls2(self.depth_mlp(self.depth_norm2(x)))

#             x_list, y_list = masked_drop_add_residual_stochastic_depth_list(
#                 x_list, y_list,
#                 residual_func=attn_residual_func,
#                 sample_drop_ratio=self.sample_drop_ratio,
#                 x_scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
#                 y_scaling_vector=self.depth_ls1.gamma if isinstance(self.depth_ls1, LayerScale) else None,
#             )
#             x_list = drop_add_residual_stochastic_depth_list(
#                 x_list,
#                 residual_func=img_ffn_residual_func,
#                 sample_drop_ratio=self.sample_drop_ratio,
#                 scaling_vector=self.ls2.gamma if isinstance(self.ls2, LayerScale) else None,
#             )
#             y_list = drop_add_residual_stochastic_depth_list(
#                 y_list,
#                 residual_func=depth_ffn_residual_func,
#                 sample_drop_ratio=self.sample_drop_ratio,
#                 scaling_vector=self.depth_ls2.gamma if isinstance(self.depth_ls2, LayerScale) else None,
#             )
#             return x_list, y_list
            
#         else:
#             def attn_residual_func(x: Tensor, y: Tensor, attn_bias=None) -> Tensor:
#                 x, y = self.attn(self.norm1(x), self.depth_norm1(y), attn_bias)
#                 x = self.ls1(x)
#                 y = self.depth_ls1(y)
#                 return x, y

#             def ffn_residual_func(x: Tensor) -> Tensor:
#                 return self.ls2(self.mlp(self.norm2(x)))
            
#             def depth_ffn_residual_func(x: Tensor) -> Tensor:
#                 return self.depth_ls2(self.depth_mlp(self.depth_norm2(x)))

#             attn_bias, x = get_masked_attn_bias_and_cat(x_list)
#             attn_bias, y = get_masked_attn_bias_and_cat(y_list)
#             x_residual, y_residual = attn_residual_func(x, y)

#     def forward(self, x_or_x_list, y_or_y_list):
#         if isinstance(x_or_x_list, Tensor):
#             return super().forward(x_or_x_list, y_or_y_list)
#         elif isinstance(x_or_x_list, list):
#             if not XFORMERS_AVAILABLE:
#                 raise AssertionError("xFormers is required for using nested tensors")
#             return self.forward_nested(x_or_x_list, y_or_y_list)
#         else:
#             raise AssertionError
# ``

# def get_masked_attn_bias_and_cat(x_list, branges=None):
#     """
#     this will perform the index select, cat the tensors, and provide the attn_bias from cache
#     """
#     batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
#     all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
#     if all_shapes not in attn_bias_cache.keys():
#         seqlens = []
#         for b, x in zip(batch_sizes, x_list):
#             for _ in range(b):
#                 seqlens.append(x.shape[1])

#         torch.set_printoptions(profile='full')
        
#         L = sum(seqlens)
#         nums1 = torch.tensor([0] + seqlens[:-1]) * 2
#         cumsum1 = torch.cumsum(nums1, dim=0)
#         nums2 = torch.tensor(seqlens) * 2
#         cumsum2 = torch.cumsum(nums2, dim=0)
#         block_sizes = cumsum2 - cumsum1

#         attn_bias = torch.full((L*2, L*2), fill_value=float("-inf"), device=x[0].device)
#         for i, j in zip(cumsum1, block_sizes):
#             attn_bias[i:i+j//2, i:i+j] = 0
#             attn_bias[i+j//2:i+j, i+j//2:i+j] = 0
#         # attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
#         # attn_bias._batch_sizes = batch_sizes

#         attn_bias_cache[all_shapes] = attn_bias

#     if branges is not None:
#         cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
#     else:
#         tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
#         cat_tensors = torch.cat(tensors_bs1, dim=1)

#     return attn_bias_cache[all_shapes], cat_tensors


# def masked_drop_add_residual_stochastic_depth_list(
#     x_list: List[Tensor],
#     y_list: List[Tensor],
#     residual_func: Callable[[Tensor, Any], Tensor],
#     sample_drop_ratio: float = 0.0,
#     x_scaling_vector=None,
#     y_scaling_vector=None
# ) -> Tensor:
#     # 1) generate random set of indices for dropping samples in the batch
#     branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
#     branges = [s[0] for s in branges_scales]
#     residual_scale_factors = [s[1] for s in branges_scales]

#     # 2) get attention bias and index+concat the tensors
#     attn_bias, x_cat = get_masked_attn_bias_and_cat(x_list, branges)
#     attn_bias, y_cat = get_masked_attn_bias_and_cat(y_list, branges)

#     # 3) apply residual_func to get residual, and split the result
#     residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore
    
#     outputs = []
#     for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
#         outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
#     return outputs
