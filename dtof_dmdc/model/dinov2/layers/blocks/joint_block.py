import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

import sys
import os

from block import *
from attentions import JointAttention
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

class JointBlock(Block):
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
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_rmsnorm=True
    ) -> None:
        
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, proj_bias, ffn_bias, drop, attn_drop, init_values, drop_path, act_layer, norm_layer, attn_class, ffn_layer)

        # Re-declare joint attention module layer
        self.depth_norm1 = norm_layer(dim)
        # Joint Attention with CrossAttention
        self.attn = JointAttention(
            [dim, dim], # if fuse_by_add else dim*2,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_rmsnorm=qk_rmsnorm
        )
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

    def forward(self, x: Tensor, y:Tensor) -> Tuple[Tensor]:

        def joint_residual_func(x:Tensor, y:Tensor, attn_bias=None)-> Tensor:
            x = self.norm1(x)
            y = self.depth_norm1(y)
            x, y = self.attn((x,y), attn_bias=attn_bias)
            x = self.ls1(x)
            y = self.depth_ls1(y)
            return x, y

        def img_ffn_residual_func(x:Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        
        def depth_ffn_residual_func(y:Tensor) -> Tensor:
            return self.depth_ls2(self.depth_mlp(self.depth_norm2(y)))

        if self.training and self.sample_drop_ratio > 0.1:
            x, y = joint_drop_add_residual_stochastic_depth(
                x, y,
                residual_func=joint_residual_func,
                sample_drop_ratio=self.sample_drop_ratio
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=img_ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio    
            )
            y = drop_add_residual_stochastic_depth(
                y,
                residual_func=depth_ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio    
            )

        elif self.training and self.sample_drop_ratio > 0.0:
            x_, y_ = joint_residual_func(x, y)
            x = x + self.drop_path1(x_)
            y = y + self.depth_drop_path1(y_)
            x = x + self.drop_path2(img_ffn_residual_func(x))
            y = y + self.depth_drop_path2(depth_ffn_residual_func(y))

        else:
            x_, y_ = joint_residual_func(x, y)
            x, y = x + x_, y + y_
            x = x + img_ffn_residual_func(x)
            y = y + depth_ffn_residual_func(y)
            
        return x, y
