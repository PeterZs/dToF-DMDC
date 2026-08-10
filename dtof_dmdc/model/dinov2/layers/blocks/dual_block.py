import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from block import *
from attentions import Attention, MemEffAttention
from drop_path import DropPath
from layer_scale import LayerScale
from mlp import Mlp

logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Block)")
    else:
        # warnings.warn("xFormers is disabled (Block)")
        from torch import unbind as unbind
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Block)")


class DualStemBlock(Block):
    def __init__(self,
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
        attn_class: Callable[..., nn.Module] = MemEffAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp):

        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, proj_bias, ffn_bias, drop, attn_drop, init_values, drop_path, act_layer, norm_layer, attn_class, ffn_layer)
        
        # Depth backbone:
        self.depth_norm1 = norm_layer(dim)
        self.depth_attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
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
    
    @torch.no_grad()
    def _init_weights(self):
        self.depth_norm1.load_state_dict(self.norm1.state_dict())
        self.depth_attn.load_state_dict(self.attn.state_dict())
        self.depth_ls1.load_state_dict(self.ls1.state_dict())
        self.depth_norm2.load_state_dict(self.norm2.state_dict())
        self.depth_mlp.load_state_dict(self.mlp.state_dict())
        self.depth_ls2.load_state_dict(self.ls2.state_dict())

    def forward(self, x: Tensor, y:Tensor) -> Tuple[Tensor]:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        def depth_attn_residual_func(x: Tensor) -> Tensor:
            return self.depth_ls1(self.depth_attn(self.depth_norm1(x)))

        def depth_ffn_residual_func(x: Tensor) -> Tensor:
            return self.depth_ls2(self.depth_mlp(self.depth_norm2(x)))
        
        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio
            )
            if y is not None:
                y = drop_add_residual_stochastic_depth(
                    y,
                    residual_func=depth_attn_residual_func,
                    sample_drop_ratio=self.sample_drop_ratio
                )
                y = drop_add_residual_stochastic_depth(
                    y,
                    residual_func=depth_ffn_residual_func,
                    sample_drop_ratio=self.sample_drop_ratio
                )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))           # im layer 1
            x = x + self.drop_path2(ffn_residual_func(x))            # im layer 2
            y = y + self.depth_drop_path1(depth_attn_residual_func(y))     # depth layer 1
            y = y + self.depth_drop_path2(depth_ffn_residual_func(y))     # dpeth layer 2
        else:
            x = x + attn_residual_func(x)           # im layer 1
            x = x + ffn_residual_func(x)            # im layer 2
            y = y + depth_attn_residual_func(y)     # depth layer 1
            y = y + depth_ffn_residual_func(y)     # dpeth layer 2
        
        return x, y


class NestedDualStemBlock(DualStemBlock):
    def forward_nested(self, x_list: List[Tensor], y_list: List[Tensor]):
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:
            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor) -> Tensor:
                return self.mlp(self.norm2(x))
            
            def depth_attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.depth_attn(self.depth_norm1(x), attn_bias=attn_bias)

            def depth_ffn_residual_func(x: Tensor) -> Tensor:
                return self.depth_mlp(self.depth_norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls2, LayerScale) else None,
            )
            y_list = drop_add_residual_stochastic_depth_list(
                y_list,
                residual_func=depth_attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.depth_ls1.gamma if isinstance(self.depth_ls1, LayerScale) else None,
            )
            y_list = drop_add_residual_stochastic_depth_list(
                y_list,
                residual_func=depth_ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.depth_ls2.gamma if isinstance(self.depth_ls2, LayerScale) else None,
            )
            return x_list, y_list
        else:
            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            def depth_attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.depth_ls1(self.depth_attn(self.depth_norm1(x), attn_bias=attn_bias))

            def depth_ffn_residual_func(x: Tensor) -> Tensor:
                return self.depth_ls2(self.depth_mlp(self.depth_norm2(x)))

            attn_bias_x, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias_x)
            x = x + ffn_residual_func(x)

            attn_bias_y, y = get_attn_bias_and_cat(y_list)
            y = y + depth_attn_residual_func(y, attn_bias=attn_bias_y)
            y = y + depth_ffn_residual_func(y)

            return attn_bias_x.split(x), attn_bias_y.split(y)
        
    def forward(self, x_or_x_list, y_or_y_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list, y_or_y_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list, y_or_y_list)
        else:
            raise AssertionError
