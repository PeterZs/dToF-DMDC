import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from block import *
from attentions import Attention, MemEffAttention, CrossAttention, MemEffCrossAttention
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


class CrossBlock(Block):
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
        cross_attn_class: Callable[..., nn.Module] = MemEffCrossAttention,
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
        
        # Cross attention parameters
        self.cross_norm_q = norm_layer(dim)
        self.cross_norm_kv = norm_layer(dim)
        self.cross_attn = cross_attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.cross_ls = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.cross_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    @torch.no_grad()
    def _init_weights(self, identity_init=True):
        self.depth_norm1.load_state_dict(self.norm1.state_dict())
        self.depth_attn.load_state_dict(self.attn.state_dict())
        self.depth_ls1.load_state_dict(self.ls1.state_dict())
        self.depth_drop_path1.load_state_dict(self.drop_path1.state_dict())
        self.depth_norm2.load_state_dict(self.norm2.state_dict())
        self.depth_mlp.load_state_dict(self.mlp.state_dict())
        self.depth_ls2.load_state_dict(self.ls2.state_dict())
        self.depth_drop_path2.load_state_dict(self.drop_path2.state_dict())
        self.cross_drop_path.load_state_dict(self.drop_path1.state_dict())
        if identity_init:
            if isinstance(self.cross_norm_q, nn.LayerNorm):
                nn.init.ones_(self.cross_norm_q.weight)
                if self.cross_norm_q.bias is not None:
                    nn.init.zeros_(self.cross_norm_q.bias)
            if isinstance(self.cross_norm_kv, nn.LayerNorm):
                nn.init.ones_(self.cross_norm_kv.weight)
                if self.cross_norm_kv.bias is not None:
                    nn.init.zeros_(self.cross_norm_kv.bias)
            self.cross_attn._init_weight_identity()
            nn.init.ones_(self.cross_ls.gamma)
            
        else:
            self.cross_norm_q.load_state_dict(self.norm1.state_dict())
            self.cross_norm_kv.load_state_dict(self.norm1.state_dict())
            self.cross_ls.load_state_dict(self.ls1.state_dict())
            self.cross_drop_path.load_state_dict(self.drop_path1.state_dict())
            # Copy attention weight
            self.cross_attn.q.weight.copy_(self.attn.qkv.weight[:self.dim])
            self.cross_attn.k.weight.copy_(self.attn.qkv.weight[self.dim:self.dim*2])
            self.cross_attn.v.weight.copy_(self.attn.qkv.weight[self.dim*2:])
            if self.attn.qkv.bias is not None:
                self.cross_attn.q.bias.copy_(self.attn.qkv.bias[:self.dim])
                self.cross_attn.k.bias.copy_(self.attn.qkv.bias[self.dim:self.dim*2])
                self.cross_attn.v.bias.copy_(self.attn.qkv.bias[self.dim*2:])
            self.cross_attn.proj.load_state_dict(self.attn.proj.state_dict())
            self.cross_attn.proj_drop.load_state_dict(self.attn.proj_drop.state_dict())

    def forward(self, x: Tensor, y:Tensor) -> Tuple[Tensor]:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        def depth_attn_residual_func(x: Tensor) -> Tensor:
            return self.depth_ls1(self.depth_attn(self.depth_norm1(x)))

        def depth_ffn_residual_func(x: Tensor) -> Tensor:
            return self.depth_ls2(self.depth_mlp(self.depth_norm2(x)))

        def cross_attn_residual_func(xy: Tensor) -> Tensor:
            x, y = unbind(xy, 0)
            x = self.cross_norm_q(x)
            y = self.cross_norm_kv(y)
            return self.cross_ls(self.cross_attn(x, y))
        
        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            y = drop_add_residual_stochastic_depth(
                y,
                residual_func=depth_attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio
            )
            x = drop_add_residual_stochastic_depth(
                torch.stack([x, y], 0),
                residual_func=cross_attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio
            )
            y = drop_add_residual_stochastic_depth(
                y,
                residual_func=depth_ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )

        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))           # im layer 1
            y = y + self.depth_drop_path1(depth_attn_residual_func(y))     # depth layer 1
            x = x + self.cross_drop_path(cross_attn_residual_func(torch.stack([x, y], 0)))  # cross block
            x = x + self.drop_path2(ffn_residual_func(x))            # im layer 2
            y = y + self.depth_drop_path2(depth_ffn_residual_func(y))     # dpeth layer 2
        else:
            x = x + attn_residual_func(x)           # im layer 1
            y = y + depth_attn_residual_func(y)     # depth layer 1
            x = x + cross_attn_residual_func(torch.stack([x, y], 0))  # cross block
            x = x + ffn_residual_func(x)            # im layer 2
            y = y + depth_ffn_residual_func(y)     # dpeth layer 2
        
        return x, y



class NestedCrossBlock(CrossBlock):
    def forward_nested(self, x_list: List[Tensor], y_list: List[Tensor]):
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)
        assert isinstance(self.cross_attn, MemEffCrossAttention)

        if self.training and self.sample_drop_ratio > 0.0:
        
            def attn_residual_func(x: Tensor) -> Tensor:
                return self.attn(self.norm1(x))

            def ffn_residual_func(x: Tensor) -> Tensor:
                return self.mlp(self.norm2(x))

            def depth_attn_residual_func(x: Tensor) -> Tensor:
                return self.depth_ls1(self.depth_attn(self.depth_norm(x)))

            def depth_ffn_residual_func(x: Tensor) -> Tensor:
                return self.depth_ls2(self.depth_mlp(self.depth_norm2(x)))

            def cross_attn_residual_func(xy: Tensor) -> Tensor:
                x, y = unbind(xy, 0)
                x = self.cross_norm_q(x)
                y = self.cross_norm_kv(y)
                return self.cross_ls(self.cross_attn(x, y))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            y_list = drop_add_residual_stochastic_depth_list(
                y_list,
                residual_func=depth_attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.depth_ls1.gamma if isinstance(self.depth_ls1, LayerScale) else None,
            )

            xy_list = [torch.stack([x, y], 0) for x, y in zip(x_list, y_list)]
            xy_list = drop_add_residual_stochastic_depth_list(
                xy_list,
                residual_func=cross_attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.cross_ls.gamma if isinstance(self.cross_ls, LayerScale) else None,
            )
            x_list = []
            y_list = []
            for xy in xy_list:
                x, y = unbind(xy, 0)
                x_list.append(x)
                y_list.append(y)
                
            x_list = drop_add_residual_stochastic_depth_list(
                x_list, 
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls2, LayerScale) else None,
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
                return self.depth_ls1(self.depth_attn(self.depth_norm(x), attn_bias=attn_bias))

            def depth_ffn_residual_func(x: Tensor) -> Tensor:
                return self.depth_ls2(self.depth_mlp(self.depth_norm2(x)))

            def cross_attn_residual_func(xy:Tensor, attn_bias=None) -> Tensor:
                x, y = unbind(xy, 0)
                x = self.cross_norm_q(x)
                y = self.cross_norm_kv(y)
                return self.cross_ls(self.cross_attn(x, y, attn_bias=attn_bias))

            attn_bias1, x = get_attn_bias_and_cat(x_list)
            attn_bias2, y = get_attn_bias_and_cat(y_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias1)
            y = y + attn_residual_func(y, attn_bias=attn_bias2)
            x_list, y_list = attn_bias1.split(x), attn_bias2.split(y)

            attn_bias3, x, y = get_cross_attn_bias_and_cat(x_list, y_list)
            x = x + cross_attn_residual_func(torch.stack([x, y], 0), attn_bias=attn_bias3)
            x = x + ffn_residual_func(x)
            y = y + depth_ffn_residual_func(y)
            return attn_bias3.split(x), attn_bias3.split(y)

    def forward(self, x_or_x_list, y_or_y_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list, y_or_y_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list, y_or_y_list)
        else:
            raise AssertionError
