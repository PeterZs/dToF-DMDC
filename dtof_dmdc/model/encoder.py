from typing import *
from numbers import Number
import importlib
import itertools
import functools
import sys

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


from .modules import ResidualConvBlock

from .dinov2.models.vision_transformer import DinoVisionTransformer
from .utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing

import math
from itertools import chain

import utils3d
##################################################
# Encoder with dual bakcbone, equiv. architecture
##################################################

class MultiModalEncoder(nn.Module):
    backbone: DinoVisionTransformer
    image_mean: torch.Tensor
    image_std: torch.Tensor
    dim_features: int

    def __init__(self, backbone: str, intermediate_layers: Union[int, List[int]], dim_out: int, block_fn: str, fuse_block_fn: str, fuse_block_layers: Union[int, List[int]], interp_mode: str, depth_fill: bool = True, **deprecated_kwargs):
        super(MultiModalEncoder, self).__init__()
        self.intermediate_layers = intermediate_layers

        # Load the backbone with extended multi-modal block
        
        self.hub_loader = getattr(importlib.import_module(".dinov2.hub.backbones", __package__), backbone)
        self.backbone_name = backbone

        BaseBlock = getattr(importlib.import_module(".dinov2.layers.blocks", __package__), block_fn)
        InterBlock = getattr(importlib.import_module(".dinov2.layers.blocks", __package__), fuse_block_fn)
        
        self.backbone = self.hub_loader(
            pretrained=False,
            # block_fn=partial(BaseBlock, attn_class=MemEffAttention),
            block_fn=BaseBlock,
            fuse_block_fn=InterBlock,
            fuse_block_layers=fuse_block_layers)

        """
        Function calling stack
        >> hub_loader = {arch_name}(pretrained, **kwargs) ex. def dinov2_vits14()
        >> _make_dinov2_model(arch_name="vit_base", pretrained, **kwargs)
        >> vits.__dict__[arch_name](**vit_kwargs) <- includes **kwargs
        """

        self.dim_features = self.backbone.blocks[0].attn.qkv.in_features
        self.num_features = intermediate_layers if isinstance(intermediate_layers, int) else len(intermediate_layers)

        self.output_projections = nn.ModuleList([
            nn.Conv2d(in_channels=self.dim_features, out_channels=dim_out, kernel_size=1, stride=1, padding=0,) 
                for _ in range(self.num_features)
        ])

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.interp_mode = interp_mode
        self.depth_fill = depth_fill
    
    @property
    def onnx_compatible_mode(self):
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value
        self.backbone.onnx_compatible_mode = value

    def init_weights(self):
        pretrained_backbone_state_dict = self.hub_loader(pretrained=True).state_dict()

        new_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_backbone_state_dict.items() if k in new_dict and v.shape == new_dict[k].shape}
        new_dict.update(pretrained_dict)
        
        self.backbone.load_state_dict(new_dict)
        
        # Initialize depth branch
        self.backbone.init_weights_multimodal()

    def enable_gradient_checkpointing(self):
        for i in range(len(self.backbone.blocks)):
            wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])

    def enable_pytorch_native_sdpa(self):
        for i in range(len(self.backbone.blocks)):
            wrap_dinov2_attention_with_sdpa(self.backbone.blocks[i].attn)

    def forward(
        self, 
        image: torch.Tensor, 
        depth: Union[torch.Tensor, List[torch.Tensor]], 
        depth_mask: torch.Tensor, 
        image_token_shape: Tuple[Union[int, torch.LongTensor], Union[int, torch.LongTensor]], 
        depth_token_shape: Tuple[Union[int, torch.LongTensor], Union[int, torch.LongTensor]],
        return_class_token: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # forward function with two multimodal input tensors

        # Process image input
        image_required_size = (image_token_shape[0] * 14, image_token_shape[1] * 14)
        image_resized = F.interpolate(image, image_required_size, mode="bilinear", align_corners=False, antialias=not self.onnx_compatible_mode)
        image_resized = (image_resized - self.image_mean) / self.image_std

        # Resize depth & depth mask input
        depth_required_size = (depth_token_shape[0] * 14, depth_token_shape[1] * 14)
        if self.interp_mode == 'masked_nearest':
            if isinstance(depth, list):
                depth_resized, depth_mask_resized = zip(*[
                    utils3d.torch.masked_nearest_resize(d, mask=m, size=depth_required_size)
                        for d, m in zip(depth, depth_mask)
                ])
                depth_resized = torch.stack(depth_resized, dim=0)
                depth_mask_resized = torch.stack(depth_mask_resized, dim=0)
            else:
                depth_resized, depth_mask_resized = utils3d.torch.masked_nearest_resize(depth, mask=depth_mask, size=depth_required_size)
            if self.depth_fill:
                depth_resized = utils3d.pt.flood_fill(depth_resized, mask=depth_mask_resized)
            else:
                depth_resized[~depth_mask_resized] = 0.0
            depth_mask_resized = depth_mask_resized.float()
        elif self.interp_mode == 'bilinear':
            if isinstance(depth, list):
                depth_resized = torch.stack([
                    F.interpolate(d[None, None], depth_required_size, mode="bilinear", align_corners=False, antialias=False)[0, 0]
                        for d in depth
                ], dim=0)
                depth_mask_resized = torch.stack([
                    F.interpolate(m[None, None].float(), depth_required_size, mode="bilinear", align_corners=False, antialias=False)[0, 0]
                        for m in depth_mask
                ], dim=0)
            else:
                depth_resized = F.interpolate(depth[:, None], depth_required_size, mode="bilinear", align_corners=False, antialias=False)[:, 0]
                depth_mask_resized = F.interpolate(depth_mask.float()[:, None], depth_required_size, mode="bilinear", align_corners=False, antialias=False)[:, 0]
        depth_stack_resized = torch.stack([depth_resized, depth_resized, depth_mask_resized], dim=1)
        depth_stack_resized = 2.0 * depth_stack_resized - 1.0

        # Get intermediate layers from the backbone
        features = self.backbone.get_intermediate_layers_multimodal(image_resized, depth_stack_resized, n=self.intermediate_layers, return_class_token=True)

        # Project features to the desired dimensionality
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, image_token_shape).contiguous())
                for proj, (feat, clstoken) in zip(self.output_projections, features)
        ], dim=1).sum(dim=1)                    

        if return_class_token:
            return x, features[-1][1]
        else:
            return x