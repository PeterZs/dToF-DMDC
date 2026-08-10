import os
import sys
from typing import *
import importlib

import click
import torch
import utils3d

from dtof_dmdc.test.interface import MonoDepthRefineInterface


class Baseline(MonoDepthRefineInterface):

    def __init__(self, num_image_tokens: int, num_depth_tokens: int, resolution_level: int, pretrained_model_name_or_path: str, use_fp16: bool, device: str = 'cuda:0', variant: str = 'vits'):
        super().__init__()
        from dtof_dmdc.model import import_model_class_by_version
        MoGeModel = import_model_class_by_version('dmdc')

        # `variant` selects the checkpoint filename when loading from the HF hub; it is
        # ignored when `pretrained_model_name_or_path` is a local checkpoint path.
        self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path, filename=f'dtof_dmdc_{variant}.pt').to(device).eval()
        
        self.device = torch.device(device)
        self.num_image_tokens = num_image_tokens
        self.num_depth_tokens = num_depth_tokens
        self.resolution_level = resolution_level
        self.use_fp16 = use_fp16
        if use_fp16:
            self.model.half()
    
    @click.command()
    @click.option('--num_image_tokens', type=int, default=None)
    @click.option('--num_depth_tokens', type=int, default=None)
    @click.option('--resolution_level', type=int, default=9)
    @click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='summerbell/dToF-DMDC')
    @click.option('--variant', type=click.Choice(['vits', 'vitb', 'vitl']), default='vits', help='Backbone variant to download from the HF hub. Ignored when --pretrained is a local checkpoint path.')
    @click.option('--fp16', 'use_fp16', is_flag=True)
    @click.option('--device', type=str, default='cuda:0')
    @staticmethod
    def load(num_image_tokens: int, num_depth_tokens: int, resolution_level: int, pretrained_model_name_or_path: str, variant: str, use_fp16: bool, device: str = 'cuda:0'):
        return Baseline(num_image_tokens, num_depth_tokens, resolution_level, pretrained_model_name_or_path, use_fp16, device, variant)

    # Implementation for inference
    @torch.inference_mode()
    def infer(self, image: torch.FloatTensor, depth: torch.FloatTensor, depth_mask: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor] = None):

        if intrinsics is not None:
            fov_x, _ = utils3d.torch.intrinsics_to_fov(intrinsics)
            fov_x = torch.rad2deg(fov_x)
        else:
            fov_x = None

        output = self.model.infer(image, depth, depth_mask, fov_x=fov_x, apply_mask=True, num_image_tokens=self.num_image_tokens, num_depth_tokens=self.num_depth_tokens, use_fp16=self.use_fp16)

        ret = {'mask': output['mask']}
        if 'points' in output and 'depth' in output and 'intrinsics' in output:
            ret.update({
                'points_scale_invariant': output['points'],
                'depth_scale_invariant': output['depth'],
                'intrinsics': output['intrinsics'],
            })
        if 'metric_depth' in output:
            ret.update({
                'depth_metric': output['metric_depth'].squeeze()
            })
        return ret

    @torch.inference_mode()
    def infer_for_evaluation(self, image: torch.FloatTensor, depth: torch.FloatTensor, depth_mask: torch.FloatTensor, intrinsics: torch.FloatTensor = None):
        if intrinsics is not None:
            fov_x, _ = utils3d.torch.intrinsics_to_fov(intrinsics)
            fov_x = torch.rad2deg(fov_x)
        else:
            fov_x = None

        output = self.model.infer(image, depth, depth_mask, fov_x=fov_x, apply_mask=False, num_image_tokens=self.num_image_tokens, num_depth_tokens=self.num_depth_tokens, use_fp16=self.use_fp16)
        
        ret = {'mask': output['mask']}

        if 'points' in output and 'depth' in output and 'intrinsics'in output:
            ret.update({
                'points_scale_invariant': output['points'],
                'depth_scale_invariant': output['depth'],
                'intrinsics': output['intrinsics']
            })
        if 'metric_depth' in output:
            ret.update({
                'depth_metric': output['metric_depth'].squeeze()
            })
        return ret
        