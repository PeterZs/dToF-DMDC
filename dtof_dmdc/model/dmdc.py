from typing import *
from numbers import Number
from functools import partial
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.amp
import torch.version
import utils3d
from huggingface_hub import hf_hub_download

from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift, angle_diff_vec3
from .utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing
from .modules import DINOv2Encoder, MLP, ConvStack
from .encoder import MultiModalEncoder

    
class MoGeModel(nn.Module):
    encoder: MultiModalEncoder
    neck: ConvStack
    points_head: ConvStack
    mask_head: ConvStack
    scale_head: MLP

    def __init__(self, 
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        points_head: Dict[str, Any] = None,
        mask_head: Dict[str, Any] = None,
        normal_head: Dict[str, Any] = None,
        scale_head: Dict[str, Any] = None,
        depth_head: Dict[str, Any] = None,
        input_depth_format: List[str] = None,
        remap_output: Literal['linear', 'sinh', 'exp', 'sinh_exp'] = 'linear',
        num_image_tokens_range: List[int] = [1200, 3600],
        num_depth_tokens_range: List[int] = [1200, 3600],
        **deprecated_kwargs
    ):
        super(MoGeModel, self).__init__()
        if deprecated_kwargs:
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        self.remap_output = remap_output
        self.num_image_tokens_range = num_image_tokens_range
        self.num_depth_tokens_range = num_depth_tokens_range
        
        self.encoder = MultiModalEncoder(**encoder) 
        self.neck = ConvStack(**neck)
        if points_head is not None:
            self.points_head = ConvStack(**points_head) 
        if mask_head is not None:
            self.mask_head = ConvStack(**mask_head)
        if normal_head is not None:
            self.normal_head = ConvStack(**normal_head)
        if scale_head is not None:
            self.scale_head = MLP(**scale_head)
        
        if depth_head is not None:
            self.depth_head = ConvStack(**depth_head)

        self.input_depth_format = input_depth_format

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path, IO[bytes]], model_kwargs: Optional[Dict[str, Any]] = None, filename: str = 'dtof_dmdc_vits.pt', **hf_kwargs) -> 'MoGeModel':
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to a local checkpoint file, or a Hugging Face repo id.
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.
        - `filename`: checkpoint filename to download when `pretrained_model_name_or_path` is a repo id.
          One of `dtof_dmdc_vits.pt` (default, paper model), `dtof_dmdc_vitb.pt`, `dtof_dmdc_vitl.pt`.
          Ignored when a local checkpoint path is given.
        - `hf_kwargs`: additional keyword arguments to pass to the `hf_hub_download` function. Ignored if `pretrained_model_name_or_path` is a local path.

        ### Returns:
        - A new instance of dToF-DMDC (`MoGeModel`) with the parameters loaded from the checkpoint.
        """
        if Path(pretrained_model_name_or_path).exists():
            checkpoint_path = pretrained_model_name_or_path
        else:
            checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename=filename,
                **hf_kwargs
            )
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        model_config = checkpoint['model_config']
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model'], strict=False)
        
        return model
    
    def init_weights(self):
        self.encoder.init_weights()

    def enable_gradient_checkpointing(self):
        self.encoder.enable_gradient_checkpointing()
        self.neck.enable_gradient_checkpointing()
        for head in ['points_head', 'normal_head', 'mask_head', 'depth_head']:
            if hasattr(self, head):
                getattr(self, head).enable_gradient_checkpointing()

    def enable_pytorch_native_sdpa(self):
        self.encoder.enable_pytorch_native_sdpa()

    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.remap_output == 'linear':
            pass
        elif self.remap_output =='sinh':
            points = torch.sinh(points)
        elif self.remap_output == 'exp':
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output =='sinh_exp':
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points

    def normalize_depth(self, depth: Union[torch.Tensor, List[torch.Tensor]], mask: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor, torch.Tensor]:
        "Normalize depth 0, 1"
        if self.input_depth_format == 'depth_lognorm':
            if isinstance(depth, list):
                d_min = torch.stack([torch.min(torch.where(m, d, torch.inf)) for d, m in zip(depth, mask)])
                d_max = torch.stack([torch.max(torch.where(m, d, -torch.inf)) for d, m in zip(depth, mask)])
            else:
                d_min = torch.min(torch.where(mask, depth, torch.inf).flatten(-2), dim=-1).values
                d_max = torch.max(torch.where(mask, depth, -torch.inf).flatten(-2), dim=-1).values

            alpha = torch.log(d_max) - torch.log(d_min)
            beta = torch.log(d_min)

        elif self.input_depth_format == 'depth_logscale':
            if isinstance(depth, list):
                d_mean = torch.stack([torch.nanmean(torch.where(m, d, torch.nan)) for d, m in zip(depth, mask)])
            else:
                d_mean = torch.nanmean(torch.where(mask, depth, torch.nan), dim=(-2, -1))

            alpha = torch.full_like(d_mean, 3.0)
            beta = d_mean - 0.5 * alpha

        alpha = torch.maximum(torch.nan_to_num(alpha, nan=1.0, posinf=1.0, neginf=1.0), torch.tensor(1e-7, device=alpha.device, dtype=alpha.dtype))
        beta = torch.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)

        if isinstance(depth, list):
            depth_normed = [(torch.log(d) - b) / a for d, a, b in zip(depth, alpha, beta)]
        else:
            depth_normed = (torch.log(depth) - beta) / alpha

        return depth_normed, alpha, beta


    def forward(self, image: torch.Tensor, in_depth: Union[List[torch.Tensor], torch.Tensor], in_depth_mask: Union[List[torch.Tensor], torch.Tensor], num_image_tokens: int, num_depth_tokens: int) -> Dict[str, torch.Tensor]:
        """
        Example dimension
            in_depth        [8, 397, 746]
            in_depth_mask   [8, 397, 746]
            s_min           [8]
            stacked_input   [8, 3, 397, 746]
        """
        
        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype

        aspect_ratio = img_w / img_h
        image_base_h, image_base_w = round((num_image_tokens / aspect_ratio) ** 0.5), round((num_image_tokens * aspect_ratio) ** 0.5)
        depth_base_h, depth_base_w = round((num_depth_tokens / aspect_ratio) ** 0.5), round((num_depth_tokens * aspect_ratio) ** 0.5)

        # Normalize input depth to 0 ~ 1
        in_depth, alpha, beta = self.normalize_depth(in_depth, in_depth_mask)

        # Backbone encoding
        features, cls_token = self.encoder(
            image,
            in_depth,
            in_depth_mask,
            (image_base_h, image_base_w),
            (depth_base_h, depth_base_w),
            return_class_token=True,
        )

        # Shared neck
        features_ = self.neck([features, None, None, None, None])

        # Heads decoding
        points, normal, mask, depth, linear_depth = (getattr(self, head)(features_)[-1] if hasattr(self, head) else None for head in ['points_head', 'normal_head', 'mask_head', 'depth_head', 'linear_depth_head'])
        metric_scale = self.scale_head(cls_token) if hasattr(self, 'scale_head') else None
        
        # Resize
        points, normal, mask, depth, linear_depth = (F.interpolate(v, (img_h, img_w), mode='bilinear', align_corners=False, antialias=False) if v is not None else None for v in [points, normal, mask, depth, linear_depth])
        
        # Remap output
        return_dict = {}
        if points is not None:
            points = points.permute(0, 2, 3, 1)
            points = self._remap_points(points)     # slightly improves the performance in case of very large output values
        if normal is not None:
            normal = normal.permute(0, 2, 3, 1)
            normal = F.normalize(normal, dim=-1)
        if mask is not None:
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()
        if depth is not None:
            depth = depth.squeeze(1)
            depth = torch.exp(alpha[:, None, None] * depth + beta[:, None, None])

        return_dict = {
            'points': points, 
            'normal': normal,
            'mask': mask,
            'metric_scale': metric_scale,
            'metric_depth': depth,
        }

        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        return return_dict

    @torch.inference_mode()
    def infer(
        self, 
        image: torch.Tensor, 
        in_depth: torch.Tensor,
        in_depth_mask: torch.Tensor,
        num_image_tokens: int = None,
        num_depth_tokens: int = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: bool = True,
        fov_x: Optional[Union[Number, torch.Tensor]] = None,
        use_fp16: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        User-friendly inference function

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W)
        - `num_tokens`: the number of base ViT tokens to use for inference, `'least'` or `'most'` or an integer. Suggested range: 1200 ~ 2500. 
            More tokens will result in significantly higher accuracy and finer details, but slower inference time. Default: `'most'`. 
        - `force_projection`: if True, the output point map will be computed using the actual depth map. Default: True
        - `apply_mask`: if True, the output point map will be masked using the predicted mask. Default: True
        - `fov_x`: the horizontal camera FoV in degrees. If None, it will be inferred from the predicted point map. Default: None
        - `use_fp16`: if True, use mixed precision to speed up inference. Default: True
            
        ### Returns

        A dictionary containing the following keys:
        - `points`: output tensor of shape (B, H, W, 3) or (H, W, 3).
        - `depth`: tensor of shape (B, H, W) or (H, W) containing the depth map.
        - `intrinsics`: tensor of shape (B, 3, 3) or (3, 3) containing the camera intrinsics.
        """
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
            in_depth = [in_depth]
            in_depth_mask = [in_depth_mask]
        else:
            omit_batch_dim = False

        image = image.to(device=self.device, dtype=self.dtype)
        in_depth = [d.to(device=self.device, dtype=self.dtype) for d in in_depth] if isinstance(in_depth, list) else in_depth.to(device=self.device, dtype=self.dtype)
        in_depth_mask = [m.to(device=self.device) for m in in_depth_mask] if isinstance(in_depth_mask, list) else in_depth_mask.to(device=self.device)

        original_height, original_width = image.shape[-2:]
        area = original_height * original_width
        aspect_ratio = original_width / original_height
        
        # Determine the number of base tokens to use
        if num_image_tokens is None:
            min_tokens, max_tokens = self.num_image_tokens_range
            num_image_tokens = round(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        if num_depth_tokens is None:
            min_tokens, max_tokens = self.num_depth_tokens_range
            num_depth_tokens = round(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        # Forward pass
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_fp16):
            output = self.forward(image, in_depth, in_depth_mask, num_image_tokens=num_image_tokens, num_depth_tokens=num_depth_tokens)
        points, normal, mask, metric_scale, metric_depth, linear_depth = (output.get(k, None) for k in ['points', 'normal', 'mask', 'metric_scale', 'metric_depth', 'linear_depth'])

        # Always process the output in fp32 precision
        points, normal, mask, metric_scale, fov_x, metric_depth, linear_depth = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [points, normal, mask, metric_scale, fov_x, metric_depth, linear_depth])
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            if mask is not None:
                mask_binary = mask > 0.5
            else:
                mask_binary = None
                
            if points is not None:
                # Convert affine point map to camera-space. Recover depth and intrinsics from point map.
                # NOTE: Focal here is the focal length relative to half the image diagonal
                if fov_x is None:
                    # Recover focal and shift from predicted point map
                    focal, shift = recover_focal_shift(points, mask_binary)
                else:
                    # Focal is known, recover shift only
                    focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))
                    if focal.ndim == 0:
                        focal = focal[None].expand(points.shape[0])
                    _, shift = recover_focal_shift(points, mask_binary, focal=focal)
                fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 
                intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
                points[..., 2] += shift[..., None, None]
                if mask_binary is not None:
                    mask_binary &= points[..., 2] > 0        # in case depth is contains negative values (which should never happen in practice)
                depth = points[..., 2].clone()
            else:
                depth, intrinsics = None, None

            # If projection constraint is forced, recompute the point map using the actual depth map & intrinsics
            if force_projection and depth is not None:
                points = utils3d.torch.depth_map_to_point_map(depth, intrinsics=intrinsics)

            # Apply metric scale
            if metric_scale is not None:
                if points is not None:
                    points *= metric_scale[:, None, None, None]
                if depth is not None:
                    depth *= metric_scale[:, None, None]

            # Apply mask
            if apply_mask and mask_binary is not None:
                points = torch.where(mask_binary[..., None], points, torch.inf) if points is not None else None
                depth = torch.where(mask_binary, depth, torch.inf) if depth is not None else None
                normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal)) if normal is not None else None
                metric_depth = torch.where(mask_binary, metric_depth, torch.inf) if metric_depth is not None else None

        return_dict = {
            'points': points,
            'intrinsics': intrinsics,
            'depth': depth,
            'mask': mask_binary,
            'normal': normal,
            'metric_depth': metric_depth,
        }

        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {k: v.squeeze(0) for k, v in return_dict.items()}

        return return_dict
