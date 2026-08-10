import numpy as np
from typing import Callable, List, Any, Tuple, Dict, Literal, Union
from pathlib import Path
import sys
from scipy.ndimage import distance_transform_edt, gaussian_filter
from .utils import remove_edge, remove_perlin, remove_square, perlin_noise_erosion

import os
import cv2
import random
import utils3d


def sample_lowres(
    depth: np.ndarray,
    mask: np.ndarray,
    resolution: Union[int, Tuple[int, int]] = (96, 360),
    erosion_thickness: Union[float, Tuple[float, float]] = 0,
    noise_strength: Union[float, Tuple[float, float]] = (0.001, 0.01),
    removal: float = 0.2,
    rng: np.random.Generator = None,
):
    # Downsample
    height, width = depth.shape[-2:]
    diagonal = (height ** 2 + width ** 2) ** 0.5
    aspect_ratio = width / height
    if isinstance(resolution, (tuple, list)):
        resolution = rng.uniform(*resolution)
    height_lr, width_lr =  round(height * resolution / diagonal), round(width * resolution / diagonal)
    depth_lr = cv2.resize(depth, (width_lr, height_lr), interpolation=cv2.INTER_NEAREST)
    mask_lr = cv2.resize(mask.astype(np.uint8), (width_lr, height_lr), interpolation=cv2.INTER_NEAREST) > 0

    # Edge erosion and dilation
    if isinstance(erosion_thickness, (tuple, list)):
        erosion_thickness = rng.uniform(*erosion_thickness)
        erosion_thickness_i = round(erosion_thickness * (height_lr ** 2 + width_lr ** 2) ** 0.5)
    depth_lr = perlin_noise_erosion(depth_lr, mask_lr, erosion_thickness_i, rng=rng)

    # Gaussian noise
    if isinstance(noise_strength, (tuple, list)):
        noise_strength = rng.uniform(*noise_strength)
    
    noise = rng.normal(0, noise_strength, size=depth_lr.shape).astype(depth_lr.dtype)
    depth_lr = depth_lr * (1 + noise)

    # Random removal
    if rng.uniform() < removal:
        edge_removal_thickness = round(rng.uniform(0.005, 0.01) * (height_lr ** 2 + width_lr ** 2) ** 0.5)
        depth_lr, mask_lr = remove_edge(depth_lr, mask_lr, thickness=edge_removal_thickness, rng=rng)
    if rng.uniform() < removal:
        depth_lr, mask_lr = remove_perlin(depth_lr, mask_lr, rng=rng, frequency=rng.uniform(4, 16))
    if rng.uniform() < removal:
        depth_lr, mask_lr = remove_square(depth_lr, mask_lr, rng=rng)

    depth_lr = np.where(mask_lr, depth_lr, np.nan)

    return depth_lr, mask_lr
