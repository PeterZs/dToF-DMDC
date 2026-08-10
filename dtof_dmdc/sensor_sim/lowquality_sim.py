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


def sample_lowquality(
    depth: np.ndarray,
    mask: np.ndarray,
    erosion_thickness: Union[float, Tuple[float, float]] = 0.01,
    noise_strength: Union[float, Tuple[float, float]] = (0.001, 0.01),
    removal: float = 0.,
    num_samples: int = None,
    rng: np.random.Generator = None,
):
    # Downsample
    height, width = depth.shape[-2:]

    # Edge erosion and dilation
    if isinstance(erosion_thickness, (tuple, list)):
        erosion_thickness = rng.uniform(*erosion_thickness)
    erosion_thickness_i = round(erosion_thickness * (height ** 2 + width ** 2) ** 0.5)
    depth = perlin_noise_erosion(depth, mask, erosion_thickness_i, rng=rng)
    
    # Gaussian noise
    if isinstance(noise_strength, (tuple, list)):
        noise_strength = rng.uniform(*noise_strength)
    depth = depth * (1 + rng.normal(0, noise_strength, size=depth.shape).astype(depth.dtype))

    # Random sampling (if num_samples is specified, otherwise dense)
    if num_samples is not None:
        valid_indices = np.argwhere(mask)
        if len(valid_indices) <= num_samples:
            sampled_mask = mask.copy()
        else:
            sampled_indices = valid_indices[rng.choice(len(valid_indices), size=num_samples, replace=False)]
            sampled_mask = np.zeros_like(mask, dtype=bool)
            sampled_mask[tuple(sampled_indices.T)] = True
        depth = np.where(sampled_mask, depth, np.nan)
        mask = sampled_mask

    # Random removal
    if rng.uniform() < removal:
        edge_removal_thickness = round(rng.uniform(0.005, 0.01) * (height ** 2 + width ** 2) ** 0.5)
        depth, mask = remove_edge(depth, mask, thickness=edge_removal_thickness, rng=rng)
    if rng.uniform() < removal:
        depth, mask = remove_perlin(depth, mask, frequency=rng.uniform(4, 16), rng=rng)
    if rng.uniform() < removal:
        depth, mask = remove_square(depth, mask, rng=rng)

    depth = np.where(mask, depth, np.nan)

    return depth, mask
