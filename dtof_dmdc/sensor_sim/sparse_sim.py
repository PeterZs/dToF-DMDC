import numpy as np
from typing import Callable, List, Any, Tuple, Dict, Literal, Union
import utils3d
import time
import math

from .utils import outlier_noise, remove_edge, remove_perlin, remove_square, perlin_noise_erosion

# random sample
def sample_random_sparse(
    depth: np.ndarray,
    depth_mask: np.ndarray,
    num_samples: Union[int, Tuple[int, int]] = 1000,
    noise_strength: Union[float, Tuple[float, float]] = 0,
    erosion_thickness: Union[float, Tuple[float, float]] = 0,
    jitter_strength: Union[float, Tuple[float, float]] = 0,
    outlier_ratio: Union[float, Tuple[float, float]] = 0,
    removal: float = 0,
    removal_perlin_threshold: Union[float, Tuple[float, float]] = 0.3,
    removal_square_ratio: Union[float, Tuple[float, float]] = 0.5,
    rng: np.random.Generator = None,
    **deprecated_kwargs
) -> np.ndarray:
    height, width = depth.shape[-2:]
    
    sample_depth, sample_depth_mask = depth, depth_mask

    # Erosion
    if erosion_thickness:
        if isinstance(erosion_thickness, (tuple, list)):
            erosion_thickness = rng.uniform(*erosion_thickness)
        erosion_thickness_i = round(erosion_thickness * (height ** 2 + width ** 2) ** 0.5)
        sample_depth = perlin_noise_erosion(sample_depth, sample_depth_mask, erosion_thickness_i, rng=rng)

    # Gaussian noise
    if noise_strength:
        if isinstance(noise_strength, (tuple, list)):
            noise_strength = rng.uniform(*noise_strength)
        sample_depth = sample_depth * (1 + rng.normal(0, noise_strength, size=sample_depth.shape).astype(depth.dtype)).clip(min=0.1)

    # Sparse sampling with optional jitter
    src_pts = np.argwhere(depth_mask)
    if isinstance(num_samples, (tuple, list)):
        num_samples = round(math.exp(rng.uniform(math.log(num_samples[0]), math.log(num_samples[1]))))
    sample_idx = rng.choice(len(src_pts), size=min(len(src_pts), num_samples), replace=False)
    src_pts = src_pts[sample_idx]
    if jitter_strength:
        if isinstance(jitter_strength, (tuple, list)):
            jitter_strength = rng.uniform(*jitter_strength)
        jitter_dist_std = jitter_strength * (height ** 2 + width ** 2) ** 0.5
        dst_pts = src_pts + np.round(rng.normal(0, jitter_dist_std, size=src_pts.shape)).astype(int)
        dst_pts = np.clip(dst_pts, 0, [height - 1, width - 1])
    else:
        dst_pts = src_pts

    sample_depth_ = np.ones_like(depth)
    sample_depth_mask_ = np.zeros(depth.shape, dtype=bool)
    sample_depth_[dst_pts[:, 0], dst_pts[:, 1]] = sample_depth[src_pts[:, 0], src_pts[:, 1]]
    sample_depth_mask_[dst_pts[:, 0], dst_pts[:, 1]] = True
    sample_depth = sample_depth_
    sample_depth_mask = sample_depth_mask_

    # Outliers
    if outlier_ratio:
        if isinstance(outlier_ratio, (tuple, list)):
            outlier_ratio = rng.uniform(*outlier_ratio)
        sample_depth, sample_depth_mask = outlier_noise(
            sample_depth, 
            sample_depth_mask,
            ratio=outlier_ratio,
            rng=rng
        )

    # Random removal
    if rng.uniform() < removal:
        if isinstance(removal_perlin_threshold, (tuple, list)):
            removal_perlin_threshold = rng.uniform(*removal_perlin_threshold)
        sample_depth, sample_depth_mask = remove_perlin(sample_depth, sample_depth_mask, frequency=rng.uniform(4, 16), threshold=removal_perlin_threshold, rng=rng)
    if rng.uniform() < removal:
        if isinstance(removal_square_ratio, (tuple, list)):
            removal_square_ratio = rng.uniform(*removal_square_ratio)
        sample_depth, sample_depth_mask = remove_square(sample_depth, sample_depth_mask, ratio=removal_square_ratio, rng=rng)

    sample_depth = np.where(sample_depth_mask, sample_depth, np.nan)

    return sample_depth, sample_depth_mask


def sample_random_grid(
    depth: np.ndarray,
    depth_mask: np.ndarray,
    depth_fov: Tuple[float, float],              # (vertical, horizontal) angular degree
    sensor_fov: Tuple[float, float] = (40, 360), # (vertical, horizontal) angular degree
    sensor_resolution: Tuple[float, float] = (0.16, 1.25), # angular resolution (degree)
    jitter: bool = False,
    drop_p: float = 0.0,
    # random_seed: int = 0
    rng=None
) -> np.ndarray:
    height, width = depth.shape

    sample_depth = np.ones_like(depth)
    sample_depth_mask = np.zeros_like(depth_mask).astype(np.bool)

    fov_x = min(depth_fov[1], sensor_fov[1]) # horizontal
    fov_y = min(depth_fov[0], sensor_fov[0]) # vertical
    
    center_x = rng.uniform(-0.5, 0.5) * (depth_fov[1]-fov_x) if fov_x < depth_fov[1] else 0
    center_y = rng.uniform(-0.5, 0.5) * (depth_fov[0]-fov_y) if fov_y < depth_fov[0] else 0

    num_x = int(fov_x / sensor_resolution[1])
    num_y = int(fov_y / sensor_resolution[0])

    longitude = np.arange(-(num_x-1)//2, (num_x+1)//2) * sensor_resolution[1] + center_x
    latitude = np.arange(-(num_y-1)//2, (num_y+1)//2) * sensor_resolution[0] + center_y

    focal_x = (width/2) / np.tan(np.deg2rad(depth_fov[1]/2))
    focal_y = (height/2) / np.tan(np.deg2rad(depth_fov[0]/2))

    col = width / 2 + np.tan(np.deg2rad(longitude)) * focal_x
    row = height / 2 + np.tan(np.deg2rad(latitude)) * focal_y

    
    x, y = np.meshgrid(col.astype(np.int64), row.astype(np.int64), indexing='xy')

    if jitter:
        offset_x = rng.uniform(-2, 2, (num_y, num_x)).astype(np.int64)
        offset_y = rng.uniform(-2, 2, (num_y, num_x)).astype(np.int64)
        x += offset_x
        y += offset_y
        x = np.clip(x, 0, width-1)
        y = np.clip(y, 0, height-1)
    
    x, y = x.flatten(), y.flatten()

    if drop_p > 0:
        
        drop_mask = rng.uniform(0.0, 1.0, len(x)) > drop_p

        x = x[drop_mask]
        y = y[drop_mask]

    
    sample_depth_mask[y, x] = True

    sample_depth_mask = np.logical_and(sample_depth_mask, depth_mask)
    sample_depth = np.where(sample_depth_mask, depth, sample_depth_mask)        
    
    return sample_depth, sample_depth_mask
