from pathlib import Path
import numpy as np
import utils3d
from typing import *
import math

from .utils import outlier_noise, remove_perlin, remove_square


class LidarDevice:
    def __init__(
        self,
        pitch=0.0,
        yaw=0.0,
        fov_h=360.0,
        fov_v=30.0,
        resolution_h=0.1,
        resolution_v=2,
        max_depth=100,
        noise_option='gaussian',
        noise: Literal["gaussian", "uniform"] | None = "gaussian",
        noise_scale: float = 0.0,
        noise_strength: float = 0.0,
        drop_ratio: float = 0.1,
        jitter_strength: float = 0.0,
        rng=None,
        **deprecated_kwargs
    ):
        self.pitch = pitch
        self.yaw = yaw
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.resolution_h = resolution_h
        self.resolution_v = resolution_v
        self.max_depth = max_depth
        
        self.noise = noise
        self.noise_strength = noise_strength
        self.noise_scale = noise_scale

        self.drop_ratio = drop_ratio
        
        self.jitter_strength = jitter_strength

        self.rng = rng
    
    def create_rays(self):
        
        if isinstance(self.fov_v, float):
            fov_v_start = -self.fov_v / 2
            fov_v_end = self.fov_v / 2
        else:
            fov_v_start = self.fov_v[0]
            fov_v_end = self.fov_v[1]

        self.pitch = self.rng.uniform(-15, 15)
        self.yaw = self.rng.uniform(-self.resolution_h, self.resolution_h)
        azimuth = np.arange(self.yaw - self.fov_h / 2, self.yaw + self.fov_h / 2, self.resolution_h)[:, np.newaxis]
        elevation = np.arange(self.pitch + fov_v_start, self.pitch + fov_v_end, self.resolution_v)[np.newaxis, :]
        elevation = np.tile(elevation, (azimuth.shape[0], 1))
        directions = np.stack([
            np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth)),
            np.sin(np.radians(elevation)),
            np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))], axis=-1)
        directions = directions.reshape(-1, 3)
        return directions


    def ray_to_pixel(self, ray_dir, K):
        """
        K : normazlied intrinsics
        """
        x = ray_dir[..., 0]
        y = ray_dir[..., 1]
        z = ray_dir[..., -1]

        u, v = x / z, y / z
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = fx * u + cx
        v = fy * v + cy

        u = np.nan_to_num(u, -1, -1, -1)
        v = np.nan_to_num(v, -1, -1, -1)

        return u, v
    
    def add_gaussian_noise(self, depth):
        if self.noise is not None:
            if self.noise_scale > 0.0:   # If noise depth-invariant scalar
                mean = 0.0
                sigma = self.noise_scale
            else:   # If noise is depth-variant array
                mean = np.zeros_like(depth)
                sigma = depth * self.noise_strength

            if self.noise == 'gaussian':
                noise = self.rng.normal(mean, sigma, depth.shape)
            elif self.noise == 'uniform':
                noise = self.rng.uniform(-sigma, sigma, depth.shape)
        else:
            noise = 0.0

        return noise

    
    def scan(self, depth, depth_mask, intrinsics, return_points=False, debug=False):
        height, width = depth.shape
        
        ray_d = self.create_rays()
        u, v = self.ray_to_pixel(ray_d, intrinsics)
        u = np.clip(u * width, -1, width).astype(np.int32)
        v = np.clip(v * height, -1, height).astype(np.int32)

        mask = (u < width - 1) & (0 <= u) & (v < height - 1) & (0 <= v)
        
        u = u[mask]
        v = v[mask]

        # Drop long range depth pixel
        d = depth[v, u]
        max_depth = max(self.max_depth, np.nanquantile(d, 0.8))
        mask = d < max_depth
        u = u[mask]
        v = v[mask]

        if self.drop_ratio > 0.0:
            drop_mask = self.rng.uniform(0.0, 1.0, len(u)) > self.drop_ratio
            u = u[drop_mask]
            v = v[drop_mask]

        lidar_depth = np.zeros_like(depth)
        lidar_mask = np.zeros_like(depth_mask)

        ret = {}
        # Generate z gaussian noise
        noise = self.add_gaussian_noise(depth[v, u])

        # Generate random shift
        jitter_dist = (math.sqrt(height ** 2 + width ** 2) * self.jitter_strength)
        jitter_stdev = self.rng.uniform(0, jitter_dist)

        shift_u = self.rng.normal(0, jitter_stdev, len(u)).astype(np.int32)
        shift_v = self.rng.normal(0, jitter_stdev, len(v)).astype(np.int32)
        shifted_u = np.clip(u + shift_u, 0, width-1)
        shifted_v = np.clip(v + shift_v, 0, height-1)
        lidar_depth[shifted_v, shifted_u] = depth[v, u] + noise
        lidar_mask[shifted_v, shifted_u] = True

        """
        u_ = u.reshape(-1)
        v_ = v.reshape(-1)
        num_shift = int(len(u_)*shift_ratio)

        permute_idx = self.rng.permutation(len(u_))
        sample_idx = permute_idx[:num_jitter]
        origin_idx = permute_idx[num_jitter:]

        # Depth without shift
        lidar_depth[v[origin_idx], u[origin_idx]] = depth[v[origin_idx], u[origin_idx]] + noise[origin_idx]
        lidar_mask[v[origin_idx], u[origin_idx]] = True

        # Depth with shift
        u_ = u[sample_idx]
        v_ = v[sample_idx]
        shift_u = self.rng.normal(0, jitter_dist, len(sample_idx)).astype(np.int32)
        shift_v = self.rng.normal(0, jitter_dist, len(sample_idx)).astype(np.int32)

        shifted_u = np.clip(u_ + shift_u, 0, width-1)
        shifted_v = np.clip(v_ + shift_v, 0, height-1)
        
        lidar_depth[shifted_v, shifted_u] = depth[v_, u_] + noise[sample_idx]
        lidar_mask[shifted_v, shifted_u] = True
        """

        lidar_mask = lidar_mask & depth_mask    # 1: Lidar depth, 0: None
        
        lidar_depth[~lidar_mask] = np.nan
        ret['depth'] = lidar_depth
        ret['depth_mask'] = lidar_mask
        
        return ret


def sample_lidar(
    depth: np.ndarray, 
    depth_mask: np.ndarray, 
    intrinsics: np.ndarray, 
    rng: np.random.Generator,
    fov_v: Tuple[Union[float, Tuple[float, float]], Tuple[float, float]] = (-15, 15),
    resolution_h: Tuple[float, float] = (0.1, 0.4),
    resolution_v: Tuple[float, float] = (0.33, 4),
    max_depth: Union[float, Tuple[float, float]] = 100,
    noise_strength: Union[float, Tuple[float, float]] = 0.003,
    drop_ratio: float = 0.1,
    outlier_ratio: float = 0,
    removal: float = 0,
    jitter_strength: Union[float, Tuple[float, float]] = 0.003,
    **kwargs
):    
    def value_or_range(x: float | Tuple[float, float]) -> float:
        return rng.uniform(*x) if isinstance(x, (tuple, list)) else x
    
    lidar = LidarDevice(
        rng=rng, 
        fov_h=360,
        fov_v=(value_or_range(fov_v[0]), value_or_range(fov_v[1])),
        resolution_h=value_or_range(resolution_h),
        resolution_v=value_or_range(resolution_v),
        jitter_strength=value_or_range(jitter_strength),
        noise_strength=value_or_range(noise_strength),
        max_depth=value_or_range(max_depth),
        drop_ratio=drop_ratio,
        **kwargs
    )
    results = lidar.scan(depth, depth_mask, intrinsics)

    sample_depth, sample_depth_mask = results['depth'], results['depth_mask']

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
        sample_depth, sample_depth_mask = remove_perlin(sample_depth, sample_depth_mask, rng=rng)
    if rng.uniform() < removal:
        sample_depth, sample_depth_mask = remove_square(sample_depth, sample_depth_mask, rng=rng)

    return sample_depth, sample_depth_mask
