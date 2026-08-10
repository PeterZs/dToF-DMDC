from typing import *
import numpy as np
import utils3d
import cv2


def remove_square(
    depth: np.ndarray,
    mask: np.ndarray,
    ratio: float = 0.5,
    rng: np.random.Generator = None
):
    height, width = mask.shape[-2:]
    square_h = round(rng.uniform(0, ratio) * height)
    square_w = round(rng.uniform(0, ratio) * width)

    top = rng.integers(0, height - square_h)
    left = rng.integers(0, width - square_w)

    new_mask = mask.copy()
    new_mask[top:top + square_h, left:left + square_w] = False
    new_depth = np.where(new_mask, depth, np.nan)
    
    return new_depth, new_mask


def remove_edge(
    depth: np.ndarray,
    mask: np.ndarray,
    thickness: int,
    ref_depth: np.ndarray = None,
    ref_mask: np.ndarray = None,
    tol: float = 0.1,
    rng = None
):
    rng = np.random.default_rng(0) if rng == None else rng

    height, width = depth.shape[-2:]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    if ref_depth is None:
        ref_depth, ref_mask = depth, mask
    
    edge_mask = utils3d.np.depth_map_edge(ref_depth, ref_mask, kernel_size=2 * thickness + 1, rtol=tol)

    new_mask = mask & ~edge_mask
    new_depth = np.where(new_mask, depth, np.nan)

    return new_depth, new_mask


def remove_perlin(
    depth: np.ndarray,
    mask: np.ndarray,
    frequency: float = 4,
    threshold: float = 0.3,
    rng: np.random.Generator = None
):  
    rng = np.random.default_rng() if rng == None else rng
    
    height, width = depth.shape[-2:]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    if isinstance(frequency, (tuple, list)):
        frequency = rng.uniform(*frequency)

    max_noise_size = 360
    noise_h, noise_w = round(height * min(1, max_noise_size / diagonal)), round(width * min(1, max_noise_size / diagonal))
    noise = utils3d.np.fractal_perlin_noise_map(
        (noise_h, noise_w),
        base_frequency=frequency,
        octaves=4,
        gain=0.7,
        seed=rng.integers(0, np.iinfo(np.int32).max)
    )
    if noise_h != height or noise_w != width:
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)

    new_mask = mask & (noise < threshold)
    new_depth = np.where(new_mask, depth, np.nan)

    return new_depth, new_mask


def boundary_noise(
    depth:np.ndarray,
    mask:np.ndarray,
    rng,
    thickness:int = 1,
    tol:float = 0.1)-> np.array:
    
    height, width = depth.shape[-2:]

    # edge_mask = depth_occlusion_edge_numpy(depth, mask=mask, thickness=2, tol=0.01)
    disp = np.where(mask, 1 / depth, 0)
    disp_pad = np.pad(disp, (thickness, thickness), constant_values=0)
    mask_pad = np.pad(mask, (thickness, thickness), constant_values=False)
    kernel_size = 2 * thickness + 1
    disp_window = utils3d.numpy.sliding_window(disp_pad, (kernel_size, kernel_size), 1, axis=(-2, -1))  # [..., H, W, kernel_size ** 2]
    mask_window = utils3d.numpy.sliding_window(mask_pad, (kernel_size, kernel_size), 1, axis=(-2, -1))  # [..., H, W, kernel_size ** 2]

    disp_mean = weighted_mean_numpy(disp_window, mask_window, axis=(-2, -1))
    fg_edge_mask = mask & (disp > (1 + tol) * disp_mean)
    bg_edge_mask = mask & (disp_mean > (1 + tol) * disp)

    disp_max = np.max(disp_window.reshape(height, width, -1), axis=-1)
    disp_min = np.min(disp_window.reshape(height, width, -1), axis=-1)
    
    noise = rng.uniform(0, 1, size=depth.shape) * (disp_max - disp_min) + disp_min
    
    edge_mask = (cv2.dilate(fg_edge_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=thickness) > 0) \
        & (cv2.dilate(bg_edge_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=thickness) > 0)

    disp[edge_mask] = noise[edge_mask]

    depth = np.where(mask&(disp>1e-6), 1/disp, 0)

    return depth, mask&(disp>1e-6)


def outlier_noise(
    depth: np.ndarray,
    mask: np.ndarray,
    ratio: float = 0.001,
    rng: np.random.Generator = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    row_idx, col_idx = np.nonzero(mask)

    if len(row_idx) == 0:
        return depth, mask
    
    depth = depth.copy()
    d_min, d_max = depth[row_idx, col_idx].min(), depth[row_idx, col_idx].max()
    if not (d_min > 0 and d_max > d_min):
        return depth, mask
    outlier = np.where(rng.uniform(0, 1, size=len(row_idx)) < ratio)[0]
    depth[row_idx[outlier], col_idx[outlier]] = np.exp(rng.uniform(np.log(d_min), np.log(d_max), size=len(outlier)))

    return depth, mask


def depth_rel_gaussian_noise(
    depth:np.ndarray,
    mask:np.ndarray,
    rng,
    noise: Literal["gaussian", "uniform", None] = None,
    noise_scale: float = 0.0,
    noise_ratio: float = 0.0
) -> np.array:
    
    depth = np.where(mask & (depth > 0), depth, np.nan).astype(np.float32)
    mask = mask & (depth > 0)

    max_range = 1e3
    s_min = max(depth[mask].min(), 1e-5)
    s_max = max(s_min * 1.1, min(depth[mask].max(), s_min * max_range))

    row_idx, col_idx = np.nonzero(mask)

    if noise is not None:
        if noise_scale > 0.0:
            # If noise depth-invariant scalar
            mean = 0.0
            sigma = noise_scale
        else:
            # If noise is depth-variant array
            mean = np.zeros_like(depth[mask])
            sigma = np.clip(depth[mask], min=s_min, max=s_max) * noise_ratio

        if noise == 'gaussian':
            """
            noise_scale: standard devlation
            1 x sigma 68   %
            2 x sigma 95   %
            3 x sigma 99.7 %
            """
            noise = rng.normal(mean, sigma * 0.5, len(row_idx)).astype(np.float32)
        elif noise == 'uniform':
            noise = rng.uniform(-sigma, sigma, len(row_idx)).astype(np.float32)
    else:
        noise = 0.0

    depth[mask] = depth[mask] + noise.astype(np.float32)

    return depth, mask


def perlin_noise_erosion(
    depth: np.ndarray,
    mask: np.ndarray,
    thickness: int,
    rng: np.random.Generator = None
) -> np.ndarray:
    max_noise_size = 360  
    if rng is None:
        rng = np.random.default_rng()
    height, width = depth.shape[-2:]
    diagonal = (height ** 2 + width ** 2) ** 0.5
    
    noise_h, noise_w = round(height * min(1, max_noise_size / diagonal)), round(width * min(1, max_noise_size / diagonal))
    noise = utils3d.np.fractal_perlin_noise_map(
        (noise_h, noise_w), 
        base_frequency=16,
        octaves=4,
        gain=0.7,
        seed=rng.integers(0, np.iinfo(np.int32).max)
    )
    if noise_h != height or noise_w != width:
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)

    radius = np.round(thickness * noise).clip(-thickness, thickness)

    result_depth = depth
    for i in range(1, thickness + 1):
        eroded = cv2.erode(np.where(mask, result_depth, np.inf), np.ones((3, 3), dtype=np.float32), iterations=1)
        dilated = cv2.dilate(np.where(mask, result_depth, -np.inf), np.ones((3, 3), dtype=np.float32), iterations=1)
        result_depth = np.where(mask, np.where(radius <= -i, eroded, np.where(radius >= i, dilated, result_depth)), np.nan)

    return result_depth

