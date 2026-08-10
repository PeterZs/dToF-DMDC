import os
from typing import *
from pathlib import Path
import math

import numpy as np
import torch
from PIL import Image
import cv2
import utils3d
import pipeline

from ..sensor_sim import sample_random_sparse, sample_lidar, sample_lowres, sample_lowquality
from ..utils.geometry_numpy import focal_to_fov_numpy, mask_aware_nearest_resize_numpy, norm3d
from ..utils.io import *
from ..utils.tools import timeit

from .. import sensor_sim


class EvalDataLoaderPipeline:

    def __init__(
        self, 
        config: Dict[str, Any],
        num_load_workers: int = 4, 
        num_process_workers: int = 8, 
    ):
        self.path = Path(config['path'])
        split = config.get('split', '.index.txt')
        filenames = Path(self.path).joinpath(split).read_text(encoding='utf-8').splitlines()
        if 'subset_size' in config:
            subset_indices = np.linspace(0, len(filenames) - 1, min(config['subset_size'], len(filenames)), dtype=int).tolist()
            filenames = [filenames[i] for i in subset_indices]
        self.filenames = filenames

        self.config = config
        self.drop_max_depth = config.get('drop_max_depth', 1000.0)
        self.include_segmentation = config.get('include_segmentation', False)
        self.include_normal = config.get('include_normal', False)
        self.max_segments = config.get('max_segments', 100)
        self.min_seg_area = config.get('min_seg_area', 1000)
        self.depth_to_normal = config.get('depth_to_normal', False)
        self.depth_unit = config.get('depth_unit', None)
        self.has_sharp_boundary = config.get('has_sharp_boundary', False) 
        self.depth_fname = config.get('depth', "depth.png")
        self.sensor_depth_fname = config.get('sensor_depth', None)
        
        self.pipeline = pipeline.Sequential([
            self._generator,
            pipeline.Parallel([self._load_instance] * num_load_workers),
            pipeline.Parallel([self._process_instance] * num_process_workers),
            pipeline.Buffer(4)
        ])

    def __len__(self):
        return math.ceil(len(self.filenames)) 

    def _generator(self):
        for idx in range(len(self)):
            yield idx
    
    def _load_instance(self, idx):
        if idx >= len(self.filenames):
            return None
        
        path = self.path.joinpath(self.filenames[idx])

        instance = {
            'filename': self.filenames[idx],
            'seed': idx
        }
        instance['image'] = read_image(Path(path, 'image.jpg'))

        depth = read_depth(Path(path, self.depth_fname))
        instance['depth'] = depth
        if self.sensor_depth_fname is not None:
            sensor_depth = read_depth(Path(path, self.sensor_depth_fname))
            instance['sensor_depth'] = sensor_depth

        if self.include_segmentation:
            segmentation_mask, segmentation_labels = read_segmentation(Path(path,'segmentation.png'))
            instance.update({
                'segmentation_mask': segmentation_mask,
                'segmentation_labels': segmentation_labels,
            })
        
        meta = read_meta(Path(path, 'meta.json'))
        instance['intrinsics'] = np.array(meta['intrinsics'], dtype=np.float32)

        return instance

    def _process_instance(self, instance: dict):
        if instance is None:
            return None

        image, depth, intrinsics = instance['image'], instance['depth'], instance['intrinsics']
        segmentation_mask, segmentation_labels = instance.get('segmentation_mask', None), instance.get('segmentation_labels', None)
        sensor_depth = instance.get('sensor_depth', None)

        depth_mask = np.isfinite(depth)

        height, width = image.shape[:2]
        rng = np.random.default_rng(instance['seed'])

        # GT depth
        if depth.shape[:2] != image.shape[:2]:
            depth, depth_mask = utils3d.np.masked_nearest_resize(
                depth, mask=depth_mask, size=image.shape[:2]
            )
        if self.depth_unit is not None:
            depth *= self.depth_unit
            if sensor_depth is not None:
                sensor_depth *= self.depth_unit

        # Sensor depth
        if sensor_depth is not None:
            sensor_depth_mask = np.isfinite(sensor_depth)
            if sensor_depth.shape[:2] != image.shape[:2]:
                sensor_depth, sensor_depth_mask = utils3d.np.masked_nearest_resize(
                    sensor_depth, mask=sensor_depth_mask, size=image.shape[:2]
                )
        else:
            # Simulate
            assert 'sensor_sim' in  self.config, "sensor_sim config required for simulating sensor depth"

            sensor_sim_choices = list(self.config['sensor_sim'].keys())
            sensor_type = sensor_sim_choices[rng.choice(len(sensor_sim_choices))]
            sensor_params = self.config['sensor_sim'][sensor_type]
            # Simulate based on sensor type
            if sensor_type == "random_sparse":
                sensor_depth, sensor_depth_mask = sensor_sim.sample_random_sparse(
                    depth, 
                    np.isfinite(depth),
                    rng=rng,
                    **sensor_params
                )
            else:
                raise ValueError(f'Unknown sensor type: {sensor_type}')
            
        # drop depth greater than drop_max_depth
        max_depth = np.nanquantile(np.where(depth_mask, depth, np.nan), 0.01) * self.drop_max_depth
        depth_mask &= depth <= max_depth
        depth = np.where(depth_mask, depth, 1.0)

        instance.update({
            'image': torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1),
            'depth': torch.from_numpy(depth).float(),
            'depth_mask': torch.from_numpy(depth_mask).bool(),
            'sensor_depth': torch.from_numpy(sensor_depth).float(),
            'sensor_depth_mask': torch.from_numpy(sensor_depth_mask).bool(),
            'intrinsics': torch.from_numpy(intrinsics).float(),
            'segmentation_mask': torch.from_numpy(segmentation_mask).long() if segmentation_mask is not None else None,
            'segmentation_labels': segmentation_labels,
            'is_metric': self.depth_unit is not None,
            'has_sharp_boundary': self.has_sharp_boundary,
        })
        
        instance = {k: v for k, v in instance.items() if v is not None}
        
        return instance

    def start(self):
        self.pipeline.start()
    
    def stop(self):
        self.pipeline.stop()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def get(self):
        return self.pipeline.get()