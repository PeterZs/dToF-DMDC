import os
import sys
from pathlib import Path
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
from typing import *
import importlib
import importlib.util

import click
from dtof_dmdc.utils.io import _pack_depth_input, write_depth

@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, help='Evaluation script.')
@click.option('--baseline', 'baseline_code_path', type=click.Path(), required=True, help='Path to the baseline model python code.')
@click.option('--config', 'config_path', type=click.Path(), default='configs/eval/all_benchmarks.json', help='Path to the evaluation configurations. '
    'Defaults to "configs/eval/all_benchmarks.json".')
@click.option('--output', '-o', 'output_path',  type=click.Path(), required=True, help='Path to the output json file.')
@click.option('--oracle', 'oracle_mode', is_flag=True, help='Use oracle mode for evaluation, i.e., use the GT intrinsics input.')
@click.option('--dump_pred', is_flag=True, help='Dump predition results.')
@click.option('--dump_gt', is_flag=True, help='Dump ground truth.')
@click.option('--subset', 'subset_size', type=int, default=None, help='Take a subset of each dataset for given the size.')
@click.pass_context
def main(ctx: click.Context, baseline_code_path: str, config_path: str, oracle_mode: bool, output_path: Union[str, Path], dump_pred: bool, dump_gt: bool, subset_size: int):
    # Lazy import
    import  cv2
    import numpy as np
    from tqdm import tqdm
    import torch
    import torch.nn.functional as F
    import utils3d

    from dtof_dmdc.test.interface import MonoDepthRefineInterface
    from dtof_dmdc.test.dataloader import EvalDataLoaderPipeline
    from dtof_dmdc.test.metrics import compute_metrics
    from dtof_dmdc.utils.geometry_torch import intrinsics_to_fov
    from dtof_dmdc.utils.vis import colorize_depth, colorize_normal, colorize_error
    from dtof_dmdc.utils.tools import key_average, flatten_nested_dict, timeit, import_file_as_module
    from dtof_dmdc.utils.geometry_numpy import mask_aware_nearest_resize_numpy
    from dtof_dmdc.utils.io import read_depth, write_depth
    
    # Load the baseline model
    module = import_file_as_module(baseline_code_path, Path(baseline_code_path).stem)
    baseline_cls: Type[MonoDepthRefineInterface] = getattr(module, 'Baseline')
    baseline: MonoDepthRefineInterface = baseline_cls.load.main(ctx.args, standalone_mode=False)
    

    # Warmup
    for _ in range(3):
        baseline.infer_for_evaluation(torch.rand(3, 224, 224, device=baseline.device), torch.rand(224, 224, device=baseline.device), torch.rand(224, 224, device=baseline.device) > 0.5)

    # Load the evaluation configurations
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Override configurations
    if subset_size is not None:
        for benchmark_name in config:
            config[benchmark_name]['subset_size'] = subset_size
    if '__shared__' in config:
        shared_config = config['__shared__']
        for benchmark_name in config:
            if benchmark_name == '__shared__':
                continue
            config[benchmark_name] = {
                **shared_config,
                **config[benchmark_name]
            }
        del config['__shared__']
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    # Iterate over the dataset

    for benchmark_name, benchmark_config in tqdm(list(config.items()), desc='Benchmarks'):
        filenames, metrics_list = [], []
        with (
            EvalDataLoaderPipeline(benchmark_config) as eval_data_pipe,
            tqdm(total=len(eval_data_pipe), desc=benchmark_name, leave=False) as pbar
        ):  
            # Iterate over the samples in the dataset
            if dump_gt or dump_pred:
                dump_filename = []

            for i in range(len(eval_data_pipe)):
                sample = eval_data_pipe.get()

                if dump_gt or dump_pred:
                    dump_filename.append(sample['filename'])

                sample = {k: v.to(baseline.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                image = sample['image']
                gt_intrinsics = sample['intrinsics']

                sensor_depth = sample['sensor_depth']
                sensor_depth_mask = sample['sensor_depth_mask']
                
                # Inference
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode(), timeit('_inference_timer', verbose=False) as timer:
                    if oracle_mode:
                        pred = baseline.infer_for_evaluation(image, sensor_depth, sensor_depth_mask, gt_intrinsics)
                    else:
                        pred = baseline.infer_for_evaluation(image, sensor_depth, sensor_depth_mask)
                    torch.cuda.synchronize()
                memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # in GB

                # Compute metrics
                metrics, misc = compute_metrics(pred, sample, vis=dump_pred or dump_gt)
                metrics['inference_time'] = timer.time
                metrics['memory_usage'] = memory_used
                metrics_list.append(metrics)

                # Dump results
                dump_path = Path(output_path.replace(".json", f"_dump"), f'{benchmark_name}', sample['filename'].replace('.zip', ''))
                
                if dump_pred:
                    dump_path.joinpath('pred').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dump_path / 'pred' / 'image.jpg'), cv2.cvtColor((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    with Path(dump_path, 'pred', 'metrics.json').open('w') as f:
                        json.dump(metrics, f, indent=4)

                    if 'pred_depth' in misc:
                        depth = misc['pred_depth'].cpu().numpy()
                        if 'mask' in pred:
                            mask = pred['mask'].cpu().numpy()
                            depth = np.where(mask, depth, np.inf)

                        cv2.imwrite(str(dump_path / 'pred' / 'depth_vis.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
                        write_depth((dump_path / 'pred' / 'depth.png'), depth)
                        
                    if 'mask' in pred:
                        mask = pred['mask'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'mask.png'), mask.astype(np.uint8) * 255)

                    if 'normal' in pred:
                        normal = pred['normal'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'normal.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))

                    if 'intrinsics' in sample:
                        intrinsics = sample['intrinsics']
                        fov_x, fov_y = intrinsics_to_fov(intrinsics)
                        with open(dump_path / 'pred' / 'fov.json', 'w') as f:
                            json.dump({
                                'fov_x': np.rad2deg(fov_x.item()),
                                'fov_y': np.rad2deg(fov_y.item()),
                                'intrinsics': intrinsics.cpu().numpy().tolist(),
                            }, f)
                    
                    if 'depth_14' in pred:
                        depth = pred['depth_14'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'depth_14.exr'), depth.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

                    if 'depth_14_masked_nearest' in pred:
                        depth = pred['depth_14_masked_nearest'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'depth_14_masked_nearest.exr'), depth.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                        
                
                if dump_gt:
                    dump_path.joinpath('gt').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dump_path / 'gt' / 'image.jpg'), cv2.cvtColor((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    if 'sensor_depth' in sample:
                        sensor_depth = sample['sensor_depth'].cpu().numpy().astype(np.float32)
                        sensor_depth_mask = sample['sensor_depth_mask'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'gt' / 'sensor_depth_vis.png'), cv2.cvtColor(colorize_depth(sensor_depth, mask=sensor_depth_mask), cv2.COLOR_RGB2BGR))

                        write_depth(dump_path / 'gt' / 'sensor_depth.png', sensor_depth)
                        cv2.imwrite(str(dump_path / 'gt' / 'sensor_depth_mask.png'), (sensor_depth_mask * 255).astype(np.uint8))

                    if 'depth' in sample:
                        depth = sample['depth']
                        mask = sample['depth_mask']
                        cv2.imwrite(str(dump_path / 'gt' / 'depth_vis.png'), cv2.cvtColor(colorize_depth(depth.cpu().numpy(), mask=mask.cpu().numpy()), cv2.COLOR_RGB2BGR))

                        depth = np.where(mask.cpu().numpy(), depth.cpu().numpy(), np.nan)
                        write_depth(str(dump_path / 'gt' / 'depth.png'), depth)
                        
                    if 'normal' in sample:
                        normal = sample['normal']
                        cv2.imwrite(str(dump_path / 'gt' / 'normal_vis.png'), cv2.cvtColor(colorize_normal(normal.cpu().numpy()), cv2.COLOR_RGB2BGR))

                    if 'depth_mask' in sample:
                        mask = sample['depth_mask']
                        cv2.imwrite(str(dump_path / 'gt' /'mask.png'), (mask.cpu().numpy() * 255).astype(np.uint8))
                        
                    if 'intrinsics' in sample:
                        intrinsics = sample['intrinsics']
                        fov_x, fov_y = intrinsics_to_fov(intrinsics)
                        with open(dump_path / 'gt' / 'fov.json', 'w') as f:
                            json.dump({
                                'fov_x': np.rad2deg(fov_x.item()),
                                'fov_y': np.rad2deg(fov_y.item()),
                                'intrinsics': intrinsics.cpu().numpy().tolist(),
                            }, f)

                # Save intermediate results
                if i % 100 == 0 or i == len(eval_data_pipe) - 1:
                    if not (dump_pred or dump_gt):
                        Path(output_path).write_text(
                            json.dumps({
                                **all_metrics, 
                                benchmark_name: key_average(metrics_list)
                            }, indent=4)
                        )
                pbar.update(1)

            all_metrics[benchmark_name] = key_average(metrics_list)

            if dump_gt or dump_pred:
                dump_path = Path(output_path.replace(".json", f"_dump"), f'{benchmark_name}', "dump_index.txt")
                dump_path.write_text("\n".join(dump_filename))

    # Save final result:
    all_metrics['mean'] = key_average(list(all_metrics.values()))
    Path(output_path).write_text(json.dumps(all_metrics, indent=4))


if __name__ == '__main__':
    main()
