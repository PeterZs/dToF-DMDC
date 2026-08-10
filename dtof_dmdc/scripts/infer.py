import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
from typing import *
import itertools
import json
import warnings

import click


@click.command(help='dToF-DMDC inference: complete a dense metric depth map from an RGB image and a sparse depth map.')
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True), required=True, help='Input RGB image or a folder of images. "jpg" and "png" are supported.')
@click.option('--sensor_depth', '-d', 'sensor_depth_path', type=click.Path(exists=True), required=True, help='Sparse depth map, or a folder of them matched to the images by filename stem. '
    'Accepts a ".npy" array of shape (H, W) in meters (invalid pixels as NaN or <= 0), or a 16-bit ".png" in the dToF-DMDC format (see dtof_dmdc/utils/io.py).')
@click.option('--output', '-o', 'output_path', default='./output', type=click.Path(), help='Output folder path')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='summerbell/dToF-DMDC', help='Pretrained model name or path.')
@click.option('--variant', type=click.Choice(['vits', 'vitb', 'vitl']), default='vits', help='Backbone variant to download from the HF hub. Ignored when --pretrained is a local checkpoint path.')
@click.option('--fov_x', 'fov_x_', type=float, default=None, help='Camera horizontal field of view in degrees, e.g. KITTI ~= 80.4. dToF-DMDC predicts metric depth only, so '
    'the camera intrinsics are built from the FoV. Required together with --fov_y for the 3D exports (--glb, --ply, --show); with --maps alone it is optional, but the '
    'point map and fov.json are only written when it is given.')
@click.option('--fov_y', 'fov_y_', type=float, default=None, help='Camera vertical field of view in degrees, e.g. KITTI ~= 27.5. Required together with --fov_x for the 3D '
    'exports (--glb, --ply, --show); otherwise it is derived from --fov_x and the image aspect ratio assuming square pixels and a centered principal point.')
@click.option('--device', 'device_name', type=str, default='cuda', help='Device name (e.g. "cuda", "cuda:0", "cpu"). Defaults to "cuda"')
@click.option('--fp16', 'use_fp16', is_flag=True, help='Use fp16 precision for much faster inference.')
@click.option('--resolution_level', type=int, default=9, help='An integer [0-9] for the resolution level for inference. \
Higher value means more tokens and the finer details will be captured, but inference can be slower. Defaults to 9.')
@click.option('--num_image_tokens', type=int, default=None, help='Number of image tokens used for inference. Overrides `resolution_level` if provided. Default: None')
@click.option('--num_depth_tokens', type=int, default=None, help='Number of sparse-depth tokens used for inference. Overrides `resolution_level` if provided. Default: None')
@click.option('--threshold', type=float, default=0.04, help='Threshold for removing mesh edges. Smaller value removes more edges. "inf" means no thresholding.')
@click.option('--maps', 'save_maps_', is_flag=True, help='Whether to save the output maps (image, point map, depth map, normal map, mask) and fov.')
@click.option('--glb', 'save_glb_', is_flag=True, help='Whether to save the output as a .glb file. The color will be saved as a texture.')
@click.option('--ply', 'save_ply_', is_flag=True, help='Whether to save the output as a .ply file. The color will be saved as vertex colors.')
@click.option('--show', 'show', is_flag=True, help='Whether to show the output in a window. Note that this requires pyglet<2 installed as required by trimesh.')
def main(
    input_path: str,
    sensor_depth_path: str,
    fov_x_: float,
    fov_y_: float,
    output_path: str,
    pretrained_model_name_or_path: str,
    variant: str,
    device_name: str,
    use_fp16: bool,
    resolution_level: int,
    num_image_tokens: int,
    num_depth_tokens: int,
    threshold: float,
    save_maps_: bool,
    save_glb_: bool,
    save_ply_: bool,
    show: bool,
):
    # Resolve the requested outputs first, so the FoV requirement below is checked before anything heavy is loaded.
    if not any([save_maps_, save_glb_, save_ply_]):
        warnings.warn('No output format specified. Defaults to saving all. Please use "--maps", "--glb", or "--ply" to specify the output.')
        save_maps_ = save_glb_ = save_ply_ = True

    # The 3D exports need the full camera geometry, which dToF-DMDC cannot recover from the metric depth alone.
    if save_glb_ or save_ply_ or show:
        missing = [name for name, value in (('--fov_x', fov_x_), ('--fov_y', fov_y_)) if value is None]
        if missing:
            raise click.UsageError(f'{" and ".join(missing)} {"is" if len(missing) == 1 else "are"} required when exporting 3D outputs (--glb, --ply, --show).')

    import math

    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm
    import trimesh
    import trimesh.visual

    from dtof_dmdc.model.dmdc import MoGeModel
    from dtof_dmdc.utils.io import save_glb, save_ply, read_depth
    from dtof_dmdc.utils.vis import colorize_depth, colorize_normal
    import utils3d

    device = torch.device(device_name)

    # Collect input images
    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).rglob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        image_paths = [Path(input_path)]
    if len(image_paths) == 0:
        raise FileNotFoundError(f'No image files found in {input_path}')

    def fov_to_intrinsics(fov_x_deg: float, fov_y_deg: Optional[float], width: int, height: int) -> np.ndarray:
        # Normalized pinhole intrinsics (focal in units of image size, centered principal point).
        fx = 0.5 / math.tan(math.radians(fov_x_deg) / 2)
        if fov_y_deg is not None:
            fy = 0.5 / math.tan(math.radians(fov_y_deg) / 2)
        else:
            # Derive fov_y from fov_x assuming square pixels: fy_px == fx_px.
            fy = fx * width / height
        return np.array([[fx, 0.0, 0.5], [0.0, fy, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)

    def load_sparse_depth(path: Path) -> np.ndarray:
        # Metric depth (H, W) in meters; invalid pixels marked NaN.
        if path.suffix.lower() == '.npy':
            depth = np.load(path).astype(np.float32).squeeze()
            depth[~(np.isfinite(depth) & (depth > 0))] = np.nan
            return depth
        return read_depth(path)  # 16-bit PNG in the dToF-DMDC format

    # Resolve the sparse depth map for each image
    sensor_depth_path = Path(sensor_depth_path)
    if sensor_depth_path.is_dir():
        depth_lookup = {p.stem: p for p in itertools.chain(sensor_depth_path.rglob('*.npy'), sensor_depth_path.rglob('*.png'))}
        def resolve_sensor_depth(image_path: Path) -> Optional[Path]:
            return depth_lookup.get(image_path.stem)
    else:
        def resolve_sensor_depth(image_path: Path) -> Optional[Path]:
            return sensor_depth_path

    model = MoGeModel.from_pretrained(pretrained_model_name_or_path, filename=f'dtof_dmdc_{variant}.pt').to(device).eval()
    if use_fp16:
        model.half()

    for image_path in (pbar := tqdm(image_paths, desc='Inference', disable=len(image_paths) <= 1)):
        depth_path = resolve_sensor_depth(image_path)
        if depth_path is None:
            warnings.warn(f'No matching sparse depth found for {image_path.name}, skipping.')
            continue

        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Sparse depth in meters; invalid pixels are NaN.
        sparse_depth = load_sparse_depth(depth_path)
        if sparse_depth.shape[:2] != (height, width):
            raise ValueError(f'Sparse depth size {sparse_depth.shape[:2]} does not match image size {(height, width)} for {image_path.name}.')
        sparse_mask = np.isfinite(sparse_depth)

        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        sparse_depth_tensor = torch.tensor(np.nan_to_num(sparse_depth), dtype=torch.float32, device=device)
        sparse_mask_tensor = torch.tensor(sparse_mask, device=device)

        # Inference
        output = model.infer(
            image_tensor, sparse_depth_tensor, sparse_mask_tensor,
            num_image_tokens=num_image_tokens, num_depth_tokens=num_depth_tokens,
            resolution_level=resolution_level, fov_x=fov_x_, use_fp16=use_fp16,
        )
        mask = output['mask'].cpu().numpy()
        normal = output['normal'].cpu().numpy() if 'normal' in output else None
        if 'points' in output:
            # Point-map model (e.g. MoGe): depth & intrinsics are recovered from the point map.
            points = output['points'].cpu().numpy()
            depth = output['depth'].cpu().numpy()
            intrinsics = output['intrinsics'].cpu().numpy()
        else:
            # dToF-DMDC predicts metric depth directly; build camera geometry from the given FoV.
            # Without a FoV there is no camera to unproject with, so only the 2D maps are available.
            depth = output['metric_depth'].cpu().numpy()
            intrinsics = fov_to_intrinsics(fov_x_, fov_y_, width, height) if fov_x_ is not None else None
            points = utils3d.numpy.depth_map_to_point_map(depth, intrinsics=intrinsics) if intrinsics is not None else None

        save_path = Path(output_path, image_path.relative_to(input_path).parent if Path(input_path).is_dir() else '', image_path.stem)
        save_path.mkdir(exist_ok=True, parents=True)

        # Save images / maps
        if save_maps_:
            cv2.imwrite(str(save_path / 'image.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_path / 'depth_vis.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_path / 'depth.exr'), depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            cv2.imwrite(str(save_path / 'mask.png'), (mask * 255).astype(np.uint8))
            if normal is not None:
                cv2.imwrite(str(save_path / 'normal.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))
            if intrinsics is not None:
                cv2.imwrite(str(save_path / 'points.exr'), cv2.cvtColor(points, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
                with open(save_path / 'fov.json', 'w') as f:
                    json.dump({
                        'fov_x': round(float(np.rad2deg(fov_x)), 2),
                        'fov_y': round(float(np.rad2deg(fov_y)), 2),
                    }, f)

        # Export mesh & visualization
        if save_glb_ or save_ply_ or show:
            mask_cleaned = mask & ~utils3d.numpy.depth_map_edge(depth, rtol=threshold)
            if normal is None:
                faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.build_mesh_from_map(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d.numpy.uv_map(height, width),
                    mask=mask_cleaned,
                    tri=True
                )
                vertex_normals = None
            else:
                faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.build_mesh_from_map(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d.numpy.uv_map(height, width),
                    normal,
                    mask=mask_cleaned,
                    tri=True
                )
            # When exporting the model, follow the OpenGL coordinate conventions:
            # - world coordinate system: x right, y up, z backward.
            # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
            vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
            if normal is not None:
                vertex_normals = vertex_normals * [1, -1, -1]

        if save_glb_:
            save_glb(save_path / 'mesh.glb', vertices, faces, vertex_uvs, image, vertex_normals)

        if save_ply_:
            save_ply(save_path / 'pointcloud.ply', vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, vertex_normals)

        if show:
            trimesh.Trimesh(
                vertices=vertices,
                vertex_colors=vertex_colors,
                vertex_normals=vertex_normals,
                faces=faces,
                process=False
            ).show()


if __name__ == '__main__':
    main()
