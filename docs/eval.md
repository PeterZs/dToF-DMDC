# Evaluation

We provide a unified evaluation script that runs dToF-DMDC (or any wrapped baseline) on a
benchmark. It loads a dataset, evaluates on-the-fly, and reports metrics to a JSON file.

We ship the following reproducible benchmarks:

| Benchmark | Sparse depth source | Config |
| --- | --- | --- |
| **KITTI** | real sparse LiDAR shipped with the dataset (`sensor_depth.png`) | [`configs/eval/kitti.json`](../configs/eval/kitti.json) |
| **ETH3D** | fixed set of **500** points pre-sampled from the GT depth (`sensor_depth.png`) | [`configs/eval/eth3d.json`](../configs/eval/eth3d.json) |
| **ETH3D (random)** | **500** points randomly sampled from the GT depth on the fly | [`configs/eval/eth3d_rand.json`](../configs/eval/eth3d_rand.json) |

For ETH3D, use [`eth3d.json`](../configs/eval/eth3d.json) to **reproduce the paper's results**: it reads a
fixed set of pre-sampled sparse points shipped with the dataset. To instead randomly sample points
and run evaluation, use [`eth3d_rand.json`](../configs/eval/eth3d_rand.json), which simulates the
sparse depth on the fly via `sensor_sim`.

## Dataset preparation

The processed evaluation datasets are hosted alongside the model weights on the Hugging Face
Hub at [`summerbell/dToF-DMDC`](https://huggingface.co/summerbell/dToF-DMDC), shipped as two
archives: `kitti.zip` and `eth3d.zip`. Download and extract them under `data/eval/` so that
the layout matches the `path` field in each config (`data/eval/kitti`, `data/eval/eth3d`):

```bash
# requires: pip install "huggingface_hub[cli]"
huggingface-cli download summerbell/dToF-DMDC kitti.zip eth3d.zip --local-dir data/eval

cd data/eval
unzip -q kitti.zip && unzip -q eth3d.zip
rm kitti.zip eth3d.zip
cd -
```

Each dataset is organized as follows:

```
data/eval/<dataset>
├── .index.txt              # list of instance paths, one per line
├── <instance>
│   ├── image.jpg           # RGB image
│   ├── depth.png           # 16-bit GT depth (see dtof_dmdc/utils/io.py)
│   ├── sensor_depth.png    # sparse sensor depth (KITTI only; ETH3D is simulated)
│   └── meta.json           # stores "intrinsics" as a normalized 3x3 matrix
...
```

Depth maps are 16-bit PNGs in logarithmic scale; read/write them with `read_depth()` /
`write_depth()` in [`dtof_dmdc/utils/io.py`](../dtof_dmdc/utils/io.py). Invalid pixels are
encoded as `NaN`/`Inf`.

## Configuration

Each config is a JSON dict keyed by benchmark name. The keys map directly onto
[`EvalDataLoaderPipeline`](../dtof_dmdc/test/dataloader.py):

- `path` — dataset root.
- `split` — index file listing instances. Defaults to `.index.txt`.
- `depth` — GT depth filename. Defaults to `depth.png`.
- `sensor_depth` — real sparse depth filename. If omitted, sparse depth is simulated from
  the GT depth using `sensor_sim`.
- `sensor_sim` — sparse-depth simulation spec. `random_sparse` with `num_samples` uniformly
  samples that many valid GT pixels. Sampling is seeded per-sample (by sample index), so it
  is deterministic and reproducible run-to-run.
- `depth_unit` — scale factor to convert GT depth to meters. Setting it marks the benchmark
  as metric (`is_metric=True`), enabling the metric-depth metrics.

## Run Evaluation

Run [`dtof_dmdc/scripts/eval_baseline.py`](../dtof_dmdc/scripts/eval_baseline.py):

```bash
# KITTI — real sparse LiDAR
python dtof_dmdc/scripts/eval_baseline.py \
    --baseline baselines/dtof_dmdc.py \
    --config configs/eval/kitti.json \
    --output eval_output/dtof_dmdc/kitti.json \
    --pretrained summerbell/dToF-DMDC

# ETH3D — fixed 500 pre-sampled points (reproduces the paper)
python dtof_dmdc/scripts/eval_baseline.py \
    --baseline baselines/dtof_dmdc.py \
    --config configs/eval/eth3d.json \
    --output eval_output/dtof_dmdc/eth3d.json \
    --pretrained summerbell/dToF-DMDC

# ETH3D — 500 points randomly sampled on the fly
python dtof_dmdc/scripts/eval_baseline.py \
    --baseline baselines/dtof_dmdc.py \
    --config configs/eval/eth3d_rand.json \
    --output eval_output/dtof_dmdc/eth3d_rand.json \
    --pretrained summerbell/dToF-DMDC
```

`--baseline`, `--config`, `--output` are consumed by the evaluation script. Any remaining
arguments (e.g. `--pretrained`, `--variant`, `--fp16`) are forwarded to the baseline's
`load()` — see [`baselines/dtof_dmdc.py`](../baselines/dtof_dmdc.py).

By default the paper model (`vits`) is loaded from the HF hub. To evaluate a different
backbone, add `--variant {vits,vitb,vitl}`, which selects the corresponding checkpoint
(`dtof_dmdc_<variant>.pt`) from `summerbell/dToF-DMDC`:

```bash
# KITTI with the ViT-L backbone
python dtof_dmdc/scripts/eval_baseline.py \
    --baseline baselines/dtof_dmdc.py \
    --config configs/eval/kitti.json \
    --output eval_output/dtof_dmdc/kitti_vitl.json \
    --pretrained summerbell/dToF-DMDC --variant vitl
```

To evaluate a local checkpoint instead, pass its path to `--pretrained` (in which case
`--variant` is ignored), e.g. `--pretrained ./dtof_dmdc_vitl.pt`.

Options of the evaluation script:

```
Usage: eval_baseline.py [OPTIONS]

  Evaluation script.

Options:
  --baseline PATH  Path to the baseline model python code.
  --config PATH    Path to the evaluation configurations.
  --output PATH    Path to the output json file.
  --dump_pred      Dump prediction results.
  --dump_gt        Dump ground truth.
  --subset INTEGER Evaluate on a uniformly sampled subset of the given size.
  --help           Show this message and exit.
```

## Wrap a Customized Baseline

Wrap any depth-completion method with
[`dtof_dmdc.test.interface.MonoDepthRefineInterface`](../dtof_dmdc/test/interface.py) and
expose it as a class named `Baseline`. See [`baselines/dtof_dmdc.py`](../baselines/dtof_dmdc.py)
for a complete example.

To sanity-check a wrapper before a full run, evaluate on a small subset with `--subset`, e.g.:

```bash
python dtof_dmdc/scripts/eval_baseline.py \
    --baseline baselines/dtof_dmdc.py \
    --config configs/eval/eth3d.json \
    --output eval_output/dtof_dmdc/eth3d_subset.json \
    --subset 5 --pretrained summerbell/dToF-DMDC
```
