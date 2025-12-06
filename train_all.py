"""
Batch training/evaluation runner without CLI arguments.

- Provide lists of data_dirs and result_dirs of the same length.
- For each pair, it runs the training loop from training.py (sequentially).
- After each run, it aggregates test metrics (PSNR, SSIM) from JSON stats
  for each eval step and computes an L1 score from saved renders vs input images.
- Writes a CSV summarizing metrics across all datasets and eval iterations.

Note:
- L1 is computed post-hoc using saved rendered images and source images resized to
  the render resolution. This approximates the training L1 and provides a consistent
  measure for comparison across eval steps. Since renders overwrite at each eval,
  we report the final render-derived L1 for all eval step rows of a dataset.
"""
# python training.py default --data_dir /mnt/c/MyFiles/Datasets/robustnerf/yoda/ --data_factor 1 --result_dir /home/anurag/codes/MV3DGS_clearer_v2/results/baseline_exps_v2/yoda --pose_opt_type mlp --no-use_corres_epipolar_loss
import csv
import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

import training as tr
from gsplat.distributed import cli
from gsplat.strategy import MCMCStrategy


# -----------------------
# User-editable lists
# -----------------------
# Example entries — edit these to your scene folders and desired output dirs.

import os
import glob

DATASET_DIRS = ['/mnt/c/MyFiles/Datasets/3rgs', '/mnt/c/MyFiles/Datasets/tandt_db/db', '/mnt/c/MyFiles/Datasets/tandt_db/tandt', '/mnt/c/MyFiles/Datasets/robustnerf']
# DATASET_DIRS = ['/home/anurag/codes/vggt/examples/']
DATA_DIRS = []
RESULT_DIRS = []

result_dir_base = '/home/anurag/codes/MV3DGS_clearer_v2/results/baseline_40_v2/'

for dataset in DATASET_DIRS:
    scene_dirs = glob.glob(os.path.join(dataset, '*'))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]
    for scene_dir in scene_dirs:
        if 'videos' not in scene_dir and 'single' not in scene_dir and 'kitchen_vggt_x' not in scene_dir:
            scene_name = os.path.basename(os.path.normpath(scene_dir))
            result_dir = os.path.join(result_dir_base, scene_name)
            DATA_DIRS.append(scene_dir)
            RESULT_DIRS.append(result_dir)

# Optionally override some training config knobs globally here
POSE_OPT_TYPE = "mlp"           # or "sfm"
USE_CORRES_EPIPOLAR_LOSS = False
DATA_FACTOR = 1                  # Use 1 for no downscale
TEST_EVERY_FOR_VAL = 10          # Must match Runner's Parser test_every usage
DISABLE_VIEWER = True            # Avoid launching the viewer in batch runs
USE_MCMC_STRATEGY = True        # Use MCMC strategy instead of default densification

def _list_images(image_dir: str) -> List[str]:
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]
    files = sorted(
        [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if os.path.splitext(f)[1] in exts
        ]
    )
    return files


def _compute_l1_from_renders(data_dir: str, result_dir: str, test_every: int) -> float:
    """Compute average L1 between saved renders and source images (resized).

    This uses the latest renders in result_dir/renders and the source images in
    data_dir/images. It selects validation images by i % test_every == 0 to match
    the evaluation subset used by training.
    """
    render_dir = os.path.join(result_dir, "renders")
    if not os.path.isdir(render_dir):
        return float("nan")

    # Collect render files rendered_0000.png, rendered_0001.png, ...
    render_files = sorted(
        [f for f in os.listdir(render_dir) if f.startswith("rendered_") and f.endswith(".png")]
    )
    if not render_files:
        return float("nan")

    # Determine validation image list deterministically
    src_images = _list_images(os.path.join(data_dir, "images"))
    # Map to val subset order
    val_indices = [i for i in range(len(src_images)) if (i % test_every) == 0]
    val_images = [src_images[i] for i in val_indices]

    # Number of renders should match number of val images; if not, use min length
    n = min(len(render_files), len(val_images))
    if n == 0:
        return float("nan")

    l1_vals: List[float] = []
    for i in range(n):
        render_path = os.path.join(render_dir, render_files[i])
        gt_path = val_images[i]

        # Load render (uint8 RGB)
        try:
            render_img = Image.open(render_path).convert("RGB")
        except Exception:
            continue
        W, H = render_img.size

        # Load GT and resize to render size
        try:
            gt_img = Image.open(gt_path).convert("RGB").resize((W, H), Image.Resampling.BICUBIC)
        except Exception:
            continue

        render_np = np.asarray(render_img).astype(np.float32) / 255.0
        gt_np = np.asarray(gt_img).astype(np.float32) / 255.0
        # Mean absolute error over HxWxC
        l1 = np.mean(np.abs(render_np - gt_np))
        l1_vals.append(float(l1))

    if not l1_vals:
        return float("nan")
    return float(np.mean(l1_vals))


def _collect_eval_stats(result_dir: str, eval_steps: List[int]):
    """Read PSNR/SSIM/LPIPS from JSON stats for each eval step.

    training.py writes stats at steps (eval_step - 1). We therefore read files
    results/<exp>/stats/val_step{step:04d}.json for step in [e-1 for e in eval_steps].
    Returns list of dicts with keys: step, psnr, ssim, lpips, num_GS (if present), ellipse_time (if present).
    """
    stats_dir = os.path.join(result_dir, "stats")
    out: List[dict] = []
    for e in eval_steps:
        step = e - 1  # training triggers eval at step-1
        fname = os.path.join(stats_dir, f"val_step{step:04d}.json")
        if not os.path.isfile(fname):
            continue
        try:
            with open(fname, "r") as f:
                data = json.load(f)
            out.append({
                "step": step,
                "psnr": float(data.get("psnr", float("nan"))),
                "ssim": float(data.get("ssim", float("nan"))),
                "lpips": float(data.get("lpips", float("nan"))),
                # Sometimes eval json includes runtime info / num_GS
                "ellipse_time": float(data.get("ellipse_time", float("nan"))),
                "num_GS": int(data.get("num_GS", data.get("num_gs", -1))) if data.get("num_GS", data.get("num_gs", None)) is not None else -1,
            })
        except Exception:
            continue
    return out


def _collect_train_stats(result_dir: str, step: int):
    """Read memory (GB), training elapsed time (s), and num_GS from train_step json.

    Files tried: train_step{step:04d}.json, then train_step{step:04d}_rank0.json
    """
    stats_dir = os.path.join(result_dir, "stats")
    base = os.path.join(stats_dir, f"train_step{step:04d}.json")
    alt = os.path.join(stats_dir, f"train_step{step:04d}_rank0.json")
    path = base if os.path.isfile(base) else alt if os.path.isfile(alt) else None
    if path is None:
        return {
            "mem": float("nan"),
            "ellipse_time": float("nan"),
            "num_GS": -1,
        }
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {
            "mem": float(data.get("mem", float("nan"))),
            "ellipse_time": float(data.get("ellipse_time", float("nan"))),
            "num_GS": int(data.get("num_GS", data.get("num_gs", -1))) if data.get("num_GS", data.get("num_gs", None)) is not None else -1,
        }
    except Exception:
        return {
            "mem": float("nan"),
            "ellipse_time": float("nan"),
            "num_GS": -1,
        }


def run_all():
    if len(DATA_DIRS) != len(RESULT_DIRS):
        raise ValueError(f"DATA_DIRS and RESULT_DIRS must have same length (got {len(DATA_DIRS)} vs {len(RESULT_DIRS)})")

    # CSV header:
    # input_dir,result_dir,num_images,step,psnr,ssim,lpips,l1,train_time_s,num_gs,mem_gb
    csv_rows: List[List[str]] = [[
        "input_dir",
        "result_dir",
        "num_images",
        "step",
        "psnr",
        "ssim",
        "lpips",
        "l1",
        "train_time_s",
        "num_gs",
        "mem_gb",
    ]]

    for data_dir, result_dir in zip(DATA_DIRS, RESULT_DIRS):
        # Prepare config (similar to training.py CLI config + Runner overrides)
        cfg = tr.Config()
        cfg.data_dir = data_dir
        cfg.data_factor = DATA_FACTOR
        cfg.result_dir = result_dir
        cfg.pose_opt_type = POSE_OPT_TYPE  # "mlp" or "sfm"
        cfg.use_corres_epipolar_loss = USE_CORRES_EPIPOLAR_LOSS
        cfg.disable_viewer = DISABLE_VIEWER
        # Use MCMC strategy instead of default densification
        if USE_MCMC_STRATEGY:
            cfg.strategy = MCMCStrategy(verbose=True)
            # Adopt the specialized hyperparameters used for MCMC in training.py
            cfg.init_opa = 0.5
            cfg.init_scale = 0.1
        cfg.opacity_reg = 0.01
        cfg.scale_reg = 0.01
        # Keep eval_steps/max_steps as in Config; Runner uses Parser(test_every=10)

        # Ensure result directory exists (avoid race issues later)
        os.makedirs(result_dir, exist_ok=True)

        # Run training/eval sequentially
        cli(tr.main, cfg, verbose=True)

        # Collect metrics
        eval_list = _collect_eval_stats(result_dir, cfg.eval_steps)
        # Compute L1 once from the final renders
        l1_val = _compute_l1_from_renders(data_dir, result_dir, TEST_EVERY_FOR_VAL)
        # Count all images in dataset
        num_images = len(_list_images(os.path.join(data_dir, "images")))

        # Append rows
        for ev in eval_list:
            step = ev["step"]
            psnr = ev["psnr"]
            ssim = ev["ssim"]
            lpips = ev.get("lpips", float("nan"))
            # Training stats (mem/time/num_GS) from train_step json at same step
            tr_stats = _collect_train_stats(result_dir, step)
            mem_gb = tr_stats["mem"]
            time_s = tr_stats["ellipse_time"]
            num_gs = tr_stats["num_GS"] if tr_stats["num_GS"] != -1 else ev.get("num_GS", -1)
            csv_rows.append([
                data_dir,
                result_dir,
                str(num_images),
                str(step),
                f"{psnr:.6f}" if not np.isnan(psnr) else "",
                f"{ssim:.6f}" if not np.isnan(ssim) else "",
                f"{lpips:.6f}" if not np.isnan(lpips) else "",
                f"{l1_val:.6f}" if not np.isnan(l1_val) else "",
                f"{time_s:.3f}" if not np.isnan(time_s) else "",
                str(num_gs) if isinstance(num_gs, int) and num_gs >= 0 else "",
                f"{mem_gb:.3f}" if not np.isnan(mem_gb) else "",
            ])

    # Write CSV at repo root
    out_csv = os.path.join(result_dir_base, "train_all_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Wrote metrics CSV to: {out_csv}")


if __name__ == "__main__":
    run_all()
