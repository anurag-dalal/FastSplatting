import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import glob
import json

fders = ['baseline_default', 'baseline_30', 'baseline_mcmc', 'baseline_mcmc_ndl', 'baseline_mcmc_smf']
fders = ['baseline_40_v2']
BASE_DIRS = '/home/anurag/codes/MV3DGS_clearer_v2/results/baseline_default/'

COL_NAMES = ['scene', 'training_time','num_gaussians','memory', 'train_psnr', 'train_ssim', 'train_lpips', 'val_psnr', 'val_ssim', 'val_lpips', 'ATE', 'RTE']
df = pd.DataFrame(columns=COL_NAMES)

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for fder in fders:
    BASE_DIR = f'/home/anurag/codes/MV3DGS_clearer_v2/results/{fder}/'
    all_scenes = glob.glob(os.path.join(BASE_DIR, '*'))
    all_scenes = [scene for scene in all_scenes if os.path.isdir(scene)]
    for scene in all_scenes:
        train_rank0_file = os.path.join(scene, 'stats', 'train_step11999_rank0.json')
        train_metrics_file = os.path.join(scene, 'stats', 'train_step11999.json')
        val_metrics_file = os.path.join(scene, 'stats', 'val_step11999.json')
        rot_err_file = os.path.join(scene, 'stats', 'plot', 'stats_11999.json')
        ate_err_file = os.path.join(scene, 'stats', 'plot', 'trj_11999.json')

        with open(train_rank0_file, 'r') as f:
            train_rank0_stats = json.load(f)
        with open(train_metrics_file, 'r') as f:
            train_metrics = json.load(f)
        with open(val_metrics_file, 'r') as f:
            val_metrics = json.load(f)
        with open(rot_err_file, 'r') as f:
            rot_err_stats = json.load(f)
        with open(ate_err_file, 'r') as f:
            ate_err_stats = json.load(f)

        scene_name = os.path.basename(os.path.normpath(scene))
        training_time = train_rank0_stats['ellipse_time']
        num_gaussians = train_rank0_stats['num_GS']
        memory = train_rank0_stats['mem']

        train_psnr = train_metrics['psnr']
        val_psnr = val_metrics['psnr']
        train_ssim = train_metrics['ssim']
        val_ssim = val_metrics['ssim']
        train_lpips = train_metrics['lpips']
        val_lpips = val_metrics['lpips']

        rot_err = sum(rot_err_stats['rot_error_frame']) / len(rot_err_stats['rot_error_frame'])
        ate_err = rot_err_stats['rmse']

        new_row = pd.DataFrame([{'scene': scene_name,
                        'training_time': training_time,
                        'num_gaussians': num_gaussians,
                        'memory': memory,
                        'train_psnr': train_psnr,
                        'val_psnr': val_psnr,
                        'train_ssim': train_ssim,
                        'val_ssim': val_ssim,
                        'train_lpips': train_lpips,
                        'val_lpips': val_lpips,
                        'ATE': ate_err,
                        'RTE': rot_err
                    }])
        df.loc[len(df)] = new_row.iloc[0]
    # Add AVG row at the bottom
    numeric_cols = ['training_time', 'num_gaussians', 'memory', 'train_psnr', 'train_ssim', 'train_lpips', 'val_psnr', 'val_ssim', 'val_lpips', 'ATE', 'RTE']
    avg_row = df[numeric_cols].mean()
    avg_row['scene'] = 'AVG'
    df.loc[len(df)] = avg_row
    df.to_csv(os.path.join('/home/anurag/codes/MV3DGS_clearer_v2/paper_figures_codes', f'summary_{os.path.basename(os.path.normpath(BASE_DIR))}.csv'), index=False)

    # # Initialize EventAccumulator with the log directory
    # event_acc = EventAccumulator(os.path.join(scene, 'tb'))

    # # Load all events
    # event_acc.Reload()

    # # Access scalar events for a specific tag
    # tag_name = "train/loss"
    # if event_acc.Tags()["scalars"] and tag_name in event_acc.Tags()["scalars"]:
    #     scalars = event_acc.Scalars(tag_name)
    #     for s in scalars:
    #         print(f"Step: {s.step}, Wall Time: {s.wall_time}, Value: {s.value}")