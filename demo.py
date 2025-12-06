import os
import glob

DATASET_DIRS = ['/mnt/c/MyFiles/Datasets/3rgs', '/mnt/c/MyFiles/Datasets/tandt_db/db', '/mnt/c/MyFiles/Datasets/tandt_db/tandt', '/mnt/c/MyFiles/Datasets/robustnerf']
# DATASET_DIRS = ['/home/anurag/codes/vggt/examples/']
DATA_DIRS = []
RESULT_DIRS = []

result_dir_base = '/home/anurag/codes/MV3DGS_clearer_v2/results/baseline_exps_v2/'

for dataset in DATASET_DIRS:
    scene_dirs = glob.glob(os.path.join(dataset, '*'))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]
    for scene_dir in scene_dirs:
        if 'videos' not in scene_dir and 'single' not in scene_dir and 'kitchen_vggt_x' not in scene_dir:
            scene_name = os.path.basename(os.path.normpath(scene_dir))
            result_dir = os.path.join(result_dir_base, scene_name)
            DATA_DIRS.append(scene_dir)
            RESULT_DIRS.append(result_dir)
DATA_DIRS = DATA_DIRS[-2:]  # robustnerf only
RESULT_DIRS = RESULT_DIRS[-2:]  # robustnerf only
for i, da in enumerate(DATA_DIRS):
    print(f'{DATA_DIRS[i]}')  