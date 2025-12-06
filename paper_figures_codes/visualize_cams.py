# import os
# import glob

# DATASET_DIRS = ['/home/anurag/codes/vggt/examples/', '/mnt/c/MyFiles/Datasets/3rgs', '/mnt/c/MyFiles/Datasets/tandt_db/db', '/mnt/c/MyFiles/Datasets/tandt_db/tandt']

# DATA_DIRS = []
# RESULT_DIRS = []

# result_dir_base = '/home/anurag/codes/MV3DGS_clearer_v2/results/baseline_exps/'

# for dataset in DATASET_DIRS:
#     scene_dirs = glob.glob(os.path.join(dataset, '*'))
#     scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]
#     for scene_dir in scene_dirs:
#         if 'videos' not in scene_dir and 'results' not in scene_dir:
#             scene_name = os.path.basename(os.path.normpath(scene_dir))
#             result_dir = os.path.join(result_dir_base, scene_name)
#             DATA_DIRS.append(scene_dir)
#             RESULT_DIRS.append(result_dir)
# for i in range(len(DATA_DIRS)):
#     print(f'{DATA_DIRS[i]}, {RESULT_DIRS[i]}')

# import pycolmap
# import os
# import glob
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# data_dir = '/mnt/c/MyFiles/Datasets/3rgs/bicycle'
# colmap_dir = os.path.join(data_dir, "sparse/0/")

# reconstruction = pycolmap.Reconstruction(colmap_dir)
# w2c_mats = []
# bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
# for image_id, image in reconstruction.images.items():
#     rot = image.cam_from_world.todict()['rotation']['quat']
#     rot = R.from_quat(rot)
#     rot = rot.as_matrix()
#     trans = image.cam_from_world.todict()['translation']
#     trans = trans.reshape(3, 1)
#     w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
#     w2c_mats.append({image.name: w2c})
# print(w2c_mats)
import numpy as np
import matplotlib.pyplot as plt

# File names (Ensure these files are in the same directory as the script)
file_est_npy = '/home/anurag/codes/MV3DGS_clearer_v2/results/baseline_40/kitchen/camtoworlds_est.npy'
file_gt_npy = '/home/anurag/codes/MV3DGS_clearer_v2/results/baseline_40/kitchen/camtoworlds_gt.npy'

# --- 1. Load Data ---
# Load the data directly from .npy files.
try:
    poses_est = np.load(file_est_npy)
    poses_gt = np.load(file_gt_npy)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the .npy files are in the current working directory.")
    exit()
print(poses_est - poses_gt)
# --- 2. Extract Camera Centers ---
# The camera center (translation vector t) is the first 3 elements of the 4th column (index 3)
# in the 4x4 matrix.
orientation_est = poses_est[:, :3, :3]  # (N, 3, 3) array of rotation matrices for estimated poses
orientation_gt = poses_gt[:, :3, :3]    # (N, 3, 3) array of rotation matrices for ground truth poses
centers_est = poses_est[:, :3, 3] # (N, 3) array of (tx, ty, tz) for estimated poses
centers_gt = poses_gt[:, :3, 3]   # (N, 3) array of (tx, ty, tz) for ground truth poses


# --- 3. Create the 3D Plot ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot estimated poses (scatter and line for trajectory)
ax.scatter(centers_est[:, 0], centers_est[:, 1], centers_est[:, 2], c='r', marker='o', label=r'Estimated Poses $\mathbf{T}_{w \leftarrow c}$', alpha=0.6)

# Plot ground truth poses (scatter and line for trajectory)
ax.scatter(centers_gt[:, 0], centers_gt[:, 1], centers_gt[:, 2], c='b', marker='^', label=r'Ground Truth Poses $\mathbf{T}_{w \leftarrow c}$', alpha=0.6)

# --- 3b. Draw orientation arrows ---
# We treat the 3x3 block as the rotation from camera to world (c->w). To visualize the viewing direction
# we project the camera forward axis. Depending on convention this is usually the -Z axis in OpenCV.
# If arrows appear reversed, switch forward_axis between [0,0,-1] and [0,0,1].
forward_axis = np.array([0.0, 0.0, 1.0])  # assumed camera looking along -Z

# Sample poses to avoid clutter if there are many.
num_poses = centers_est.shape[0]
max_arrows = 80
step = max(1, num_poses // max_arrows)
sample_idx = np.arange(0, num_poses, step)

# Compute direction vectors for sampled estimated poses
dirs_est = (orientation_est[sample_idx] @ forward_axis.reshape(3, 1)).reshape(-1, 3)
dirs_gt = (orientation_gt[sample_idx] @ forward_axis.reshape(3, 1)).reshape(-1, 3)

# Arrow length scaled to scene extent
extent_min = np.minimum(centers_est.min(axis=0), centers_gt.min(axis=0))
extent_max = np.maximum(centers_est.max(axis=0), centers_gt.max(axis=0))
scene_scale = np.linalg.norm(extent_max - extent_min)
arrow_length = scene_scale * 0.04  # 4% of scene diagonal

# Normalize direction vectors then scale
def normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n
dirs_est_n = normalize(dirs_est) * arrow_length
dirs_gt_n = normalize(dirs_gt) * arrow_length

centers_est_s = centers_est[sample_idx]
centers_gt_s = centers_gt[sample_idx]

# Plot quiver arrows
ax.quiver(centers_est_s[:, 0], centers_est_s[:, 1], centers_est_s[:, 2],
          dirs_est_n[:, 0], dirs_est_n[:, 1], dirs_est_n[:, 2],
          color='darkred', length=1.0, normalize=False, linewidth=0.8, arrow_length_ratio=0.25)
ax.quiver(centers_gt_s[:, 0], centers_gt_s[:, 1], centers_gt_s[:, 2],
          dirs_gt_n[:, 0], dirs_gt_n[:, 1], dirs_gt_n[:, 2],
          color='navy', length=1.0, normalize=False, linewidth=0.8, arrow_length_ratio=0.25)

# Optional: annotate first pose for reference
if num_poses > 0:
    ax.text(centers_est[0, 0], centers_est[0, 1], centers_est[0, 2], 'Est[0]', color='red')
    ax.text(centers_gt[0, 0], centers_gt[0, 1], centers_gt[0, 2], 'GT[0]', color='blue')

# --- 4. Add Labels and Save ---
ax.set_xlabel('X (World)')
ax.set_ylabel('Y (World)')
ax.set_zlabel('Z (World)')
ax.set_title('3D Camera Trajectories (Estimated vs. Ground Truth)')
ax.legend()
ax.grid(True)

plot_filename = 'camera_poses_3d_npy.png'
# fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()

print(f"Plot saved as {plot_filename}")
print(f"Number of Estimated Poses: {centers_est.shape[0]}")
print(f"Number of Ground Truth Poses: {centers_gt.shape[0]}")