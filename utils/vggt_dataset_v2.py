import os
import glob
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import imageio.v2 as imageio
import cv2
import pycolmap
from scipy.spatial.transform import Rotation as R
# VGGT imports
from vggt_n.models.vggt import VGGT
from vggt_n.utils.load_fn import load_and_preprocess_images_ratio
from vggt_n.utils.pose_enc import pose_encoding_to_extri_intri
from vggt_n.utils.geometry import unproject_depth_map_to_point_map
from vggt_n.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues

# GA utilities (global alignment)
import utils.opt as opt_utils


def _list_images(image_dir: str) -> List[str]:
	exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
	files = []
	for e in exts:
		files.extend(glob.glob(os.path.join(image_dir, e)))
	files = sorted(files)
	if len(files) == 0:
		raise FileNotFoundError(f"No images found in {image_dir}")
	return files


@torch.inference_mode()
def _run_vggt(images: torch.Tensor, device: str = "cuda", dtype: torch.dtype = torch.float16, chunk_size: int = 256):
	"""Run VGGT to get extrinsics (w2c), intrinsics, depth and confidence.

	Args:
		images: [S, 3, H, W] float32 in [0,1]
		device: cuda/cpu
		dtype: float16 or bfloat16 on modern GPUs
		chunk_size: model chunk size
	Returns:
		extrinsic (S, 3, 4) w2c, intrinsic (S, 3, 3), depth (S, H, W), depth_conf (S, H, W)
	"""
	model = VGGT(chunk_size=chunk_size)
	_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
	model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
	model.eval().to(device).to(dtype)
	model.track_head = None

	preds = model(images.to(device, dtype), verbose=False)
	extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
	extrinsic = extrinsic.squeeze(0).cpu().numpy()  # (S, 3, 4)
	intrinsic = intrinsic.squeeze(0).cpu().numpy()  # (S, 3, 3)
	depth = preds["depth"].squeeze(0).cpu().numpy()  # (S, H, W)
	depth_conf = preds["depth_conf"].squeeze(0).cpu().numpy()  # (S, H, W)
	return extrinsic, intrinsic, depth, depth_conf


def align_pose(pose_a, pose_b):
	"""Align pose_b to pose_a using Umeyama (similarity) on camera centers.

	Returns a 4x4 transform T such that: aligned_b = T @ b.
	"""
	import evo.core.geometry as geometry
	device = pose_a.device if torch.is_tensor(pose_a) else torch.device("cpu")

	if torch.is_tensor(pose_a):
		a_centers = pose_a[:, :3, 3].T
	else:
		a_centers = torch.from_numpy(pose_a[:, :3, 3].copy()).T
	if torch.is_tensor(pose_b):
		b_centers = pose_b[:, :3, 3].T
	else:
		b_centers = torch.from_numpy(pose_b[:, :3, 3].copy()).T

	r, t, c = geometry.umeyama_alignment(a_centers.cpu().numpy(), b_centers.cpu().numpy(), with_scale=True)
	T = torch.eye(4, device=device)
	T[:3, :3] = c * torch.from_numpy(r).to(device).float()
	T[:3, 3] = torch.from_numpy(t).to(device).float()
	return T


class Parser:
	"""VGGT-backed parser that estimates cameras and depth, with optional GA.

	Produces fields compatible with the training pipeline, inspired by mast3r.Parser.
	"""

	def __init__(
		self,
		data_dir: str,
		factor: int = 1,
		normalize: bool = False,
		test_every: int = 8,
		use_global_alignment: bool = True,
		max_query_pts: Optional[int] = 4096,
		shared_camera: bool = False,
		chunk_size: int = 256,
		results_dir: Optional[str] = None,
	):
		
		self.data_dir = data_dir
		self.factor = factor
		self.normalize = normalize
		self.test_every = test_every

		image_dir = os.path.join(data_dir, "images")
		image_path_list = _list_images(image_dir)
		base_names = [os.path.basename(p) for p in image_path_list]

		## for training with 30 images only
		if len(base_names) > 40:
			import random
			random_indexes = random.sample(range(len(base_names)), 40)

			base_names = [base_names[i] for i in random_indexes]
			image_path_list = [image_path_list[i] for i in random_indexes]
		
		
		self.image_names = base_names
		# Load images for VGGT (ratio preserve, common size)
		target_size = 518
		images_tensor, original_coords = load_and_preprocess_images_ratio(image_path_list, target_size)
		device = "cuda" if torch.cuda.is_available() else "cpu"
		dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
		
		# Run VGGT
		extrinsic, intrinsic, depth_map, depth_conf = _run_vggt(images_tensor, device=device, dtype=dtype, chunk_size=chunk_size)
		
		# Optional global alignment (pose optimization)
		if use_global_alignment:
			print("[VGGT Parser] Running global alignment (pose optimization)...")
			if max_query_pts is None:
				max_query_pts = 4096 if len(images_tensor) < 500 else 2048
			match_outputs = opt_utils.extract_matches(extrinsic, intrinsic, images_tensor.to(device), depth_conf, base_names, max_query_pts)
			match_outputs["original_width"] = images_tensor.shape[-1]
			match_outputs["original_height"] = images_tensor.shape[-2]
			extrinsic, intrinsic = opt_utils.pose_optimization(
				match_outputs,
				extrinsic,
				intrinsic,
				images_tensor.to(device),
				depth_map,
				depth_conf,
				base_names,
				device=device,
				shared_intrinsics=shared_camera,
			)

		# Convert w2c extrinsics to c2w camtoworlds (4x4)
		S = extrinsic.shape[0]
		w2c_4 = np.repeat(np.eye(4)[None, ...], S, axis=0)
		w2c_4[:, :3, :4] = extrinsic
		c2w_4 = np.linalg.inv(w2c_4)
		self.camtoworlds = c2w_4.astype(np.float32)  # (S, 4, 4)

		colmap_dir = os.path.join(data_dir, "sparse/0/")
		if not os.path.exists(colmap_dir):
			colmap_dir = os.path.join(data_dir, "sparse")
		if not os.path.exists(colmap_dir):
			print(f"COLMAP directory {colmap_dir} does not exist.")
			self.camtoworlds_gt = self.camtoworlds.copy()  # pseudo-GT
		else:
			print(f"COLMAP directory {colmap_dir} exist. Alligning poses...")
			reconstruction = pycolmap.Reconstruction(colmap_dir)
			c2w_mats = []
			bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
			fix = np.eye(4, dtype=np.float32)
			fix[1, 1] = -1  # flip Y
			fix[2, 2] = -1  # flip Z
			# get colmap camera poses
			for image_id, image in reconstruction.images.items():
				rot = image.cam_from_world.todict()['rotation']['quat']
				rot = R.from_quat(rot)
				rot = rot.as_matrix()
				trans = image.cam_from_world.todict()['translation']
				trans = trans.reshape(3, 1)
				w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
				c2w = np.linalg.inv(w2c)
				c2w = fix @ c2w
				c2w_mats.append({image.name: c2w})
			c2w_mats_sorted = []
			idx_not_found = []
			for i, name in enumerate(base_names):
				found = False
				for wm in c2w_mats:
					if name in wm:
						c2w_mats_sorted.append(wm[name])
						found = True
						break
				if not found:
					print(f'Warning: {name} not found in COLMAP reconstruction.')
					idx_not_found.append(i)

			self.camtoworlds_gt = np.array(c2w_mats_sorted).astype(np.float32)  # (S, 4, 4)
			
			print('-----------------------',self.camtoworlds.shape, self.camtoworlds_gt.shape)
			# Align estimated from VGGTX to GT
			T_align = align_pose(torch.from_numpy(self.camtoworlds_gt), torch.from_numpy(self.camtoworlds))
			# self.camtoworlds = np.einsum('ij,bjk->bik', T_align.numpy(), self.camtoworlds_gt)
			self.camtoworlds_gt = T_align.numpy() @ self.camtoworlds_gt

			# hack patch: scale correction
			scale_factor = self.camtoworlds[0,0,0] / self.camtoworlds_gt[0,0,0]
			self.camtoworlds_gt[:,0:3, 0:3] *= scale_factor

		np.save(os.path.join(results_dir, "camtoworlds_gt.npy"), self.camtoworlds_gt)
		np.save(os.path.join(results_dir, "camtoworlds_est.npy"), self.camtoworlds)
		
		self.intrinsics = intrinsic.astype(np.float32)  # (S, 3, 3)

		# Save resized images to a dedicated folder to match intrinsics resolution
		new_h, new_w = depth_map.shape[1], depth_map.shape[2]
		resized_dir = os.path.join(data_dir, f"images_{new_w}")
		os.makedirs(resized_dir, exist_ok=True)
		# If files not exist, write them (avoid re-writing every time)
		for src, name, tensor_img in zip(image_path_list, base_names, images_tensor):
			out_path = os.path.join(resized_dir, name)
			if not os.path.exists(out_path):
				# tensor_img is [3,H,W] in [0,1]
				img_np = (tensor_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
				imageio.imwrite(out_path, img_np)

		self.image_paths = [os.path.join(resized_dir, n) for n in base_names]
		self.image_size = (new_w, new_h)  # (W,H)

		# Build train/val split
		names_wo_ext = [os.path.splitext(n)[0] for n in base_names]
		self.train_split = [n for i, n in enumerate(names_wo_ext) if (i % test_every) != 0]
		self.test_split = [n for i, n in enumerate(names_wo_ext) if (i % test_every) == 0]

		# Camera dicts
		self.camera_ids = [i + 1 for i in range(S)]
		self.Ks_dict = {cid: self.intrinsics[i] for i, cid in enumerate(self.camera_ids)}
		self.params_dict = {cid: [] for cid in self.camera_ids}
		self.imsize_dict = {cid: Image.open(path).size for cid, path in zip(self.camera_ids, self.image_paths)}
		self.mask_dict = {cid: None for cid in self.camera_ids}

		# Depth maps (store disparity-like for training stability)
		disp_maps = 1.0 / (depth_map + 1e-8)
		self.depthmaps = disp_maps.astype(np.float32)  # (S, H, W)
		self.depthconfs = depth_conf.astype(np.float32)

		# Create a point cloud from depth (keep reasonably sized)
		points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)  # (S, H, W, 3)
		# Confidence thresholding similar to demo
		conf_thres_value = np.percentile(depth_conf, 0.5)
		conf_mask = depth_conf >= conf_thres_value
		conf_mask = randomly_limit_trues(conf_mask, 500_000)

		pts = points_3d[conf_mask]
		# Associated RGBs from resized images
		imgs_uint8 = (images_tensor.numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
		rgbs = imgs_uint8[conf_mask]
		if pts.shape[0] == 0:
			# Fallback: random Gaussian init
			print("[VGGT Parser] No confident points found; using random init point cloud")
			num_pts = 150_000
			pts = np.random.normal(0, 1.5, (num_pts, 3)).astype(np.float32)
			rgbs = np.random.randint(0, 255, (num_pts, 3), dtype=np.uint8)

		# Optionally subsample to a fixed budget
		max_pts = 200_000
		if pts.shape[0] > max_pts:
			sel = np.random.choice(pts.shape[0], size=max_pts, replace=False)
			pts = pts[sel]
			rgbs = rgbs[sel]

		self.points = pts.astype(np.float32)
		self.points_rgb = rgbs.astype(np.uint8)
		self.points_err = np.zeros((self.points.shape[0],), dtype=np.float32)

		# Scene scale from camera centers
		camera_locations = self.camtoworlds[:, :3, 3]
		scene_center = np.mean(camera_locations, axis=0)
		dists = np.linalg.norm(camera_locations - scene_center, axis=1)
		self.scene_scale = float(np.max(dists))


class Dataset:
	"""Dataset compatible with the training pipeline (similar to mast3r.Dataset).

	Returns per-item dict with image, K, camtoworld (4x4), optional depth, etc.
	"""

	def __init__(
		self,
		parser: Parser,
		split: str = "train",
		patch_size: Optional[int] = None,
		load_depths: bool = False,
		verbose: bool = False,
	):
		self.parser = parser
		self.split = split
		self.patch_size = patch_size
		self.load_depths = load_depths

		# Build indices from split
		name_list = [os.path.splitext(n)[0] for n in self.parser.image_names]
		if split in ["train"]:
			self.indices = [name_list.index(n) for n in self.parser.train_split]
		else:  # "val" or "test"
			self.indices = [name_list.index(n) for n in self.parser.test_split]

		self.intrinsics = self.parser.intrinsics[self.indices]
		self.camtoworlds = self.parser.camtoworlds[self.indices]
		self.camtoworlds_gt = self.parser.camtoworlds_gt[self.indices]
		self.image_size = self.parser.image_size

		if verbose:
			print(split, len(self.indices))

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, item: int) -> Dict[str, Any]:
		idx = self.indices[item]
		image = imageio.imread(self.parser.image_paths[idx])[..., :3]

		camera_id = self.parser.camera_ids[idx]
		K = self.parser.Ks_dict[camera_id].copy()
		camtoworld = self.parser.camtoworlds[idx]
		camtoworld_gt = self.parser.camtoworlds_gt[idx]
		mask = self.parser.mask_dict[camera_id]

		# Ensure camtoworld is 4x4; if 3x4, pad a bottom row [0,0,0,1]
		if camtoworld.shape == (3, 4):
			camtoworld_padded = np.eye(4, dtype=camtoworld.dtype)
			camtoworld_padded[:3, :4] = camtoworld
			camtoworld = camtoworld_padded

		if self.patch_size is not None:
			# Random crop on the image and adjust K accordingly
			h, w = image.shape[:2]
			ph = pw = self.patch_size
			x = np.random.randint(0, max(w - pw, 1))
			y = np.random.randint(0, max(h - ph, 1))
			image = image[y : y + ph, x : x + pw]
			K[0, 2] -= x
			K[1, 2] -= y

		data = {
			"K": torch.from_numpy(K).float(),
			"camtoworld": torch.from_numpy(camtoworld).float(),
			"camtoworld_gt": torch.from_numpy(camtoworld_gt).float(),
			"image": torch.from_numpy(image).float(),	
			"image_id": item,
		}
		if mask is not None:
			data["mask"] = torch.from_numpy(mask).bool()

		if self.load_depths:
			# Return VGGT pseudo disparity map resized to current image size
			disp = self.parser.depthmaps[idx]
			# Ensure 2D (H,W) by squeezing a trailing channel dimension if present
			if disp.ndim == 3 and disp.shape[-1] == 1:
				disp = disp[..., 0]
			elif disp.ndim == 3 and disp.shape[0] == 1:
				# handle (1,H,W)
				disp = disp[0]
			disp_t = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
			H, W = image.shape[:2]
			disp_resized = F.interpolate(disp_t, size=(H, W), mode="bilinear", align_corners=False).squeeze()
			# Return 2D [H, W]; DataLoader will batch to [B, H, W]
			data["depth"] = disp_resized.float()

		return data

