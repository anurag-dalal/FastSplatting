<p align="center">
  <h1 align="center"><strong> <img src="assets/logo.png" width="40" height="30">  FastSplatting: SfM free Ultra Fast Gaussian Splatting</strong></h1>
	<p align="center">
    <a href="https://github.com/anurag-dalal">Anurag Dalal </a><a href="https://orcid.org/0009-0007-9228-8222"><img src="assets/ORCID_iD.svg.png" width="15" height="15"></a>
    ·
    <a href="https://www.uia.no/english/about-uia/employees/danielh/index.html">Daniel Hagen </a><a href="https://orcid.org/0000-0002-7030-6676"><img src="assets/ORCID_iD.svg.png" width="15" height="15"></a>
    ·
    <a href="https://www.uia.no/english/about-uia/employees/kristimk/index.html">Kristian Muri Knausgård</a> <a href="https://orcid.org/0000-0003-4088-1642"><img src="assets/ORCID_iD.svg.png" width="15" height="15"></a>
    ·
    <a href="https://www.uia.no/english/about-uia/employees/kjellgr/index.html">Kjell Gunnar Robbersmyr </a><a href="https://orcid.org/0000-0001-9578-7325"><img src="assets/ORCID_iD.svg.png" width="15" height="15"></a>
  </p>
  <p align="center">
    <em>Department of Engineering Sciences, University of Agder, Jon Lilletuns vei 9, 4879 Grimstad, Norway</em>
  </p>

</p>
<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2509.25191-b31b1b.svg)](http://arxiv.org/)
[![](https://img.shields.io/badge/%F0%9F%9A%80%20-Project%20Page-blue)](https://github.com/anurag-dalal/FastSplatting)

</div>

<div align="center">
    <img src="assets/architecture.png">
</div>

## 📰 News
**[17.12.2025]** Code of FastSplatting had been released!

**[DD.MM.YYYY]** Paper release of our FastSplatting on arXiv! (coming soon)

## 🔎 Purpose
Instructions to run training and batch runs using
`training.py` and `train_all.py` included in this repo. Also visualize using `simple_viewer.py`

## ⚡ Quick Start
This project uses miniconda to install dependencies. CUDA 12.1 is used on a RTX 4090 GPU.
First, clone this repository to your local machine, and install the dependencies. 

```bash
git clone --recursive https://github.com/anurag-dalal/FastSplatting.git 
cd FastSplatting
conda env create -f environment.yml
conda activate FastSplatting
pip install -r requirements.txt
```

## 🎓 Training scripts
Now, put the image collection to path/to/your/scene. And to train use the foloowing:
```bash
python training.py default --data_dir path/to/your/scene --data_factor 1 --result_dir path/to/store/results --pose_opt_type mlp
```
If COLMAP path exist in `sparse/0` in data-dir, it will be used as fround truth, otherwise the VGGT-X outputs will be used.

***Configs***
- default/mcmc : The training strategy to use.
- data_factor : Downscale factor of image, if set to 2, then height and width will be halved.
- pose_opt_type(mlp/sfm) : Pose optimization strategy.


**Batch runs using `train_all.py`**:
- `train_all.py` is a convenience helper that iterates a list of dataset folders and result directories, runs the training (`training.main`) sequentially, then collects evaluation metrics (PSNR/SSIM/LPIPS) and computes an L1 between saved renders and source images.
- Edit the `DATASET_DIRS` / `result_dir_base` variables at the top of `train_all.py` to point to your datasets and desired output base. Then run:

```bash
python train_all.py
```

After a run, outputs are written under each `path/to/store/results/<scene>/` with the following subfolders:
- `ckpts/`  : saved model checkpoints
- `stats/`  : JSON metrics recorded during training / eval
- `renders/`: rendered images from eval steps
- `tb/`     : TensorBoard logs

`train_all.py` also writes a summary CSV `train_all_metrics.csv` to the configured result directory base including PSNR/SSIM/LPIPS/L1 and runtime/memory fields.

## 📸 Visulization
Use `simple_viewer.py` to visualize the trained 3DGS.
```bash
python ./simple_viewer.py --ckpt /path/to/trained/model/ckpts/ckpt_6999_rank0.pt --port 8081
```
## 🗂️ Datasets
- [Mip-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)
- [Tanks and Temples](https://www.tanksandtemples.org/download/)
- [RobustNeRF](https://storage.googleapis.com/jax3d-public/projects/robustnerf/robustnerf.tar.gz)

## 📋 Results
<div align="center">
    <img src="assets/renders.png">
</div>

The table below shows comparison across MipNerf Scenes:

| Scene   | 3DGS PSNR ↑ | 3DGS SSIM ↑ | 3DGS LPIPS ↓ | ZeroGS PSNR ↑ | ZeroGS SSIM ↑ | ZeroGS LPIPS ↓ | 3RGS PSNR ↑ | 3RGS SSIM ↑ | 3RGS LPIPS ↓ | FastSplat PSNR ↑ | FastSplat SSIM ↑ | FastSplat LPIPS ↓ |
|---------|-------------|-------------|--------------|---------------|---------------|----------------|------------|------------|---------------|------------------|------------------|-------------------|
| garden  | 24.85 | 0.729 | 0.126 | 25.47 | 0.839 | 0.107 | 26.44 | 0.820 | 0.131 | 22.65 | 0.6122 | 0.1415 |
| counter | 27.57 | 0.862 | 0.209 | 26.87 | 0.873 | 0.124 | 28.80 | 0.897 | 0.157 | 26.16 | 0.8293 | 0.1097 |
| bicycle | 17.52 | 0.303 | 0.567 | 23.10 | 0.707 | 0.201 | 24.89 | 0.727 | 0.252 | 24.22 | 0.7103 | 0.2003 |
| room    | 30.66 | 0.899 | 0.204 | -     | -     | -     | 31.82 | 0.924 | 0.154 | 28.48 | 0.8921 | 0.0897 |

## 🤗 Citation
If you find this repository useful for your research, please use the following BibTeX entry for citation.
```
COMING SOON
```

## 🤝 Acknowledgements
Thanks to [VGGT-X](https://github.com/Linketic/VGGT-X) and [3RGS](https://github.com/zsh523/3rgs) and many other inspiring works in the community. Thanks for their great work!

## 🪪 License
See the [LICENSE](LICENSE.txt) file for details about the license under which this code is made available.