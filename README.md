# Gaussian-Splatting (Reproducible WSL Stack)

<p align="center">
  <img src="https://img.shields.io/badge/Platform-WSL2-blue" />
  <img src="https://img.shields.io/badge/CUDA-12.8-green" />
  <img src="https://img.shields.io/badge/PyTorch-2.8.0-orange" />
  <img src="https://img.shields.io/badge/COLMAP-CUDA--Enabled-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<p align="center">
  Reproducible 3D Gaussian Splatting pipeline with CUDA-enabled COLMAP and a local FastAPI UI for object-centric reconstruction in WSL.
</p>

---

# 🚀 Quick Start (5 Minutes)

```bash
# 1️⃣ Clone
 git clone --recursive https://github.com/adityagit94/Gaussian-Splatting.git
 cd Gaussian-Splatting

# 2️⃣ Install system dependencies
 chmod +x scripts/install_system_deps_wsl.sh
 ./scripts/install_system_deps_wsl.sh

# 3️⃣ Create environment
 conda create -n gs python=3.10 -y
 conda activate gs

# 4️⃣ Install PyTorch (CUDA 12.8 wheels)
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 5️⃣ Install Python deps
 pip install -r requirements.txt

# 6️⃣ Start Local UI
 cd gs_platform/app
 uvicorn server:app --host 0.0.0.0 --port 7860
```

Open: http://127.0.0.1:7860

---

# 📸 Platform Preview

<img width="7620" height="4448" alt="127 0 0 1_7860_(High res)" src="https://github.com/user-attachments/assets/17738385-1956-4c4a-b43f-855f89ee39fe" />


---

# 📁 Repository Structure

```
Gaussian-Splatting/
│
├── gaussian-splatting/        # Graphdeco-Inria GS (submodule recommended)
├── gs_platform/               # Local FastAPI UI
├── scripts/                   # Setup & build scripts
├── docs/                      # Documentation & screenshots
├── requirements.txt
├── environment.md
└── README.md
```

---

# 🧱 One-Command Setup (Recommended)

You can optionally create a full automated setup script:

```bash
chmod +x scripts/full_setup.sh
./scripts/full_setup.sh
```

Example content of `full_setup.sh`:

```bash
#!/bin/bash
set -e

./scripts/install_system_deps_wsl.sh
conda create -n gs python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

# 🔧 System Requirements

- Windows 11
- WSL2 Ubuntu 22.04
- NVIDIA GPU (WSL CUDA supported)
- NVIDIA driver installed on Windows
- ⚠️ Recommended Python Version: 3.10  
Python 3.13 is NOT supported due to CUDA extension build issues.

Verify GPU:

```bash
nvidia-smi
```

---

# 🏗 Build COLMAP (CUDA)

```bash
chmod +x scripts/build_colmap_cuda.sh
./scripts/build_colmap_cuda.sh
```

Verify:

```bash
colmap -h | grep CUDA
```

---

# 🧠 Install Gaussian Splatting

```bash
git submodule add https://github.com/graphdeco-inria/gaussian-splatting.git gaussian-splatting
git submodule update --init --recursive

pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -e gaussian-splatting/submodules/simple-knn
pip install -e gaussian-splatting/submodules/fused-ssim
```

---

# 🎯 Example Training Command

```bash
SCENE="/mnt/c/gs_data/datasets/bullet"
UNDIST="$SCENE/undistorted_0"
OUT="/mnt/c/gs_data/outputs/bullet_run1"

python gaussian-splatting/train.py -s "$UNDIST" -m "$OUT" \
  --iterations 60000 \
  --resolution 2 \
  --densify_from_iter 800 \
  --densify_until_iter 4000 \
  --densification_interval 250 \
  --densify_grad_threshold 0.0015 \
  --opacity_reset_interval 800 \
  --percent_dense 0.0015 \
  --lambda_dssim 0.3 \
  --random_background
```

---

# 🖥 Local Platform UI

```bash
cd gs_platform/app
uvicorn server:app --host 0.0.0.0 --port 7860
```

Access:

```
http://127.0.0.1:7860
```

---

# 🧹 Recommended Settings (Object-Only Scenes)

To reduce floaters:

- `--resolution 2`
- Lower `--percent_dense`
- Reduce `--densify_until_iter`
- Use `--random_background`
- Crop/delete background gaussians after training

---

# 🛠 Troubleshooting

## CUDA OOM

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## CUDA Not Detected

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

# 📌 Environment Reference

See `environment.md` for: In my case these were the 
- Driver version - 576.88
- CUDA version - 12.9
- PyTorch version - 2.8.0+cu128 (CUDA 12.8 wheels)
- COLMAP commit - 3.14.0.dev0 (Commit fe411191 on 2026-02-12) with CUDA

---

# 📜 License & Attribution

Gaussian Splatting by Graphdeco-Inria.

This repository provides a reproducible wrapper + local execution platform around it.

