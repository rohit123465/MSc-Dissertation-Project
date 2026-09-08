#!/bin/bash
#SBATCH --job-name=semantic-003
#SBATCH --output=/dcs/22/u2265464/Documents/MSc-Dissertation-Project/evaluation/logs/%x-%j.out
#SBATCH --error=/dcs/22/u2265464/Documents/MSc-Dissertation-Project/evaluation/logs/%x-%j.err
#SBATCH --partition=falcon-debug
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:ampere:1

set -euo pipefail

PROJECT_DIR="/dcs/22/u2265464/Documents/MSc-Dissertation-Project"
VENV_DIR="$PROJECT_DIR/.venv-tia210"
TASK_FILE="$PROJECT_DIR/evaluation/tasks/semantic_003.json"

cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/evaluation/logs"
mkdir -p "$PROJECT_DIR/evaluation/results/semantic_003"

echo "========================================"
echo "Semantic-segmentation functional test"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Compute node: $(hostname)"
echo "Start time: $(date --iso-8601=seconds)"
echo "Project: $PROJECT_DIR"
echo "Task: $TASK_FILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"

module load tiatoolbox/2.1.0
source "$VENV_DIR/bin/activate"

echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"

echo "Loaded modules:"
module list

echo "Original LD_LIBRARY_PATH:"
echo "${LD_LIBRARY_PATH:-not-set}"

# The tiatoolbox module automatically loads CUDA 12.2 and its cuDNN
# libraries. The activated environment uses PyTorch 2.12.0+cu130.
# Removing the external runtime-library search path prevents the older
# cuDNN library from overriding PyTorch's bundled CUDA 13 libraries.
unset LD_LIBRARY_PATH

echo "Cleaned LD_LIBRARY_PATH:"
echo "${LD_LIBRARY_PATH:-not-set}"

echo "GPU information:"
nvidia-smi

echo "Running CUDA and cuDNN preflight test..."

python - <<'PY'
import sys

import torch
import torch.nn as nn
import torchvision
import tiatoolbox

print("Python:", sys.version)
print("Python executable:", sys.executable)
print("TIAToolbox:", tiatoolbox.__version__)
print("Torch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())
print("cuDNN available:", torch.backends.cudnn.is_available())
print("cuDNN version:", torch.backends.cudnn.version())

if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch cannot access the GPU allocated by Slurm."
    )

print("GPU:", torch.cuda.get_device_name(0))

model = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    padding=1,
).cuda()

input_tensor = torch.rand(
    2,
    3,
    224,
    224,
    device="cuda",
)

with torch.inference_mode():
    output_tensor = model(input_tensor)

torch.cuda.synchronize()

print("cuDNN convolution succeeded")
print("Output shape:", tuple(output_tensor.shape))
print("Output mean:", output_tensor.mean().item())
PY

echo "CUDA and cuDNN preflight passed."
echo "Starting semantic-segmentation function..."

python "$PROJECT_DIR/evaluation/run_tool.py" \
    --task "$TASK_FILE"

echo "Semantic-segmentation function finished."
echo "End time: $(date --iso-8601=seconds)"

echo "Generated files:"

find "$PROJECT_DIR/evaluation/results/semantic_003" \
    -type f \
    -printf '%p %s bytes\n'