#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python}"

echo "[MegaTrain ROCm] Verifying preinstalled ROCm PyTorch..."
"${python_bin}" - <<'PY'
try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is not installed. Install a ROCm-enabled torch build before "
        "running scripts/install_rocm.sh."
    ) from exc

hip_version = getattr(torch.version, "hip", None)
if not hip_version:
    raise SystemExit(
        f"Expected a ROCm-enabled torch build, but found torch "
        f"{torch.__version__} with hip={hip_version!r}."
    )

print(f"Detected torch {torch.__version__} with ROCm runtime {hip_version}.")
PY

echo "[MegaTrain ROCm] Installing editable MegaTrain package..."
"${python_bin}" -m pip install -e "${repo_root}"

echo "[MegaTrain ROCm] Installing validated ROCm extras..."
"${python_bin}" -m pip install -r "${repo_root}/requirements-rocm.txt"

# Build from source on ROCm instead of relying on the package's default wheel path.
echo "[MegaTrain ROCm] Building causal-conv1d with ROCm-safe flags..."
CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    "${python_bin}" -m pip install --force-reinstall --no-build-isolation --no-deps \
    "causal-conv1d==1.6.1"

echo "[MegaTrain ROCm] Verifying fast-path imports..."
"${python_bin}" - <<'PY'
import torch
import causal_conv1d
import deepspeed
import flash_attn
import fla

from deepspeed.ops.adam import DeepSpeedCPUAdam
from flash_attn.losses.cross_entropy import CrossEntropyLoss

print(f"torch={torch.__version__} hip={torch.version.hip}")
print(f"deepspeed_cpu_adam={DeepSpeedCPUAdam.__name__}")
print(f"flash_attn_ce={CrossEntropyLoss.__name__}")
print(f"flash_linear_attention={fla.__name__}")
print(f"causal_conv1d={causal_conv1d.__name__}")
PY

cat <<'EOF'
[MegaTrain ROCm] Install complete.

Validated default ROCm path:
- flash_attention_2
- flash-attn CrossEntropyLoss
- DeepSpeed CPUAdam
- flash-linear-attention
- causal-conv1d (ROCm source build)

Note:
- This script intentionally does not install torchvision.
- For text-only configs on this branch, torchvision is not required.
- For VLM work, install a ROCm-compatible torchvision build that matches your
  existing PyTorch instead of using the generic PyPI wheel.
EOF