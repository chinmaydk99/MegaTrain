# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd MegaTrain

# Install the validated ROCm stack for this branch
bash scripts/install_rocm.sh
```

That's it! The installation will automatically:
- Verify that the environment already has ROCm-enabled PyTorch
- Install MegaTrain in editable mode
- Install the validated ROCm-side fast-path dependencies
- Build `causal-conv1d` with the ROCm-safe flags required by this branch

For the exact AMD-side package notes and caveats, see `ROCM_ENVIRONMENT.md`.

## Usage

### Using YAML Configuration (Recommended)

```bash
# Run the validated 7B ROCm smoke test
python examples/train.py --config examples/configs/qwen_7b_rocm_paper.yaml --num-steps 5 --eval-num-samples 8

# Train with Qwen 32B
python examples/train.py --config examples/configs/qwen_32b_mi355x.yaml

# Train with Qwen 3.5 27B
python examples/train.py --config examples/configs/qwen3_5_27b.yaml
```

### Override Configuration

```bash
# Override specific parameters
python examples/train.py \
    --config examples/configs/qwen_32b_mi355x.yaml \
    --batch-size 64 \
    --num-steps 500
```

### Using in Python

```python
from infinity import CPUMasterModel, CPUMasterConfig, MetaMathDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load configuration
config = CPUMasterConfig(
    model_name="Qwen/Qwen2.5-32B-Instruct",
    dataset_path="/path/to/dataset",
    batch_size=96,
    max_seq_len=1024,
)

# Load model
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
hf_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

# Create CPU Master model
model = CPUMasterModel(hf_model, config)

# Train...
```

## Configuration Files

Configuration files are located in `examples/configs/`:

- **qwen_7b_rocm_paper.yaml** - Qwen 2.5 7B ROCm paper-path config
- **qwen_14b_rocm_paper.yaml** - Qwen 2.5 14B ROCm paper-path config
- **qwen_32b_mi355x.yaml** - Qwen 2.5 32B single-MI355X showcase config
- **qwen_72b_mi355x.yaml** - Qwen 2.5 72B proof-of-life config
- **qwen3_5_27b.yaml** - Qwen 3.5 27B hybrid attention config

See `examples/configs/README.md` for detailed configuration guide.

## What Gets Installed

When you run `bash scripts/install_rocm.sh`:

1. **ROCm Check**: Verifies the current `torch` build exposes `torch.version.hip`
2. **Python Package**: editable `megatrain` package with all modules
3. **ROCm Extras**: `flash-attn`, `deepspeed`, `flash-linear-attention`, `kernels`, and `accelerate`
4. **ROCm Source Build**: `causal-conv1d` built with the validated ROCm flags

## Verify Installation

```python
import torch

from deepspeed.ops.adam import DeepSpeedCPUAdam
from flash_attn.losses.cross_entropy import CrossEntropyLoss

print(torch.__version__, torch.version.hip)
print(DeepSpeedCPUAdam.__name__)
print(CrossEntropyLoss.__name__)
```

## Requirements

- **GPU**: AMD Instinct MI300 / MI350 / MI355-class GPU
- **CPU RAM**: 256GB+ for 32B models
- **PyTorch**: ROCm-enabled build already installed in the container or image
- **Python**: 3.9+

## Troubleshooting

### ROCm Install Script Failed

The script expects ROCm-enabled PyTorch to be installed already. Check:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.hip)"
```

If `torch.version.hip` prints `None`, install the correct ROCm PyTorch build first.

### `torchvision` / VLM Import Issues

The default ROCm bootstrap on this branch intentionally does not install `torchvision`.
For text-only configs, this is fine. For VLM work, install a ROCm-compatible
`torchvision` build that matches the existing PyTorch in your image instead of using
the generic PyPI wheel.

### Import Errors

Make sure you're in the correct directory:
```bash
cd MegaTrain
bash scripts/install_rocm.sh
```

### Out of Memory

Reduce batch size in your config file:
```yaml
training:
  batch_size: 64  # Reduce from 96
```

## Next Steps

- Read the [Configuration Guide](examples/configs/README.md)
- Check the [Main README](README.md) for architecture details
- Explore example configurations in `examples/configs/`
