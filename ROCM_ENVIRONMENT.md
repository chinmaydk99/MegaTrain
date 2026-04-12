# ROCm Environment Notes

This branch is validated for AMD / ROCm environments where a ROCm-enabled PyTorch
build is already installed in the container or base image.

The goal of this file is to keep the ROCm bring-up reproducible from git instead
of relying on ad hoc shell history from one long-lived container.

## Validated Install Path

Use the repo-owned bootstrap script:

```bash
bash scripts/install_rocm.sh
```

The script performs one validated sequence:

1. Verifies that the current `torch` build is ROCm-enabled (`torch.version.hip`).
2. Installs MegaTrain in editable mode.
3. Installs the validated ROCm extras from `requirements-rocm.txt`.
4. Builds `causal-conv1d==1.6.1` with `CAUSAL_CONV1D_FORCE_BUILD=TRUE` and `--no-build-isolation`.
5. Verifies the important fast-path imports (`DeepSpeedCPUAdam`, flash-attn CE, flash-linear-attention, and `causal_conv1d`).

## Validated Smoke Test

The following command completed successfully in a fresh `cdk-nightly2` container:

```bash
python examples/train.py \
  --config examples/configs/qwen_7b_rocm_paper.yaml \
  --num-steps 5 \
  --eval-num-samples 8
```

That run confirmed the intended ROCm fast path end to end:

- `flash_attention_2`
- flash-attn `CrossEntropyLoss`
- `DeepSpeedCPUAdam`
- CPU-master training plus exact-match evaluation export/materialization

## Validated Python Packages

These package versions were present in the clean ROCm validation environment:

- `accelerate==1.13.0`
- `deepspeed==0.18.9`
- `flash_attn==2.8.3`
- `flash-linear-attention==0.4.2`
- `causal_conv1d==1.6.1`
- `datasets==4.8.4`
- `einops==0.8.2`
- `ninja==1.13.0`
- `psutil==7.2.2`
- `kernels==0.13.0`

## Important Caveats

- `torchvision` is intentionally not part of the default ROCm bootstrap. In the fresh-container validation, a plain `pip install torchvision` path tried to pull a CUDA PyTorch stack, which is unsafe for this ROCm branch.
- For text-only models such as the Qwen 7B / 14B / 32B configs used in the showcase, `torchvision` is not required.
- If you need VLM-side `AutoProcessor` support, install a ROCm-compatible `torchvision` build that matches the already-installed PyTorch in your container or image. Do not rely on the generic PyPI wheel.
- `kernels` is still optional for the main showcase path. It remains recorded here because it was used during the GPT-OSS stretch probe, even though that model stayed blocked on ROCm by upstream CUDA-only kernel variants.

## Why This Exists

The code and configs already live in git. This note and `scripts/install_rocm.sh`
close the remaining environment gap so the validated AMD path is also described in
git rather than hidden in one specific container's history.
