# MI355X Showcase Results

This note captures the current single-GPU AMD MI355X / MI350X MegaTrain bring-up results.

## Current Results

| Model | Recipe | Result |
| --- | --- | --- |
| `Qwen2.5-7B-Instruct` | 1000 steps, `BS=96` | Training completed. Loss `0.6643 -> 0.1432`, peak GPU `46.93 GB`, peak CPU `102.40 GB`, average throughput about `406.5 TFLOPS`. |
| `Qwen2.5-14B-Instruct` | 20-step sanity, `BS=256` | Stable. Loss `0.6963 -> 0.1504`, peak GPU `137.13 GB`, peak CPU `190.56 GB`, average throughput about `507.2 TFLOPS`, exact-match `73.44% (47/64)` on a capped eval. |
| `Qwen2.5-32B-Instruct` | 20-step smoke, `BS=300` | Stable. Loss `0.4657 -> 0.1295`, peak GPU `221.00 GB`, peak CPU `393.24 GB`, average throughput about `524.6 TFLOPS`, exact-match `100% (4/4)` on a tiny capped eval. |
| `Qwen2.5-72B-Instruct` | proof-of-life, `BS=200` | Real training steps on one GPU. Step 1: `268.5 TFLOPS`, step 2: `556.1 TFLOPS`, trainer-reported GPU `215.27 GB`, `rocm-smi` VRAM about `266.6 GB`, CPU about `855.8 GB`. |
| `openai/gpt-oss-120b` | stretch probe | Blocked by upstream CUDA-only attention kernel packaging, not host RAM exhaustion. `transformers` tried to load `kernels-community/vllm-flash-attn3`, but no ROCm build variant was available. |

## Why This Matters

- `14B` is a correctness and performance baseline.
- `32B` is the first strong showcase point: full-parameter single-GPU fine-tuning without LoRA or distributed training.
- `72B` is the jaw-drop proof-of-life: a dense model that normally pushes users toward multi-GPU training can execute real training steps on one accelerator.
- The `120B` stretch result is also informative: software kernel availability becomes the next wall after memory capacity.

## Practical Notes

- Full exact-match eval over the entire `118.5k` MetaMathQA eval split is operationally too slow for iteration. Capped eval subsets are the right default for development runs.
- Measured runs should stay sequential on one GPU. Even with isolated HBM, parallel jobs on the same node pollute host-side CPU / memory / PCIe behavior and distort throughput numbers.
- The node's large host RAM budget is essential to the story. The claim is best framed as **single-accelerator, single-node, no distributed training stack**, not "consumer workstation magic."
