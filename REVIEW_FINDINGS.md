# Lead Reviewer Report: MegaTrain ROCm Port vs. Paper (2604.05091v1)

**Reviewer**: Lead review chat
**Date**: 2026-04-11
**Scope**: Full code review of MegaTrain ROCm port against the paper's Algorithm 1, Sections 3.1–3.4, and Appendix A.

---

## Overall Assessment

**The math aligns. The port is structurally faithful.** The three-stream pipeline, double buffering, stateless layer templates, K-slab gradient pool, and block-wise recomputation all correctly implement Algorithm 1 and the appendix details. The ROCm abstraction layer (`gpu_compat.h`, `device.py`, build scripts) is clean and well-structured.

Six findings below — one high priority, two medium, three informational.

---

## Findings

### FINDING 1 — Info | cpu_master.py:978 | Cold-start first layer

The forward pass does a synchronous load of layer 0 before the pipeline starts:

```python
self._load_layer_to_buffer_async(0, 0)
self.weight_stream.synchronize()   # <-- blocks
self._unflatten_to_layer(0, 0)
```

This is a practical necessity to prime the pipeline. The paper's Algorithm 1 doesn't special-case it, but it's not a correctness issue — just a known one-time throughput hit per step.

**Action**: None required.

---

### FINDING 2 — Good | cpu_master.py:500-501 | template_free_events

The port adds `template_free_events` which prevents the GPU template from being reused until the gradient D2H finishes. The paper (Appendix A.4) describes ping-pong binding but doesn't explicitly describe this protection.

This is a **correct addition** — without it, there's a race between template reuse and gradient evacuation on both CUDA and ROCm.

**Action**: None required. Good work.

---

### FINDING 3 — Low | cpu_master.py:611 | Grad accumulation dtype fragility

```python
p_cpu.grad.add_(grad_view.to(device='cpu', dtype=p_cpu.grad.dtype))
```

The grad slab is in `config.dtype` (bfloat16), CPU master params are float32 (line 380). The `.to(dtype=...)` conversion is correct, but there's an implicit assumption that `p_cpu.grad` is always FP32. If someone changes the CPU master dtype, this could silently lose precision.

**Action**: Add an assertion or comment clarifying the FP32 master assumption.

---

### FINDING 4 — Medium | gpu_compat.h:37 | hipHostMalloc flag for MI355X

```cpp
#define GPU_MALLOC_HOST(ptr, size) hipHostMalloc(ptr, size)
```

`hipHostMalloc` defaults to `hipHostMallocDefault`. On MI355X's multi-GCD (Graphics Compute Die) architecture, you may get better PCIe bandwidth by using `hipHostMallocPortable`, which makes the pinned buffer accessible from all GCDs without extra address translation.

`cudaMallocHost` always allocates portable pinned memory, so the CUDA path doesn't have this issue.

**Action**: Benchmark both `hipHostMalloc(ptr, size)` and `hipHostMalloc(ptr, size, hipHostMallocPortable)` on the MI355X. If portable is faster for H2D/D2H, update the macro:

```cpp
#define GPU_MALLOC_HOST(ptr, size) hipHostMalloc(ptr, size, hipHostMallocPortable)
```

---

### FINDING 5 — Medium | optimizer.py:108 | Double bias correction in custom AdamW

```python
m_hat = p.m / bias_correction1           # Already corrected
step_size = self.lr / bias_correction1    # Corrected AGAIN
p.master.addcdiv_(m_hat, denom, value=-step_size)
```

The first moment gets divided by `(1 - beta1^t)` twice — once when computing `m_hat`, and again through `step_size`. The effective update uses `m / (1 - beta1^t)^2` instead of the correct `m / (1 - beta1^t)`.

**However**: `train.py` uses `torch.optim.AdamW` or `DeepSpeedCPUAdam` (lines 244-262), NOT this custom optimizer. The `optimizer.py` class appears to be unused dead code.

**Action**: Either fix the math (`step_size = self.lr`, not `self.lr / bias_correction1`) or remove/deprecate this file. As-is it will confuse anyone who tries to use it.

---

### FINDING 6 — HIGH | configs/*_rocm.yaml | attn_implementation will crash without flash-attn

All three ROCm YAML configs set:

```yaml
attn_implementation: "flash_attention_2"
```

But the feasibility analysis and TDD plan correctly identified that flash-attn may not be installed on the MI355X. The `default_attn_implementation()` in `training.py` correctly returns `"sdpa"` when on ROCm without flash-attn — but the **YAML config overrides this default**.

If flash-attn isn't installed, HuggingFace will try to use FA2 and throw an error at model load time.

**Action** (pick one):
1. Remove `attn_implementation` from the ROCm YAML files (let the default function choose)
2. Change to `attn_implementation: "sdpa"` in all `*_rocm.yaml` files
3. Add a comment explaining that FA2 requires the ROCm flash-attn fork

**Fix example** (option 2):

For `qwen_7b_rocm.yaml`, `qwen_14b_rocm.yaml`, and `llama3_8b_rocm.yaml`:
```yaml
  attn_implementation: "sdpa"
```

---

## Files Reviewed

| File | Lines | Verdict |
|------|-------|---------|
| `infinity/model/cpu_master.py` | 1331 | PASS — Algorithm 1 implemented correctly |
| `examples/train.py` | 428 | PASS — Training loop matches paper Section 3.1 |
| `csrc/gpu_compat.h` | 100 | PASS — Clean CUDA/HIP abstraction (see Finding 4) |
| `csrc/memory_ops.cpp` | 235 | PASS — Pool + async memcpy use compat macros correctly |
| `infinity/device.py` | 105 | PASS — Backend detection with proper caching |
| `infinity/config/training.py` | 136 | PASS — ROCm fallback to SDPA is correct |
| `infinity/optimizer.py` | 179 | FAIL — Dead code with math bug (Finding 5) |
| `setup.py` | 118 | PASS — Correct ROCm extension build |
| `csrc/setup.py` | 46 | PASS — Correct ROCm extension build |
| `scripts/benchmark_rocm.py` | 425 | PASS — Good coverage of bandwidth + overlap |
| `examples/configs/*_rocm.yaml` | 38 each | FAIL — Hardcoded FA2 (Finding 6) |

---

## Priority Actions for Dev Chat

1. **[HIGH] Fix ROCm YAML configs** — Change `attn_implementation` to `"sdpa"` or remove the line
2. **[MEDIUM] Benchmark hipHostMallocPortable** — On MI355X, test if portable pinned memory improves H2D/D2H bandwidth
3. **[MEDIUM] Fix or remove optimizer.py** — Double bias correction bug in dead code
4. **[LOW] Add dtype assertion in _grad_worker** — Protect against future changes to master weight precision
