# Open / Unfixed CPU vs GPU Bug Reproducers

Minimal repros for **open and unfixed** CPU vs GPU divergence bugs in PyTorch / TensorFlow.
Each `.py` file is a single bare bug — no helpers, no harness — and runs in a few lines.

Reproducer outputs below are captured on:
- PyTorch **2.11.0+cu128**, CUDA **12.8**
- TensorFlow **2.21.0**
- GPU: **NVIDIA RTX 6000 Ada Generation** (compute 8.9)

---

## Seed bugs — one reproducer per upstream issue

### `cpu_gpu/tf_115731_cumsum_bf16.py` — tensorflow#115731

```
CPU error vs fp64 ref: 1.4584e+03
GPU error vs fp64 ref: 6.3028e+01
CPU/GPU error ratio: 23.1x
```

→ `tf.math.cumsum` with `bfloat16` input stays in bf16 on CPU and accumulates large rounding error; GPU silently promotes to fp32 internally and is ~23× more accurate.

### `cpu_gpu/tf_115733_reduce_std_fp16.py` — tensorflow#115733

```
reduce_std  cpu=nan  gpu=inf
```

→ `tf.math.reduce_std` on `float16` input: CPU returns `nan` (Welford `inf − inf`); GPU returns `inf` (two-pass algorithm overflows).

### `cpu_gpu/tf_115736_cast_nan_inf_to_int.py` — tensorflow#115736

```
to=int32  cpu=[-2147483648 -2147483648 -2147483648]  gpu=[          0  2147483647 -2147483648]
to=int64  cpu=[-9223372036854775808 -9223372036854775808 -9223372036854775808]  gpu=[-9223372036854775808  9223372036854775807 -9223372036854775808]
```

→ `tf.cast(NaN/+Inf/-Inf → int32/int64)`. CPU collapses everything to `INT_MIN`; GPU returns `0` for NaN, `INT_MAX` for `+Inf`, and `INT_MIN` for `-Inf`.

### `cpu_gpu/pt_156020_ifft_complex64.py` — pytorch#156020

```
cpu: tensor([0.+infj, 0.+0.j, 0.+0.j, 0.+0.j])
gpu: tensor([nan+infj, 0.+0.j, 0.+0.j, 0.+0.j])
```

→ `torch.fft.ifft` on `complex64` with a large pure-imaginary input: CPU keeps the real part at `0`; CUDA injects `nan` into the real part of the first output element.

### `cpu_gpu/pt_156959_mixture_log_prob.py` — pytorch#156959

```
pdf_cpu range: [9.890028e-14, 7.659545e-02]
pdf_gpu range: [8.349946e-05, 1.744377e-01]
max abs diff: 1.483258e-01
```

→ `MixtureSameFamily(MultivariateNormal).log_prob` on a 27-component GMM evaluated over a 1000×1000 grid. GPU floor is ~10⁹× higher than CPU and the peak PDF is 2.3× larger. Max abs PDF diff 0.148.

### `cpu_gpu/pt_158172_fmin_int_out.py` — pytorch#158172

```
cpu: tensor([-1], dtype=torch.int16)
gpu: tensor([-31072], dtype=torch.int16)
```

→ `torch.fmin(int64, int64, out=int16)`: CPU computes the min in int64 (`min(100000, -1) = -1`) then narrows. CUDA pre-casts inputs to int16 first, then takes the min, returning `-31072`.

### `cpu_gpu/pt_158419_copysign_fp16_nan.py` — pytorch#158419

```
cpu: tensor([ 1., -1.,  1.], dtype=torch.float16)
gpu: tensor([1., 1., 1.], dtype=torch.float16)
```

→ `torch.copysign` on `float16` with NaN sign argument. CPU honours the NaN sign bit (`-NaN → -1`); CUDA loses it (`-NaN → +1`). fp16-specific.

### `cpu_gpu/pt_170168_linspace_int64.py` — pytorch#170168

```
cpu: [4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -3]
gpu: [4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -3]
differing indices: [14, 21, 35]
```

→ `torch.linspace(4.3, -3, 50, dtype=int64)` differs at indices 14, 21, 35 between CPU and CUDA.

---

## Variants — same bug, different input

Each variant takes a seed bug above and exercises it with a different input that still produces a clearly divergent CPU vs GPU output. Together they widen the trigger surface.

### `cpu_gpu/tf_115731_cumsum_fp16.py` — tensorflow#115731 (fp16 path)

```
CPU error vs fp64 ref: 1.4662e+03
GPU error vs fp64 ref: 5.4564e+01
CPU/GPU error ratio: 26.9x
```

→ Originally reported for bf16; **fp16 is broken too**, ~27× error ratio.

### `cpu_gpu/tf_115733_reduce_mean_fp16.py` — tensorflow#115733 (`reduce_mean`)

```
reduce_mean  cpu=nan  gpu=-452.5
```

→ Bug propagates to `reduce_mean`. Cleaner output: CPU NaN, GPU finite scalar.

### `cpu_gpu/tf_115736_cast_inf_to_uint64.py` — tensorflow#115736 (uint64)

```
cpu: [9223372036854775808]
gpu: [18446744073709551615]
```

→ Affects unsigned int targets too: `+inf → uint64` saturates to `2^63` on CPU but `2^64 − 1` on GPU.

### `cpu_gpu/tf_86256_adjust_hue_nan.py` — tensorflow#86256 (NaN class flip)

```
cpu: [nan 0.5 nan 0.5 0.5 0.5]
gpu: [0.5 0.5 0.5 0.5 0.5 0.5]
```

→ A single NaN in the R channel poisons CPU output but is silently absorbed on GPU. Output classes (NaN vs finite) differ, not just values.

### `cpu_gpu/pt_156020_ifft_near_fltmax.py` — pytorch#156020 (near `FLT_MAX`)

```
cpu: tensor([0.+infj, 0.+0.j, 0.+nanj, 0.+0.j])
gpu: tensor([nan+infj, 0.+0.j, nan+nanj, 0.+0.j])
```

→ `imag = 3.4e38` (vs the issue's `8.5e37`): NaNs at different indices, different real-part bit patterns.

### `cpu_gpu/pt_158172_fmax_uint8_out.py` — pytorch#158172 (`fmax` + uint8)

```
cpu: tensor([160], dtype=torch.uint8)
gpu: tensor([255], dtype=torch.uint8)
```

→ Generalises beyond `fmin` + int16: `fmax` with `uint8` `out=` also diverges. CPU narrows after the reduction; CUDA narrows before.

### `cpu_gpu/pt_158419_signbit_fp16_neg_nan.py` — pytorch#158419 (signbit primitive)

```
cpu: tensor([True, True, True])
gpu: tensor([False, False, False])
```

→ `torch.signbit(-NaN)` on fp16 is itself the divergence — `copysign` is just the visible symptom.

### `cpu_gpu/pt_162235_median_neg_zero.py` — pytorch#162235 (median)

```
cpu: tensor(-0.) signbit: True
gpu: tensor(0.)  signbit: False
```

→ `torch.median([-0.0, 0.0])` flips the sign bit on CUDA. `median` was not in the issue's original op list.

### `cpu_gpu/pt_170168_linspace_large_range.py` — pytorch#170168 (1000 points, 1e6 range)

```
# differing indices: 32
first 5 cpu: [1000000, 997997, 995995, 993993, 991991]
first 5 gpu: [1000000, 997998, 995996, 993994, 991992]
```

→ Issue example was 50 points, 3 differing indices. With 1000 points across `[1e6, -1e6]`, **32 indices** disagree — the rounding-order bug grows with element count.

---

## Summary

| Issue | Seed file | Variant file |
|---|---|---|
| tensorflow#115731 (`cumsum`)               | `tf_115731_cumsum_bf16.py`       | `tf_115731_cumsum_fp16.py`        |
| tensorflow#115733 (`reduce_std`)           | `tf_115733_reduce_std_fp16.py`   | `tf_115733_reduce_mean_fp16.py`   |
| tensorflow#115736 (`cast`)                 | `tf_115736_cast_nan_inf_to_int.py` | `tf_115736_cast_inf_to_uint64.py` |
| tensorflow#86256  (`adjust_hue`)           | —                                | `tf_86256_adjust_hue_nan.py`      |
| pytorch#156020    (`fft.ifft`)             | `pt_156020_ifft_complex64.py`    | `pt_156020_ifft_near_fltmax.py`   |
| pytorch#156959    (`MixtureSameFamily`)    | `pt_156959_mixture_log_prob.py`  | —                                 |
| pytorch#158172    (`fmin/fmax` int `out=`) | `pt_158172_fmin_int_out.py`      | `pt_158172_fmax_uint8_out.py`     |
| pytorch#158419    (`copysign` fp16 NaN)    | `pt_158419_copysign_fp16_nan.py` | `pt_158419_signbit_fp16_neg_nan.py` |
| pytorch#162235    (`-0.0` semantics)       | —                                | `pt_162235_median_neg_zero.py`    |
| pytorch#170168    (`linspace` int)         | `pt_170168_linspace_int64.py`    | `pt_170168_linspace_large_range.py` |
