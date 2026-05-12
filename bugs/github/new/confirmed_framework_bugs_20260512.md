# Confirmed Framework Bugs With New Inputs

Filtered on 2026-05-12 after re-review.
Only confirmed bugs are kept.
Dropped as non-bugs or too weak: TensorFlow #99759, TensorFlow #93648, PyTorch #170762, PyTorch #156152.

Environment: TensorFlow 2.21.0, PyTorch 2.11.0+cu128.

## TensorFlow #97125
confirmation: CPU matches integer overflow semantics used by NumPy; GPU returns an unrelated value.

```python
import tensorflow as tf

x = tf.constant([-47], tf.int64)
y = tf.constant([66], tf.int64)
with tf.device("/CPU:0"):
    print(tf.pow(x, y).numpy())
with tf.device("/GPU:0"):
    print(tf.pow(x, y).numpy())
```

## TensorFlow #93162
confirmation: identical sparse cross with `inf` serializes to different int sentinel strings.

```python
import tensorflow as tf

a = tf.constant([[float("inf")], [2.0]], tf.float32)
b = tf.constant([[5], [6]], tf.int32)
with tf.device("/CPU:0"):
    print(tf.sparse.cross([a, b]).values.numpy())
with tf.device("/GPU:0"):
    print(tf.sparse.cross([a, b]).values.numpy())
```

## TensorFlow #94378
confirmation: CPU rejects an out-of-bounds sparse index; GPU silently returns a tensor.

```python
import tensorflow as tf

def f(device):
    with tf.device(device):
        try:
            print(tf.raw_ops.SparseToDense(
                sparse_indices=tf.constant([2], tf.int32),
                output_shape=tf.constant([1], tf.int32),
                sparse_values=tf.constant([7], tf.uint16),
                default_value=tf.constant(0, tf.uint16),
                validate_indices=False,
            ).numpy())
        except Exception as e:
            print(type(e).__name__)

f("/CPU:0")
f("/GPU:0")
```

## TensorFlow #94376
confirmation: CPU rejects negative sparse gradient indices; GPU silently returns output.

```python
import tensorflow as tf

def f(device):
    with tf.device(device):
        try:
            print(tf.raw_ops.SparseSegmentSqrtNGradV2(
                grad=tf.constant([1.0, 2.0, 3.0], tf.float64),
                indices=tf.constant([-2], tf.int64),
                segment_ids=tf.constant([-2], tf.int64),
                dense_output_dim0=tf.constant(2, tf.int32),
            ))
        except Exception as e:
            print(type(e).__name__)

f("/CPU:0")
f("/GPU:0")
```

## TensorFlow #94151
confirmation: CPU rejects negative sparse gradient indices; GPU silently returns output.

```python
import tensorflow as tf

def f(device):
    with tf.device(device):
        try:
            print(tf.raw_ops.SparseSegmentSumGradV2(
                grad=tf.constant([1.0, 2.0, 3.0], tf.float64),
                indices=tf.constant([-2], tf.int64),
                segment_ids=tf.constant([-2], tf.int64),
                dense_output_dim0=tf.constant(2, tf.int32),
            ))
        except Exception as e:
            print(type(e).__name__)

f("/CPU:0")
f("/GPU:0")
```

## TensorFlow #106388
confirmation: CPU rejects out-of-range reverse indices; GPU silently returns output.

```python
import tensorflow as tf

def f(device):
    with tf.device(device):
        try:
            print(tf.raw_ops.SparseFillEmptyRowsGrad(
                reverse_index_map=tf.constant([20, 21], tf.int64),
                grad_values=tf.constant([3, 4, 5], tf.int64),
            ))
        except Exception as e:
            print(type(e).__name__)

f("/CPU:0")
f("/GPU:0")
```

## TensorFlow #94119
confirmation: process terminates with floating-point exception.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
import numpy as np
import tensorflow as tf

shape = tf.constant([6, 8, 5, 1, 4], tf.int32)
grad = tf.constant(np.full((6, 8, 5, 1, 4), 1.4013e-45, np.float32), tf.bfloat16)
print(tf.raw_ops.AvgPool3DGrad(
    orig_input_shape=shape,
    grad=grad,
    ksize=[1, 2, 2, 2, 1],
    strides=[1, 1, 1, 1, 1],
    padding="VALID",
    data_format="NDHWC",
))
```

## TensorFlow #97042
confirmation: CPU matches NumPy-style int16 wraparound; GPU saturates/clamps.

```python
import tensorflow as tf

x = tf.constant([[[[70000.0], [90000.0]], [[110000.0], [130000.0]]]], tf.float32)
with tf.device("/CPU:0"):
    print(tf.experimental.numpy.cumsum(x, axis=1, dtype=tf.int16).numpy())
with tf.device("/GPU:0"):
    print(tf.experimental.numpy.cumsum(x, axis=1, dtype=tf.int16).numpy())
```

## TensorFlow #94655
confirmation: singular matrix should have zero sign and `-inf` logabsdet; GPU returns finite logabsdet.

```python
import tensorflow as tf

x = tf.constant([[2.0, 4.0, 6.0], [4.0, 10.0, 12.0], [6.0, 12.0, 18.0]], tf.float32)
with tf.device("/CPU:0"):
    print([v.numpy() for v in tf.linalg.slogdet(x)])
with tf.device("/GPU:0"):
    print([v.numpy() for v in tf.linalg.slogdet(x)])
```

## TensorFlow #96180
confirmation: complex reciprocal with infinite component disagrees with NumPy-compatible zero result on GPU.

```python
import tensorflow as tf

x = tf.constant([[complex(float("inf"), 2.0)], [complex(-float("inf"), float("inf"))]], tf.complex128)
with tf.device("/CPU:0"):
    print(tf.math.reciprocal(x).numpy())
with tf.device("/GPU:0"):
    print(tf.math.reciprocal(x).numpy())
```

## TensorFlow #98410
confirmation: complex absolute with infinite component should be infinite; GPU returns NaN.

```python
import tensorflow as tf

x = tf.constant([complex(float("inf"), float("nan")), complex(float("nan"), -float("inf"))], tf.complex128)
with tf.device("/CPU:0"):
    print(tf.math.abs(x).numpy())
with tf.device("/GPU:0"):
    print(tf.math.abs(x).numpy())
```

## TensorFlow #94657
confirmation: CPU rejects singular solve; GPU returns identity.

```python
import tensorflow as tf

x = tf.constant([[2.0, 4.0, 6.0], [4.0, 10.0, 12.0], [6.0, 12.0, 18.0]], tf.float64)
def f(device):
    with tf.device(device):
        try:
            print(tf.linalg.solve(x, x).numpy())
        except Exception as e:
            print(type(e).__name__)

f("/CPU:0")
f("/GPU:0")
```

## TensorFlow #97204
confirmation: CPU silently returns scalar for non-broadcastable shapes; GPU rejects input.

```python
import tensorflow as tf

x = tf.ones((3, 1), tf.float32)
y = tf.ones((1, 5, 2), tf.float32)
def f(device):
    with tf.device(device):
        try:
            print(tf.raw_ops.NotEqual(x=x, y=y, incompatible_shape_error=False).numpy())
        except Exception as e:
            print(type(e).__name__)

f("/CPU:0")
f("/GPU:0")
```

## TensorFlow #97102
confirmation: CPU matrix rank disagrees with GPU and NumPy for a rank-1 matrix.

```python
import tensorflow as tf

x = tf.ones((32, 53), tf.float64) * -88917319269045.0
with tf.device("/CPU:0"):
    print(tf.linalg.matrix_rank(x, tol=6.0).numpy())
with tf.device("/GPU:0"):
    print(tf.linalg.matrix_rank(x, tol=6.0).numpy())
```

## TensorFlow #106602
confirmation: CPU propagates NaN in segment max; GPU drops it and returns the initialization value.

```python
import numpy as np
import tensorflow as tf

x = tf.constant([7.0, np.nan, -2.0], tf.float32)
ids = tf.constant([0, 4, 5], tf.int32)
with tf.device("/CPU:0"):
    print(tf.math.unsorted_segment_max(x, ids, 9).numpy())
with tf.device("/GPU:0"):
    print(tf.math.unsorted_segment_max(x, ids, 9).numpy())
```

## PyTorch #165650
confirmation: CPU rejects integer division by zero; CUDA returns values.

```python
import torch

def f(device):
    try:
        print(torch.remainder(torch.tensor([-5, 7], dtype=torch.int32, device=device), torch.tensor([0, 0], dtype=torch.int32, device=device)))
    except Exception as e:
        print(type(e).__name__)

f("cpu")
f("cuda")
```

## PyTorch #171356
confirmation: CPU rejects out-of-range float16 scalar; CUDA accepts it.

```python
import torch

def f(device):
    try:
        print(torch.clip(torch.zeros(2, dtype=torch.float16, device=device), min=-70000.0))
    except Exception as e:
        print(type(e).__name__)

f("cpu")
f("cuda")
```

## PyTorch #171536
confirmation: CPU handles empty input; CUDA errors with stride overflow.

```python
import torch

def f(device):
    try:
        print(torch.nn.PixelUnshuffle(2305843009213693952)(torch.zeros((0, 0, 0), device=device)))
    except Exception as e:
        print(type(e).__name__)

f("cpu")
f("cuda")
```

## PyTorch #154312
confirmation: singular logdet should be `-inf`; CUDA returns finite value.

```python
import torch

x = torch.tensor([[2.0, 4.0, 6.0], [4.0, 10.0, 12.0], [6.0, 12.0, 18.0]])
print(x.logdet())
print(x.cuda().logdet().cpu())
```

## PyTorch #158428
confirmation: CPU rejects singular Cholesky inverse; CUDA returns NaNs.

```python
import torch

x = torch.zeros((2, 2), dtype=torch.complex128)
def f(t):
    try:
        print(torch.cholesky_inverse(t))
    except Exception as e:
        print(type(e).__name__)

f(x)
f(x.cuda())
```

## PyTorch #145837
confirmation: backward returns invalid jagged gradient shape.

```python
import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.randn(1, 512, 16))
    def forward(self, x):
        for _ in range(10):
            y = torch.nested.to_padded_tensor(x, 0.0) * self.p
            x = torch.nested.narrow(y, 1, 0, x.offsets().diff(), layout=torch.jagged).contiguous()
        return x

x = torch.nested.nested_tensor([torch.randn(512 - i, 16) for i in range(4)], device="cuda", layout=torch.jagged)
M().cuda()(x).mean().backward()
```

## PyTorch #154606
confirmation: complex cumulative product with infinity gives incompatible NaN/Inf patterns.

```python
import torch

x = torch.tensor([complex(2, float("inf")), complex(-3, 4)], dtype=torch.complex128)
print(torch.cumprod(x, 0))
print(torch.cumprod(x.cuda(), 0).cpu())
```

## PyTorch #154474
confirmation: FFT with infinite input gives incompatible NaN/Inf patterns across CPU/CUDA.

```python
import torch

x = torch.tensor([torch.inf, -2.0, 3.0, -torch.inf])
print(torch.fft.fft(x))
print(torch.fft.fft(x.cuda()).cpu())
```

## PyTorch #156020
confirmation: IFFT overflow gives incompatible NaN/Inf patterns across CPU/CUDA.

```python
import torch

x = torch.zeros(8, dtype=torch.cfloat) + 1j * 1.0e38
print(torch.fft.ifft(x))
print(torch.fft.ifft(x.cuda()).cpu())
```

## PyTorch #158172
confirmation: CPU and CUDA choose different values before writing to an overflowing `out` dtype.

```python
import torch

def f(device):
    out = torch.tensor([], dtype=torch.int16, device=device)
    torch.fmin(torch.tensor([-3000000], dtype=torch.int64, device=device), torch.tensor([-2], device=device), out=out)
    print(out.cpu())

f("cpu")
f("cuda")
```

## PyTorch #158419
confirmation: `copysign` must copy NaN sign bit; CUDA ignores the negative NaN sign.

```python
import torch

x = torch.full((2, 2), 2.0, dtype=torch.float16)
y = torch.tensor([float("nan"), -float("nan")], dtype=torch.float16)
print(torch.copysign(x, y))
print(torch.copysign(x.cuda(), y.cuda()).cpu())
```

## PyTorch #158412
confirmation: robust complex absolute should not overflow for `(2, max_float)`; CUDA returns `inf`.

```python
import torch

x = torch.complex(torch.full((2, 2), 2.0, dtype=torch.float64), torch.full((2, 2), torch.finfo(torch.float64).max, dtype=torch.float64))
print(torch.absolute(x))
print(torch.absolute(x.cuda()).cpu())
```

## PyTorch #159870
confirmation: sigmoid at complex real part `-inf` should tend to zero; CPU returns NaN.

```python
import torch

x = torch.tensor(complex(-float("inf"), 2.0))
print(torch.nn.functional.sigmoid(x))
print(torch.nn.functional.sigmoid(x.cuda()).cpu())
```

## PyTorch #153358
confirmation: qint32 dequantization overflows on CPU but not CUDA.

```python
import torch

q = torch._make_per_tensor_quantized_tensor(torch.tensor([2147483646, 2147483647], dtype=torch.int32), scale=1e-10, zero_point=-2147483648)
print(torch.dequantize(q))
print(torch.dequantize(q.cuda()).cpu())
```

## PyTorch #146570
confirmation: geometric distribution is positive-valued; CUDA produces `-inf`.

```python
import torch

print(torch.full((1, 1), float("inf"), dtype=torch.float16).geometric_(5e-9))
print(torch.full((1, 1), float("inf"), dtype=torch.float16, device="cuda").geometric_(5e-9).cpu())
```
