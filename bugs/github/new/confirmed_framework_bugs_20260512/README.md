# Confirmed Framework Bugs: Repro Scripts and Outputs

Environment: TensorFlow 2.21.0, PyTorch 2.11.0+cu128.

Total scripts: 30

## TensorFlow #106388

File: `tensorflow/tf_106388.py`

Output:
```text
InvalidArgumentError
SparseFillEmptyRowsGrad(d_values=<tf.Tensor: shape=(2,), dtype=int64, numpy=array([0, 0])>, d_default_value=<tf.Tensor: shape=(), dtype=int64, numpy=12>)
```

## TensorFlow #106602

File: `tensorflow/tf_106602.py`

Output:
```text
[ 7.0000000e+00 -3.4028235e+38 -3.4028235e+38 -3.4028235e+38
            nan -2.0000000e+00 -3.4028235e+38 -3.4028235e+38
 -3.4028235e+38]
[ 7.0000000e+00 -3.4028235e+38 -3.4028235e+38 -3.4028235e+38
 -3.4028235e+38 -2.0000000e+00 -3.4028235e+38 -3.4028235e+38
 -3.4028235e+38]
```

## TensorFlow #93162

File: `tensorflow/tf_93162.py`

Output:
```text
[b'-9223372036854775808_X_5' b'2_X_6']
[b'9223372036854775807_X_5' b'2_X_6']
```

## TensorFlow #94119

File: `tensorflow/tf_94119.py`

Output:
```text
Process terminated by signal 8 (SIGFPE)
```

## TensorFlow #94151

File: `tensorflow/tf_94151.py`

Output:
```text
Process terminated by signal 6 (SIGABRT)
```

## TensorFlow #94376

File: `tensorflow/tf_94376.py`

Output:
```text
Process terminated by signal 6 (SIGABRT)
```

## TensorFlow #94378

File: `tensorflow/tf_94378.py`

Output:
```text
InvalidArgumentError
[0]
```

## TensorFlow #94655

File: `tensorflow/tf_94655.py`

Output:
```text
[np.float32(-0.0), np.float32(-inf)]
[np.float32(1.0), np.float32(-13.052013)]
```

## TensorFlow #94657

File: `tensorflow/tf_94657.py`

Output:
```text
InvalidArgumentError
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

## TensorFlow #96180

File: `tensorflow/tf_96180.py`

Output:
```text
[[nan+nanj]
 [nan+nanj]]
[[ 0.+0.j]
 [-0.-0.j]]
```

## TensorFlow #97042

File: `tensorflow/tf_97042.py`

Output:
```text
[[[[  4464]
   [ 24464]]

  [[-16608]
   [ 23392]]]]
[[[[32767]
   [32767]]

  [[   -2]
   [   -2]]]]
```

## TensorFlow #97102

File: `tensorflow/tf_97102.py`

Output:
```text
3
1
```

## TensorFlow #97125

File: `tensorflow/tf_97125.py`

Output:
```text
[-3898719529840714591]
[2209]
```

## TensorFlow #97204

File: `tensorflow/tf_97204.py`

Output:
```text
True
InvalidArgumentError
```

## TensorFlow #98410

File: `tensorflow/tf_98410.py`

Output:
```text
[inf inf]
[nan nan]
```

## PyTorch #145837

File: `pytorch/pt_145837.py`

Output:
```text
RuntimeError: Function CloneBackward0 returned an invalid gradient at index 0 - got [4, j21, 16] but expected shape compatible with [4, j20, 16]
```

## PyTorch #146570

File: `pytorch/pt_146570.py`

Output:
```text
tensor([[inf]], dtype=torch.float16)
tensor([[-inf]], dtype=torch.float16)
```

## PyTorch #153358

File: `pytorch/pt_153358.py`

Output:
```text
tensor([-2.0000e-10, -1.0000e-10])
tensor([0.4295, 0.4295])
```

## PyTorch #154312

File: `pytorch/pt_154312.py`

Output:
```text
tensor(-inf)
tensor(-13.0520)
```

## PyTorch #154474

File: `pytorch/pt_154474.py`

Output:
```text
tensor([nan+0.j, inf-infj, inf+0.j, inf+infj])
tensor([nan+0.j, nan+nanj, inf+0.j, nan+nanj])
```

## PyTorch #154606

File: `pytorch/pt_154606.py`

Output:
```text
tensor([nan+infj, nan+nanj], dtype=torch.complex128)
tensor([2.+infj, -inf-infj], dtype=torch.complex128)
```

## PyTorch #156020

File: `pytorch/pt_156020.py`

Output:
```text
tensor([0.+infj, 0.+0.j, 0.+0.j, 0.+0.j, 0.+nanj, 0.+0.j, 0.+0.j, 0.+0.j])
tensor([nan+infj, 0.+0.j, 0.+0.j, 0.+0.j, nan+nanj, 0.+0.j, 0.+0.j, 0.+0.j])
```

## PyTorch #158172

File: `pytorch/pt_158172.py`

Output:
```text
tensor([14656], dtype=torch.int16)
tensor([-2], dtype=torch.int16)
```

## PyTorch #158412

File: `pytorch/pt_158412.py`

Output:
```text
tensor([[1.7977e+308, 1.7977e+308],
        [1.7977e+308, 1.7977e+308]], dtype=torch.float64)
tensor([[inf, inf],
        [inf, inf]], dtype=torch.float64)
```

## PyTorch #158419

File: `pytorch/pt_158419.py`

Output:
```text
tensor([[ 2., -2.],
        [ 2., -2.]], dtype=torch.float16)
tensor([[2., 2.],
        [2., 2.]], dtype=torch.float16)
```

## PyTorch #158428

File: `pytorch/pt_158428.py`

Output:
```text
_LinAlgError
tensor([[nan+nanj, nan+nanj],
        [nan+nanj, nan+nanj]], device='cuda:0', dtype=torch.complex128)
```

## PyTorch #159870

File: `pytorch/pt_159870.py`

Output:
```text
tensor(nan+nanj)
tensor(-0.+0.j)
```

## PyTorch #165650

File: `pytorch/pt_165650.py`

Output:
```text
RuntimeError
tensor([-1, -1], device='cuda:0', dtype=torch.int32)
```

## PyTorch #171356

File: `pytorch/pt_171356.py`

Output:
```text
RuntimeError
tensor([0., 0.], device='cuda:0', dtype=torch.float16)
```

## PyTorch #171536

File: `pytorch/pt_171536.py`

Output:
```text
tensor([], size=(0, 0, 0))
RuntimeError
```
