import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

q = torch._make_per_tensor_quantized_tensor(torch.tensor([2147483646, 2147483647], dtype=torch.int32), scale=1e-10, zero_point=-2147483648)
print(torch.dequantize(q))
print(torch.dequantize(q.cuda()).cpu())
