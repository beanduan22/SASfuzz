import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

x = torch.zeros((2, 2), dtype=torch.complex128)
def f(t):
    try:
        print(torch.cholesky_inverse(t))
    except Exception as e:
        print(type(e).__name__)

f(x)
f(x.cuda())
