import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

def f(device):
    try:
        print(torch.remainder(torch.tensor([-5, 7], dtype=torch.int32, device=device), torch.tensor([0, 0], dtype=torch.int32, device=device)))
    except Exception as e:
        print(type(e).__name__)

f("cpu")
f("cuda")
