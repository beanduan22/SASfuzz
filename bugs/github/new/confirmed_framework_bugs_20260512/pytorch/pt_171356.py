import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

def f(device):
    try:
        print(torch.clip(torch.zeros(2, dtype=torch.float16, device=device), min=-70000.0))
    except Exception as e:
        print(type(e).__name__)

f("cpu")
f("cuda")
