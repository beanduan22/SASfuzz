import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

def f(device):
    try:
        print(torch.nn.PixelUnshuffle(2305843009213693952)(torch.zeros((0, 0, 0), device=device)))
    except Exception as e:
        print(type(e).__name__)

f("cpu")
f("cuda")
