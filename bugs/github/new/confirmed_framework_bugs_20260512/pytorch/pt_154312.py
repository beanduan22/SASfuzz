import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

x = torch.tensor([[2.0, 4.0, 6.0], [4.0, 10.0, 12.0], [6.0, 12.0, 18.0]])
print(x.logdet())
print(x.cuda().logdet().cpu())
