import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
