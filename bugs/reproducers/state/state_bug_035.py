import torch
import torch.nn as nn
import numpy as np

assert torch.cuda.is_available(), 'CUDA is required'

class Model(nn.Module):
    def forward(self, x):
        mag, sign = x
        h = torch.copysign(mag, sign)
        return h

model = Model()
mag = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float16)
sign = torch.from_numpy(np.array([0x7E00, 0xFE00, 0x3C00], dtype=np.uint16).view(np.float16))
with torch.no_grad():
    y_cpu = model((mag, sign))
    y_gpu = model((mag.cuda(), sign.cuda())).cpu()

print('CPU:', y_cpu, torch.signbit(y_cpu))
print('CUDA:', y_gpu, torch.signbit(y_gpu))
