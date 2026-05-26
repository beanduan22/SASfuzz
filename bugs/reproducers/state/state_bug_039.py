import torch
import numpy as np
assert torch.cuda.is_available(), 'CUDA is required'
src = torch.from_numpy(np.array([65024, 65024, 65024], dtype=np.uint16).view(np.float16))
with torch.no_grad():
    cpu = torch.signbit(src)
    gpu = torch.signbit(src.cuda()).cpu()
print(f'state=gradient_tracking(torch.no_grad) cpu={cpu.tolist()} gpu={gpu.tolist()}')
