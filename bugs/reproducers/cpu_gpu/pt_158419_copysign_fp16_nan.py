                                                  
import numpy as np, torch

mag = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float16)
sgn = torch.from_numpy(np.array([0x7e00, 0xfe00, 0x3c00], dtype=np.uint16).view(np.float16))

cpu = torch.copysign(mag, sgn)
gpu = torch.copysign(mag.cuda(), sgn.cuda()).cpu()
print("cpu:", cpu)
print("gpu:", gpu)
