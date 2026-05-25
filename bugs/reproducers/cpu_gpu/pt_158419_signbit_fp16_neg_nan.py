                                                                               
import numpy as np, torch

x = torch.from_numpy(np.array([0xfe00, 0xfe00, 0xfe00], dtype=np.uint16).view(np.float16))

print("cpu:", torch.signbit(x))
print("gpu:", torch.signbit(x.cuda()).cpu())
