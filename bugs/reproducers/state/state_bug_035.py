import torch
import numpy as np


def run():
    assert torch.cuda.is_available(), 'CUDA is required'
    with torch.no_grad():
        mag = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float16)
        sign = torch.from_numpy(np.array([32256, 65024, 15360], dtype=np.uint16).view(np.float16))
        cpu = torch.copysign(mag, sign)
        gpu = torch.copysign(mag.cuda(), sign.cuda()).cpu()
    ok = not torch.equal(torch.signbit(cpu), torch.signbit(gpu))
    print(f'state=gradient_tracking(torch.no_grad) cpu={cpu} sign={torch.signbit(cpu)} gpu={gpu} sign={torch.signbit(gpu)}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
