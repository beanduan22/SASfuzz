import torch
import numpy as np


def run():
    assert torch.cuda.is_available(), 'CUDA is required'
    src = torch.from_numpy(np.array([65024, 65024, 65024], dtype=np.uint16).view(np.float16))
    with torch.no_grad():
        cpu = torch.signbit(src)
        gpu = torch.signbit(src.cuda()).cpu()
    ok = not torch.equal(cpu, gpu)
    print(f'state=gradient_tracking(torch.no_grad) cpu={cpu.tolist()} gpu={gpu.tolist()}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
