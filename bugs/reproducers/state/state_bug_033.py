import torch
import numpy as np


def _max_abs_diff(a, b):
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        diff = np.abs(aa - bb)
    if diff.size == 0:
        return 0.0
    if np.all(np.isnan(diff)):
        return float("nan")
    return float(np.nanmax(diff))


def run():
    assert torch.cuda.is_available(), 'CUDA is required'
    torch.manual_seed(0)
    with torch.no_grad():
        n = 1000000
        x = 1.0 + 0.0001 * torch.randn(n, dtype=torch.float32)
        ref = np.cumprod(x.numpy().astype(np.float64))
        cpu = torch.cumprod(x, dim=0).numpy()
        gpu = torch.cumprod(x.cuda(), dim=0).cpu().numpy()
    cpu_err = float(np.max(np.abs(cpu - ref)))
    gpu_err = float(np.max(np.abs(gpu - ref)))
    ok = _max_abs_diff(cpu, gpu) > 0.001
    print(f'state=gradient_tracking(torch.no_grad) cpu_err={cpu_err:.4e} gpu_err={gpu_err:.4e}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
