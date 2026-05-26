import torch


def run():
    assert torch.cuda.is_available(), 'CUDA is required'
    torch.manual_seed(0)
    x = torch.randn(32, 32)
    with torch.no_grad():
        x_cpu = x.clone()
        x_gpu = x.clone().cuda()
        cpu = torch.mm(x_cpu, x_cpu, out=x_cpu)
        gpu = torch.mm(x_gpu, x_gpu, out=x_gpu).cpu()
    close = torch.allclose(cpu, gpu, atol=1e-05, rtol=1e-05, equal_nan=True)
    diff = float((cpu - gpu).abs().max().item())
    print(f'state=gradient_tracking(torch.no_grad) allclose={close} max_diff={diff:.4e}')
    print('BUG_REPRODUCED' if not close else 'NOT_REPRODUCED')
    return


run()
