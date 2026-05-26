import torch


def run():
    assert torch.cuda.is_available(), 'CUDA is required'
    with torch.no_grad():
        cpu = torch.linspace(3.7, -3, 10, dtype=torch.int64, device='cpu')
        gpu = torch.linspace(3.7, -3, 10, dtype=torch.int64, device='cuda').cpu()
    ok = not torch.equal(cpu, gpu)
    print(f'state=gradient_tracking(torch.no_grad) cpu={cpu.tolist()} gpu={gpu.tolist()}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
