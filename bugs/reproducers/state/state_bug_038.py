import torch


def run():
    assert torch.cuda.is_available(), 'CUDA is required'
    with torch.no_grad():
        x = torch.tensor([-0.0])
        cpu = torch.clamp(x, min=0, max=1)
        gpu = torch.clamp(x.cuda(), min=0, max=1).cpu()
    ok = bool(torch.signbit(cpu).item()) != bool(torch.signbit(gpu).item())
    print(f'state=gradient_tracking(torch.no_grad) cpu={cpu} sign={torch.signbit(cpu)} gpu={gpu} sign={torch.signbit(gpu)}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
