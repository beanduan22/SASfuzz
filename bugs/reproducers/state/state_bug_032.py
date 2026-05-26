import torch


def run():
    assert torch.cuda.is_available(), 'CUDA is required'
    with torch.no_grad():
        x_cpu = torch.tensor([1, 2, 3, 4, 5, 6, 7])
        y_cpu = torch.tensor([0, 1, 1, 2])
        z_cpu = torch.tensor([12, 14, 16, 18, 20])
        x_gpu = x_cpu.cuda()
        y_gpu = y_cpu.cuda()
        z_gpu = z_cpu.cuda()
        cpu = torch.Tensor.scatter(x_cpu, 0, y_cpu, z_cpu)
        gpu = torch.Tensor.scatter(x_gpu, 0, y_gpu, z_gpu).cpu()
    ok = not torch.equal(cpu, gpu)
    print(f'state=gradient_tracking(torch.no_grad) cpu={cpu.tolist()} gpu={gpu.tolist()}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
