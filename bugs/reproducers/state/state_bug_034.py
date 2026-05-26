import torch
assert torch.cuda.is_available(), 'CUDA is required'
with torch.no_grad():
    lhs = torch.tensor([100000], dtype=torch.int64)
    rhs = torch.tensor([-1], dtype=torch.int64)
    out_cpu = torch.empty(1, dtype=torch.uint8)
    out_cuda = torch.empty(1, dtype=torch.uint8, device='cuda')
    torch.fmax(lhs, rhs, out=out_cpu)
    torch.fmax(lhs.cuda(), rhs.cuda(), out=out_cuda)
    gpu = out_cuda.cpu()
print(f'state=gradient_tracking(torch.no_grad) cpu={out_cpu.tolist()} gpu={gpu.tolist()}')
