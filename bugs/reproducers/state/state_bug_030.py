import torch
assert torch.cuda.is_available(), 'CUDA is required'
with torch.no_grad():
    cpu = torch.linspace(3.7, -3, 10, dtype=torch.int64, device='cpu')
    gpu = torch.linspace(3.7, -3, 10, dtype=torch.int64, device='cuda').cpu()
print(f'state=gradient_tracking(torch.no_grad) cpu={cpu.tolist()} gpu={gpu.tolist()}')
