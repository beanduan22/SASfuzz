import torch
assert torch.cuda.is_available(), 'CUDA is required'
torch.manual_seed(0)
with torch.no_grad():
    x = torch.randn(1000, dtype=torch.float32) * 1e+19 + 1e+20
    ref = torch.std(x.double()).item()
    cpu = torch.std(x).item()
    gpu = torch.std(x.cuda()).cpu().item()
print(f'state=gradient_tracking(torch.no_grad) ref={ref:.4e} cpu={cpu:.4e} gpu={gpu}')
