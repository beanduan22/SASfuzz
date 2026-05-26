import torch
assert torch.cuda.is_available(), 'CUDA is required'
x = torch.tensor([[0.01, 0.0, 0.0, 0.0, 0.1], [0.0, 0.01, 0.0, 0.1, 0.0], [0.0, 0.0, 0.01, 0.0, 0.0], [0.0, 0.1, 0.0, 0.01, 0.0], [0.1, 0.0, 0.0, 0.0, 0.01]])
with torch.no_grad():
    cpu_vals, cpu_vecs = torch.lobpcg(x)
    gpu_vals, gpu_vecs = torch.lobpcg(x.cuda())
diff = float((cpu_vecs.abs() - gpu_vecs.cpu().abs()).abs().max().item())
print(f'state=gradient_tracking(torch.no_grad) eig_cpu={cpu_vals} eig_gpu={gpu_vals.cpu()} vec_abs_diff={diff:.4e}')
