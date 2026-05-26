import torch
assert torch.cuda.is_available(), 'CUDA is required'
import torch.nn as nn

class Model(nn.Module):

    def forward(self, x):
        y = x.view(x.size(0), -1)
        return torch.matrix_exp(y)
inp = torch.tensor([[1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 10.0, 200.0]])
model = Model()
traced_cpu = torch.jit.trace(model, inp)
traced_gpu = torch.jit.trace(model.cuda(), inp.cuda())
cpu = traced_cpu(inp)
gpu = traced_gpu(inp.cuda()).cpu()
print(f'state=execution_mode(torch.jit.trace) cpu={cpu} gpu={gpu}')
