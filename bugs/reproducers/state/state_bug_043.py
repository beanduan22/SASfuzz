import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        h = x.view(x.size(0), -1)
        return torch.matrix_exp(h)

model = Model()
x = torch.tensor([[1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 10.0, 200.0]])
y_eager = model(x)
traced = torch.jit.trace(model, x)
y_traced = traced(x)

print('Eager:', y_eager)
print('Traced:', y_traced)
