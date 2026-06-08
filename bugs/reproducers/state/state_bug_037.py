# Issue: https://github.com/pytorch/pytorch/issues/181534
# Status: fixed
# State: gradient tracking
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CTCLoss(blank=0)
        self.targets = torch.tensor([[1]])
        self.input_lens = torch.tensor([3])
        self.target_lens = torch.tensor([1])

    def forward(self, x):
        h = self.loss(x, self.targets, self.input_lens, self.target_lens)
        return h

torch.manual_seed(123)
model = Model()
x = F.log_softmax(torch.randn(3, 1, 3), dim=-1).double()
x.requires_grad_(True)
out = model(x)
out.sum().backward()
try:
    torch.autograd.gradcheck(lambda z: model(z), (x,), raise_exception=True)
    check = 'passed'
except Exception as exc:
    check = type(exc).__name__ + ': ' + str(exc).splitlines()[0]

print('Gradient:', x.grad)
print('Reference gradient:', check)
