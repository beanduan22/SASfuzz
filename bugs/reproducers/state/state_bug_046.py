import torch
import torch.nn as nn
torch.manual_seed(0)
fc1 = nn.Linear(8, 8)
fc2 = nn.Linear(8, 8)
bn = nn.BatchNorm1d(8)
x = torch.randn(4, 8, requires_grad=True)
with torch.enable_grad():
    y = fc1(x)
    y = torch.nn.functional.hardswish(y)
    y = bn(y)
    y = fc2(y)
    y = torch.log1p(y)
    out = torch.xlogy(y, y)
minimal = torch.xlogy(torch.tensor(0.0), torch.tensor(0.0))
print(f'state=gradient_tracking(torch.enable_grad) minimal_xlogy_0_0={minimal} any_model_nan={torch.isnan(out).any().item()}')
