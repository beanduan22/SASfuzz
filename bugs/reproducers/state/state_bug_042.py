import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'
torch.manual_seed(0)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sens = nn.LazyBatchNorm1d()

    def forward(self, x):
        h = x.view(x.size(0), -1)
        return self.sens(h)

x = torch.rand(2, 3, 32, 32)
model = Model()
model.train()
y_train = model(x)
model.eval()
y_eval = model.cuda()(x.cuda()).cpu()

print('Train:', y_train.flatten()[:5])
print('Eval:', y_eval.flatten()[:5])
print('Max diff:', float(torch.nan_to_num((y_train - y_eval).abs(), nan=0.0).max().item()))
