import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sens = nn.ChannelShuffle(groups=3)
        self.pool = nn.AvgPool1d(kernel_size=2)

    def forward(self, x):
        h = x.view(-1, 3, 32 * 32)
        h = self.pool(h)
        return self.sens(h)

model = Model()
x = torch.rand(2, 3, 32, 32)
model.train()
y_train = model.cpu()(x)
model.eval()
try:
    y_eval = model.cuda()(x.cuda())
except Exception as exc:
    y_eval = type(exc).__name__ + ': ' + str(exc).splitlines()[0]

print('Train:', tuple(y_train.shape))
print('Eval:', y_eval)
