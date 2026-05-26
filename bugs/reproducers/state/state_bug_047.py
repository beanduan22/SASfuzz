import torch
import torch.nn as nn

assert torch.cuda.is_available(), 'CUDA is required'

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sens = nn.ReLU()

    def forward(self, x):
        h = self.sens(x)
        return h

model = Model()
x = torch.tensor([-0.0])
model.train()
y_train = model(x)
model.eval()
y_eval = model(x.cuda()).cpu()

print('Train:', y_train, torch.signbit(y_train))
print('Eval:', y_eval, torch.signbit(y_eval))
