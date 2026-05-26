import torch


def run():
    assert torch.cuda.is_available(), 'CUDA is required'
    import torch.nn as nn

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.avg_pool1d = nn.AvgPool1d(kernel_size=2)
            self.channel_shuffle = nn.ChannelShuffle(groups=3)

        def forward(self, x):
            x = x.view(-1, 3, 32 * 32)
            x = self.avg_pool1d(x)
            return self.channel_shuffle(x)
    model = Model()
    model.train()
    model.eval()
    x = torch.rand(2, 3, 32, 32)
    cpu_err = gpu_err = None
    try:
        cpu = model.cpu()(x)
    except Exception as exc:
        cpu = None
        cpu_err = type(exc).__name__ + ': ' + str(exc)[:120]
    try:
        gpu = model.cuda()(x.cuda())
    except Exception as exc:
        gpu = None
        gpu_err = type(exc).__name__ + ': ' + str(exc)[:160]
    ok = cpu is not None and gpu_err is not None and ('channel_shuffle' in gpu_err)
    print(f'state=execution_mode(train/eval switch) cpu_shape={(None if cpu is None else tuple(cpu.shape))} cpu_err={cpu_err} gpu_err={gpu_err}')
    print('BUG_REPRODUCED' if ok else 'NOT_REPRODUCED')
    return


run()
