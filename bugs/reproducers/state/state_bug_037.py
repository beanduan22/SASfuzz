import torch


def run():
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(123)
    t, n, c = (3, 1, 3)
    log_probs = F.log_softmax(torch.randn(t, n, c), dim=-1).double()
    log_probs.requires_grad_(True)
    targets = torch.tensor([[1]])
    input_lens = torch.tensor([t])
    target_lens = torch.tensor([1])
    loss_fn = nn.CTCLoss(blank=0)

    def fn(inp):
        return loss_fn(inp, targets, input_lens, target_lens)
    try:
        torch.autograd.gradcheck(fn, (log_probs,), raise_exception=True)
    except Exception as exc:
        print(f'state=gradient_tracking(torch.autograd.gradcheck) err={type(exc).__name__}: {str(exc)[:200]}')
        print('BUG_REPRODUCED' if True else 'NOT_REPRODUCED')
        return
    print('state=gradient_tracking(torch.autograd.gradcheck) gradcheck passed')
    print('BUG_REPRODUCED' if False else 'NOT_REPRODUCED')
    return


run()
