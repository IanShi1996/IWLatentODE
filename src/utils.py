import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def view_with_k(t, k):
    # TODO: Merge with view with mk
    return t.view(t.shape[0] // k, k, *t.shape[1:])


def view_with_mk(t, m, k):
    return t.view(t.shape[0] // (m * k), m, k, *t.shape[1:])


def gpu(t, device=device):
    return torch.Tensor(t).float().to(device)


def asnp(t):
    return t.detach().cpu().numpy()


class RunningAverageMeter(object):
    """Compute and stores the average and current value.
    This implementation was taken from the original Neural ODE
    repository: https://github.com/rtqichen/torchdiffeq.
    """

    def __init__(self, momentum=0.99):
        """Initialize RunningAverageMeter.
        Args:
            momentum (float, optional): Momentum coefficient. Defaults to 0.99.
        """
        self.momentum = momentum
        self.reset()

    def reset(self):
        """Reset running average to zero."""
        self.val = None
        self.avg = 0

    def update(self, val):
        """Update running average with new value.
        Args:
            val (float): New value.
        """
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
