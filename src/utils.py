import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reshape_by_args(t, args, repeat=False):
    """Reshapes tensor into form specified by training arguments.

    Assumes tensor is of shape (BMK x L x D) or (BMK x H) where:
        B = batch size, M = number of ELBO samples, K = number of
        importance samples, L = sequence length, H = latent dimension,
        and D = data dimension.

    If repeat is true, this function first expands the input tensor into the
    above space before reshaping.

    Outputs tensors in the shape of (B x M x K x L x D) or (B x M x K x H).

    Args:
        t (torch.Tensor): Tensor to be reshaped.
        args (dict): Argument dictionary. Relevant keys are:
            args['M'] (int): Number of ELBO samples.
            args['K'] (int): Number of importance samples.
        repeat (boolean): If true, first performs sample repeating.
    """
    n_samp = args['M'] * args['K']
    if repeat:
        t = torch.repeat_interleave(t, n_samp, 0)

    return t.view(t.shape[0] // n_samp, args['M'], args['K'], *t.shape[1:])


def reshape_all_by_args(*t_arr, args):
    out = []
    for t in t_arr:
        out.append(reshape_by_args(t, args))
    return out


def mean_samples_all(*t_arr):
    out = []
    for t in t_arr:
        out.append(torch.mean(torch.mean(t, 1), 1))
    return out


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
