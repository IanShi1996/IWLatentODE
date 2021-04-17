from miw_latode import MIWLatentODE
from estimators import get_ciw_elbo
from utils import reshape_by_args


class CIWLatentODE(MIWLatentODE):
    """
    Combination Imporatance-Weighted Latent ODE. Combines the VAE and IW ELBO
    into a single ELBO controlled by combination parameter beta. Intuitively,
    uses IW ELBO gradients unless they become extremely small.
    See: https://arxiv.org/pdf/1802.04537.pdf.
    """
    def __init__(self, enc, nodef, dec):
        super().__init__(enc, nodef, dec)

    def get_elbo(self, x, pred_x, z0, qz0_mean, qz0_logvar, args):
        x = reshape_by_args(x, args, repeat=True)
        return get_ciw_elbo(x, pred_x, z0, qz0_mean, qz0_logvar, args)
