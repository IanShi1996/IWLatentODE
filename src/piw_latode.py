import torch

from base_model import LatentODE
from estimators import get_miw_elbo
from utils import reshape_by_args, reshape_all_by_args


class PIWLatentODE(LatentODE):
    """
    Partially Importance-Weighted Latent ODE. Optimizies recognition network
    using MIW ELBO, and optimizes the generative network using the IW
    ELBO. This mitigates effects of importance samples.
    See: https://arxiv.org/pdf/1802.04537.pdf.

    WARNING: Does NOT integrate with normal training pipeline.
    """
    def __init__(self, enc, nodef, dec):
        super().__init__(enc, nodef, dec)

    def forward(self, x, ts, args):
        qz0_mean, qz0_logvar = self.get_latent_initial_state(x, ts)

        n_samp = args['M'] * args['K']
        qz0_mean = torch.repeat_interleave(qz0_mean, n_samp, 0)
        qz0_logvar = torch.repeat_interleave(qz0_logvar, n_samp, 0)

        z0, _ = self.reparameterize(qz0_mean, qz0_logvar)

        pred_z = self.generate_from_latent(z0, ts, args['model_rtol'],
                                           args['model_atol'], args['method'])
        pred_x = self.dec(pred_z)

        return pred_x, z0, qz0_mean, qz0_logvar

    def get_elbo(self, x, pred_x, z0, qz0_mean, qz0_logvar, args):
        miw_x = reshape_by_args(x, args, repeat=True)
        miw_in = reshape_all_by_args(pred_x, z0, qz0_mean, qz0_logvar,
                                     args=args)
        miw_elbo = get_miw_elbo(miw_x, *miw_in, args=args)

        n_samp = args['M'] * args['K']
        iw_args = {'M': 1, 'K': n_samp, 'l_std': args['l_std']}

        iw_x = reshape_by_args(x, iw_args, repeat=True)
        iw_in = reshape_all_by_args(pred_x, z0, qz0_mean, qz0_logvar,
                                    args=iw_args)
        iw_elbo = get_miw_elbo(iw_x, *iw_in, args=iw_args)

        return iw_elbo, miw_elbo
