import torch

from base_model import LatentODE
from estimators import get_miw_elbo
from utils import reshape_all_by_args, reshape_by_args, mean_samples_all


class MIWLatentODE(LatentODE):
    """
    Multiply Importance-Weighted Latent ODE. Uses multiple
    estimates of ELBO made construction from importance-weighted
    samples to address gradient estimator accuracy issues caused by
    importance sampling. See: https://arxiv.org/pdf/1802.04537.pdf.
    """
    def __init__(self, enc, nodef, dec):
        super().__init__(enc, nodef, dec)

    def forward(self, x, ts, args, mean=False):
        """Compute MIWLatent ODE forward pass.

        Data is output in preparation for ELBO computation. First three
        dimensions of output Tensors are B x M x K, where B is batch size,
        M is number of ELBO samples, and K is number of importance samples.

        Alternatively, mean flag results in output of mean values for all
        outputs.

        Args:
            x (torch.Tensor): Observed data.
            ts (torch.Tensor): Times of observation.
            args (dict): Additional arguments.
                args['model_rtol'] (float): ODE solve relative tolerance.
                args['model_atol'] (float): ODE solve absolute tolerance.
                args['method'] (string): ODE solver.
                args['M'] (int): Number of ELBO samples.
                args['K'] (int): Number of importance samples.
            mean (boolean): Returns mean output if true.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
                Predicted data, latent initial states, variational posterior
                means, variational posterior log variances.
        """
        qz0_mean, qz0_logvar = self.get_latent_initial_state(x, ts)

        n_samp = args['M'] * args['K']
        qz0_mean = torch.repeat_interleave(qz0_mean, n_samp, 0)
        qz0_logvar = torch.repeat_interleave(qz0_logvar, n_samp, 0)

        z0, _ = self.reparameterize(qz0_mean, qz0_logvar)
        pred_z = self.generate_from_latent(z0, ts, args['model_rtol'],
                                           args['model_atol'], args['method'])
        pred_x = self.dec(pred_z)

        out = reshape_all_by_args(pred_x, z0, qz0_mean, qz0_logvar, args=args)
        out = mean_samples_all(*out) if mean else out

        return out

    def get_elbo(self, x, pred_x, z0, qz0_mean, qz0_logvar, args):
        x = reshape_by_args(x, args, repeat=True)
        return get_miw_elbo(x, pred_x, z0, qz0_mean, qz0_logvar, args)
