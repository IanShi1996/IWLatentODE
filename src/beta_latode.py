import math
import torch

from base_model import LatentODE, log_normal_pdf, normal_kl


class BetaLatentODE(LatentODE):
    def __init__(self, enc, nodef, dec):
        super().__init__(enc, nodef, dec)

    def forward(self, x, ts, args):
        qz0_mean, qz0_logvar = self.get_latent_initial_state(x, ts)

        z0, eps = self.reparameterize(qz0_mean, qz0_logvar)

        pred_z = self.generate_from_latent(z0, ts, args['model_rtol'],
                                           args['model_atol'], args['method'])
        pred_x = self.dec(pred_z)

        return pred_x, z0, qz0_mean, qz0_logvar, eps

    def get_elbo(self, x, pred_x, z0, qz0_mean, qz0_logvar, eps, args):
        noise_std_ = torch.zeros(pred_x.size(), device=x.device) + args['l_std']
        noise_logvar = 2. * torch.log(noise_std_)

        logpx = log_normal_pdf(x, pred_x, noise_logvar).sum(-1).sum(-1)

        pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size(), device=x.device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)

        # Overloaded beta, but oh well.
        return torch.mean(-logpx + args['beta'] * analytic_kl)
