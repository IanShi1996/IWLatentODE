import math
import torch

from base_model import LatentODE, log_normal_pdf, normal_kl
from utils import view_with_mk


class CIWLatentODE(LatentODE):
    def __init__(self, enc, nodef, dec):
        super().__init__(enc, nodef, dec)

    def forward(self, x, ts, args):
        M, K = args['M'], args['K']
        qz0_mean, qz0_logvar = self.get_latent_initial_state(x, ts)

        qz0_mean = torch.repeat_interleave(qz0_mean, M * K, 0)
        qz0_logvar = torch.repeat_interleave(qz0_logvar, M * K, 0)

        z0, eps = self.reparameterize(qz0_mean, qz0_logvar)

        pred_z = self.generate_from_latent(z0, ts, args['model_rtol'],
                                           args['model_atol'], args['method'])
        pred_x = self.dec(pred_z)

        pred_x = view_with_mk(pred_x, M, K)
        z0 = view_with_mk(z0, M, K)
        qz0_mean = view_with_mk(qz0_mean, M, K)
        qz0_logvar = view_with_mk(qz0_logvar, M, K)
        eps = view_with_mk(eps, M, K)

        return pred_x, z0, qz0_mean, qz0_logvar, eps

    def get_elbo(self, x, pred_x, z0, qz0_mean, qz0_logvar, eps, args):
        iwae_elbo = self.get_iwae_elbo(x, pred_x, z0, qz0_mean, qz0_logvar,
                                       eps, args)

        # Compress K dimension
        pred_x = torch.mean(pred_x, 2)
        z0 = torch.mean(pred_x, 2)
        qz0_mean = torch.mean(qz0_mean, 2)
        qz0_logvar = torch.mean(qz0_logvar, 2)
        eps = torch.mean(eps, 2)

        normal_elbo = self.get_standard_elbo(x, pred_x, z0, qz0_mean,
                                             qz0_logvar, eps, args)

        return args['beta'] * normal_elbo + (1 - args['beta']) * iwae_elbo

    def get_standard_elbo(self, x, pred_x, z0, qz0_mean, qz0_logvar, eps, args):
        x = torch.repeat_interleave(x, args['M'], 0)
        x = x.view((x.shape[0] // args['M'], args['M'], *x.shape[1:]))

        noise_std_ = torch.zeros(pred_x.size(), device=x.device) + args['l_std']
        noise_logvar = 2. * torch.log(noise_std_)

        logpx = log_normal_pdf(x, pred_x, noise_logvar).sum(-1).sum(-1)

        pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size(), device=x.device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)

        return torch.mean(-logpx + analytic_kl)

    def get_iwae_elbo(self, x, pred_x, z0, qz0_mean, qz0_logvar, eps, args):
        """

        Args:
            x: B x L x D
            pred_x: B x M x K x L x D
            z0: B x M x K x H
            qz0_mean: B x M x K x H
            qz0_logvar: B x M x K x H
            eps: B x M x K x H
            args: float

        Returns:

        """
        x = torch.repeat_interleave(x, args['M'] * args['K'], 0)
        x = view_with_mk(x, args['M'], args['K'])

        noise_std_ = torch.zeros(pred_x.size(), device=x.device) + args['l_std']
        noise_logvar = 2. * torch.log(noise_std_)

        data_ll = log_normal_pdf(x, pred_x, noise_logvar).sum(-1).sum(-1)

        const = -0.5 * torch.log(torch.Tensor([2. * math.pi]).to(x.device))
        prior_z = torch.sum(const - 0.5 * z0 ** 2, -1)

        var_ll = torch.sum(const - 0.5 * qz0_logvar - 0.5 * eps ** 2, -1)

        unnorm_weight = data_ll + prior_z - var_ll
        unnorm_weight_detach = unnorm_weight.detach()

        total_weight = torch.logsumexp(unnorm_weight_detach, -1).unsqueeze(-1)
        log_norm_weight = unnorm_weight_detach - total_weight

        iwae_elbo = torch.sum(torch.exp(log_norm_weight) * unnorm_weight, -1)
        iwae_elbo = -torch.mean(iwae_elbo)

        return iwae_elbo
