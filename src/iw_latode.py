import math
import torch

from base_model import LatentODE, log_normal_pdf
from utils import view_with_k


class IWLatentODE(LatentODE):

    def __init__(self, enc, nodef, dec):
        super().__init__(enc, nodef, dec)

    def reparameterize(self, qz0_mean, qz0_logvar, K):
        epsilon = torch.randn(qz0_mean.size(), device=qz0_mean.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        return z0, epsilon

    def forward(self, x, ts, M, K, rtol=1e-3, atol=1e-4, method='dopri5'):
        qz0_mean, qz0_logvar = self.get_latent_initial_state(x, ts)

        qz0_mean = torch.repeat_interleave(qz0_mean, K, 0)
        qz0_logvar = torch.repeat_interleave(qz0_logvar, K, 0)

        z0, eps = self.reparameterize(qz0_mean, qz0_logvar, K)

        pred_z = self.generate_from_latent(z0, ts, rtol, atol, method)
        pred_x = self.dec(pred_z)

        pred_x = view_with_k(pred_x, K)
        z0 = view_with_k(z0, K)
        qz0_mean = view_with_k(qz0_mean, K)
        qz0_logvar = view_with_k(qz0_logvar, K)
        eps = view_with_k(eps, K)

        return pred_x, z0, qz0_mean, qz0_logvar, eps

    def get_elbo(self, x, pred_x, z0, qz0_mean, qz0_logvar, eps, noise_std=0.1):
        """

        Args:
            x: B x L x D
            pred_x: B x K x L x D
            z0: B x K x H
            qz0_mean: B x K x H
            qz0_logvar: B x K x H
            eps: B x K x H
            noise_std: float

        Returns:

        """
        K = pred_x.size(1)
        x = torch.repeat_interleave(x, K, 0)
        x = view_with_k(x, K)

        noise_std_ = torch.zeros(pred_x.size(), device=x.device) + noise_std
        noise_logvar = 2. * torch.log(noise_std_)

        data_ll = log_normal_pdf(x, pred_x, noise_logvar).sum(-1).sum(-1)

        const = -0.5 * torch.log(torch.Tensor([2. * math.pi]).to(x.device))
        prior_z = torch.sum(const - 0.5 * z0 ** 2, -1)

        var_ll = torch.sum(const - 0.5 * qz0_logvar - 0.5 * eps ** 2, -1)

        unnorm_weight = data_ll + prior_z - var_ll
        unnorm_weight_detach = unnorm_weight.detach()

        total_weight = torch.logsumexp(unnorm_weight_detach, 1).unsqueeze(1)
        log_norm_weight = unnorm_weight_detach - total_weight

        iwae_elbo = torch.sum(torch.exp(log_norm_weight) * unnorm_weight, -1)
        iwae_elbo = -torch.mean(iwae_elbo)

        return iwae_elbo
