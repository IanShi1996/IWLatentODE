import math
import torch


def log_normal_pdf(x, mean, logvar):
    """Compute log PDF of x under Gaussian parameterized by mean and logvar.

    Args:
        x (torch.Tensor): Observed data points.
        mean (torch.Tensor): Mean of gaussian distribution.
        logvar (torch.Tensor): Log variance of gaussian distribution.

    Returns:
        torch.Tensor: Log PDF of data under specified gaussian.
    """
    const = -0.5 * torch.log(torch.Tensor([2. * math.pi])).to(x.device)
    return const - 0.5 * logvar - 0.5 * ((x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    """Compute analytic KL divergence between two gaussian distributions.

    Computes analytic KL divergence between two gaussians parameterized by given
     mean and variances. All inputs must have the same dimension.
     Implementation from: https://github.com/rtqichen/torchdiffeq.

    Args:
        mu1 (torch.Tensor): Mean of first gaussian distribution.
        lv1 (torch.Tensor): Log variance of first gaussian distribution.
        mu2 (torch.Tensor): Mean of second gaussian distribution.
        lv2 (torch.Tensor): Log variance of second gaussian distribution.

    Returns:
        torch.Tensor: Analytic KL divergence between given distributions.
    """
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def get_analytic_elbo(x, pred_x, qz0_mean, qz0_logvar, args):
    """Compute VAE ELBO using analytic KL divergence.

    Inputs expected to be in shape:
        x: B x L x D
        pred_x: B x L x D
        z0: B x H
        qz0_mean: B x H
        qz0_logvar: B x H
    where
        B = Batch size
        L = Sequence length
        D = Data dimension
        H = Latent dimension

    Args:
        x (torch.Tensor): Observed data points.
        pred_x (torch.Tensor): Data points predicted by model.
        qz0_mean (torch.Tensor): Variational posterior means.
        qz0_logvar (torch.Tensor): Log of variational posterior variances.
        args (dict): Argument dictionary. Relevant keys are:
            args['l_std'] (float): Fixed standard deviation for likelihood.
            args['beta'] (float): KL Divergence weight. Used for beta VAE.

    Returns:
        torch.Tensor: ELBO computed using analytic KL.
    """
    noise_std_ = torch.zeros(x.size(), device=x.device) + args['l_std']
    noise_logvar = 2. * torch.log(noise_std_)

    logpx = log_normal_pdf(x, pred_x, noise_logvar).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size(), device=x.device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1)

    beta = 1 if 'beta' not in args else args['beta']
    return torch.mean(-logpx + beta * analytic_kl, dim=0)


def get_miw_elbo(x, pred_x, z0, qz0_mean, qz0_logvar, args):
    """Computes ELBO with using multiple samples.

    Takes args['M'] ELBO estimates each computing using args['K'] importance
    weights. Can be directly used for MIW ELBO. Set args['M'] = 1 for IW
    ELBO, and args['K'] = 1 for VAE ELBO.

    Inputs must be in the shape:
        x: B x M x K x L x D
        pred_x: B x M x K x L x D
        z0: B x M x K x H
        qz0_mean: B x M x K x H
        qz0_logvar: B x M x K x H
    where
        B = Batch size
        M = Number of ELBO samples
        K = Number of importance samples
        L = Sequence length
        D = Data dimension
        H = Latent dimension

    Args:
        x (torch.Tensor): Observed data points.
        pred_x (torch.Tensor): Data points predicted by model.
        z0 (torch.Tensor): Sampled latent initial states.
        qz0_mean (torch.Tensor): Variational posterior means.
        qz0_logvar (torch.Tensor): Log of variational posterior variances.
        args (dict): Argument dictionary. Relevant keys are:
            args['l_std'] (float): Fixed standard deviation for likelihood.
            args['K'] (int): Number of importance samples.

    Returns:
        torch.Tensor: The multiply importance-weighted elbo.
    """
    noise_std_ = torch.zeros(x.size()).to(x.device) + args['l_std']
    noise_logvar = 2. * torch.log(noise_std_)

    zero_mean = torch.zeros_like(z0, device=x.device)
    one_logvar = torch.ones_like(z0, device=x.device)

    log_pxCz0 = log_normal_pdf(x, pred_x, noise_logvar).sum(-1).sum(-1)
    log_pz0 = log_normal_pdf(z0, zero_mean, one_logvar).sum(-1)
    log_qz0Cx = log_normal_pdf(z0, qz0_mean, qz0_logvar).sum(-1)

    log_iw = log_pxCz0 + log_pz0 - log_qz0Cx

    iw_elbo = torch.logsumexp(log_iw, -1)
    k_const = torch.ones_like(iw_elbo, device=x.device) * args['K']
    miw_elbo = -1 * torch.mean(iw_elbo - torch.log(k_const))

    return miw_elbo


def get_ciw_elbo(x, pred_x, z0, qz0_mean, qz0_logvar, args):
    """ Compute combination importance weighted ELBO.

    Inputs must be in the shape:
        x: B x M x K x L x D
        pred_x: B x M x K x L x D
        z0: B x M x K x H
        qz0_mean: B x M x K x H
        qz0_logvar: B x M x K x H
    where
        B = Batch size
        M = Number of ELBO samples
        K = Number of importance samples
        L = Sequence length
        D = Data dimension
        H = Latent dimension

    Args:
        x (torch.Tensor): Observed data points.
        pred_x (torch.Tensor): Data points predicted by model.
        z0 (torch.Tensor): Sampled latent initial states.
        qz0_mean (torch.Tensor): Variational posterior means.
        qz0_logvar (torch.Tensor): Log of variational posterior variances.
        args (dict): Argument dictionary. Relevant keys are:
            args['l_std'] (float): Fixed standard deviation for likelihood.
            args['M'] (int): Number of ELBO samples.
            args['K'] (int): Number of importance samples.
            args['beta'] (float): Combination parameter.

    Returns:
        torch.Tensor: The combination importance-weighted elbo.
    """
    if 'beta' not in args or args['beta'] > 1 or args['beta'] < 0:
        raise ValueError("Invalid beta parameter. Valid only between [0, 1].")

    total_samp = args['M'] * args['K']

    var_args = {'M': total_samp, 'K': 1, 'l_std': args['l_std']}
    vae_elbo = get_miw_elbo(x, pred_x, z0, qz0_mean, qz0_logvar, var_args)

    iw_args = {'M': 1, 'K': total_samp, 'l_std': args['l_std']}
    iw_elbo = get_miw_elbo(x, pred_x, z0, qz0_mean, qz0_logvar, iw_args)

    return args['beta'] * vae_elbo + (1 - args['beta']) * iw_elbo
