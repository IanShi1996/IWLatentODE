import numpy as np

import torch
import torch.nn as nn
from torchdiffeq import odeint as odeint


def log_normal_pdf(x, mean, logvar):
    """Compute log pdf of data under gaussian specified by parameters.
    Implementation taken from: https://github.com/rtqichen/torchdiffeq.
    Args:
        x (torch.Tensor): Observed data points.
        mean (torch.Tensor): Mean of gaussian distribution.
        logvar (torch.Tensor): Log variance of gaussian distribution.
    Returns:
        torch.Tensor: Log probability of data under specified gaussian.
    """
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)

    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    """Compute analytic KL divergence between two gaussian distributions.
    Computes analytic KL divergence between two multivariate gaussians which
    are parameterized by the given mean and variances. All inputs must have
    the same dimension.
    Implementation taken from: https://github.com/rtqichen/torchdiffeq.
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


class Swish(nn.Module):
    """Swish activation function.
    Implements swish activation function: https://arxiv.org/pdf/1710.05941.pdf.
    Claimed by NODE authors to perform well in NODEs.
    """

    def __init__(self):
        """Initialize swish activation function."""
        super(Swish, self).__init__()

    def forward(self, x, beta=1):
        """Compute swish forward pass.
        Args:
            x (torch.Tensor): Input data.
            beta (float, optional): Scaling factor. Defaults to 1.
        Returns:
            torch.Tensor: Data with swish non-linearity applied.
        """
        return x * torch.sigmoid(beta * x)


ACTIVATION_DICT = {
    'Swish': Swish,
    'Tanh': nn.Tanh,
    'ReLU': nn.ReLU,
    'Softplus': nn.Softplus
}


class GRU(nn.Module):
    """Gated Recurrent Unit.
    Implementation is borrowed from https://github.com/YuliaRubanova/latent_ode
    which in turn uses http://www.wildml.com/2015/10/recurrent-neural-network-
    tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
    """

    def __init__(self, latent_dim, input_dim, n_units=100):
        """Initilize GRU.
        Args:
            latent_dim (int): Dimension of latent state.
            input_dim (int): Dimension of input.
            n_units (int, optional): Number of GRU units.
        """
        super(GRU, self).__init__()

        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim * 2),
            nn.Sigmoid())
        self.init_network(self.update_gate)

        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim * 2),
            nn.Sigmoid())
        self.init_network(self.reset_gate)

        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim * 2))
        self.init_network(self.new_state_net)

    def forward(self, x, h):
        """Compute GRU forward pass.
        Args:
            x (torch.Tensor): Input date for specific timepoint.
            h (torch.Tensor): Previous hidden state.
        Returns:
            torch.Tensor: Updated hidden state.
        """
        input_concat = torch.cat([h, x], -1)

        update_gate = self.update_gate(input_concat)
        reset_gate = self.reset_gate(input_concat)

        concat = torch.cat([h * reset_gate, x], -1)

        new_state = self.new_state_net(concat)

        new_y = (1 - update_gate) * new_state + update_gate * h

        return new_y

    def init_network(self, net):
        """Initialize network using normal distribution.
        Args:
            net (nn.Module): NN to initialize.
        """
        for module in net.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.1)
                nn.init.constant_(module.bias, val=0)


class EncoderGRUODE(nn.Module):
    """GRU with hidden dynamics represented by Neural ODE.
    Implements the GRU-ODE model in: https://arxiv.org/abs/1907.03907.
    Observations are encoded by a RNN/GRU. Between observations, the hidden
    state is evolved using a Neural ODE.
    Attributes:
        gru (nn.Module): GRU unit used to encode input data.
        node (nn.Module): Neural ODE used to evolve hidden dynamics.
        out (nn.Module): NN mapping from hidden state to output.
        latent_dim (int): Dimension of latent state.
    """

    def __init__(self, latent_dim, rec_gru, rec_node, rec_output):
        """Initialize GRU-ODE model.
        This module is intended for use as the encoder of a latent NODE.
        Args:
            latent_dim (int): Dimension of latent state.
            rec_gru (nn.Module): GRU used to encoder input data.
            rec_node (nn.Module): NODE used to evolve state between GRU units.
            rec_output (nn.Module): Final linear layer.
        """
        super().__init__()

        self.gru = rec_gru
        self.node = rec_node
        self.out = rec_output
        self.latent_dim = latent_dim

    def forward(self, x, tps):
        """Compute forward pass of GRU-ODE.
        Expects input of shape (B x T x D) and time observation of shape (T).
        Supports input masked by 2D binary array of shape (B x T).
        ODE dynamics are solved using euler's method. Other solvers decrease
        performance and increase runtime.

        Args:
            x (torch.Tensor): Data observations.
            tps (torch.Tensor): Timepoints.
        Returns:
            torch.Tensor: Output representing latent parameters.
        """
        h = torch.zeros(x.size(0), self.latent_dim * 2).to(x.device)

        # Insert dummy time point which is discarded later.
        tps = torch.cat(((tps[0] - 0.01).unsqueeze(0), tps), 0)

        for i in range(x.size(1)):
            if i != 0:
                h_ode = odeint(self.node, h, tps[i:i + 2], method="euler")[1]
            else:
                # Don't evolve hidden state prior to first observation
                h_ode = h

            h_rnn = self.gru(x[:, i, :], h_ode)
            h = h_rnn

        out = self.out(h)
        return out


class ODEFuncNN(nn.Module):
    """Approximates ODE dynamics using a neural network.
    Attributes:
        act (nn.Module): Activation function to use between layers.
        fc_in (nn.Module): Linear layer mapping input to hidden state.
        fc_out (nn.Module): Linear layer mapping hidden state to output.
        fc_hidden (nn.ModuleList): Hidden layers.
    """

    def __init__(self, input_dim, n_hidden, n_layer, act_type, output_dim=None):
        """Initialize NN representing ODE function.
        Args:
            input_dim (int): Dimension of input data.
            n_hidden (int): Number of hidden units in NN.
            n_layer (int): Number of layers in NN.
            act_type (string): Type of activation to use between layers.
            output_dim (int): Dimension of NN output; defaults to input_dim
        Raises:
            ValueError: Thrown when activation function is unknown.
        """
        super().__init__()

        output_dim = input_dim if output_dim is None else output_dim
        self.fc_in = nn.Linear(input_dim, n_hidden)
        self.fc_out = nn.Linear(n_hidden, output_dim)

        layers = [nn.Linear(n_hidden, n_hidden) for _ in range(n_layer - 1)]
        self.fc_hidden = nn.ModuleList(layers)

        try:
            self.act = ACTIVATION_DICT[act_type]()
        except KeyError:
            raise ValueError("Unsupported activation function.")

    def forward(self, t, x):
        """Compute forward pass.
        Time must be passed in according to the torchdiffeq framework, but is
        generally unused.
        Args:
            t (torch.Tensor): Timepoints of observation.
            x (torch.Tensor): Data observations.
        Returns:
            torch.Tensor: Output of forward pass.
        """
        h = self.fc_in(x)
        h = self.act(h)

        for layer in self.fc_hidden:
            h = layer(h)
            h = self.act(h)

        out = self.fc_out(h)
        return out


class NeuralODE(nn.Module):
    """Neural Ordinary Differential Equation.
    Implements Neural ODEs as described by: https://arxiv.org/abs/1806.07366.
    ODE solve uses a semi-norm. See: https://arxiv.org/abs/2009.09457.

    Attributes:
        nodef (nn.Module): NN which approximates ODE function.
    """

    def __init__(self, input_dim, n_hidden, n_layer, act_type):
        """Initialize Neural ODE.
        Args:
            input_dim (int): Dimension of input data.
            n_hidden (int): Number of hidden units in NN.
            n_layer (int): Number of layers in NN.
            act_type (string): Type of activation to use between layers.
        """
        super().__init__()

        self.nodef = ODEFuncNN(input_dim, n_hidden, n_layer, act_type)

    def forward(self, z0, ts, rtol=1e-3, atol=1e-4, method='dopri5'):
        """Compute forward pass of NODE.
        Args:
            z0 (torch.Tensor): Initial state of ODE.
            ts (torch.Tensor): Time points of observations.
            rtol (float, optional): Relative tolerance of ode solver.
            atol (float, optional): Absolute tolerance of ode solver.
            method (str, optional): Type of ODE solver used.
        Returns:
            torch.Tensor: Result of ode solve from initial state.
        """
        z = odeint(self.nodef, z0, ts, rtol=rtol, atol=atol, method=method)
        return z.permute(1, 0, 2)


class DecoderNN(nn.Module):
    """Feed-forward decoder neural network.
    Map from a latent representation of an ODE back into the data dimension.

    Attributes:
        act (nn.Module): Activation function.
        fc_in (nn.Module): Linear layer mapping from latent to hidden state.
        fc_out (nn.Module): Linear Layer mapping from hidden to data space.
        fc_hidden (nn.ModuleList): Hidden layers.
    """

    def __init__(self, latent_dim, obs_dim, n_hidden, n_layer, act_type):
        """Initialize decoder network.
        Args:
            latent_dim (int): Dimension of latent representation.
            obs_dim (int): Dimension of data space.
            n_hidden (int): Number of hidden units.
            n_layer (int): Number of hidden layers.
            act_type (string): Type of activation to use between layers.
        Raises:
            ValueError: Thrown when activation function is unsupported.
        """
        super().__init__()

        self.fc_in = nn.Linear(latent_dim, n_hidden)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(n_hidden, n_hidden) for _ in range(n_layer - 1)])
        self.fc_out = nn.Linear(n_hidden, obs_dim)

        try:
            self.act = ACTIVATION_DICT[act_type]()
        except KeyError:
            raise ValueError("Unsupported activation function.")

    def forward(self, x):
        """Compute forward pass.
        Args:
            x (torch.Tensor): Latent trajectory data points.
        Returns:
            torch.Tensor: Output in data space.
        """
        out = self.fc_in(x)
        out = self.act(out)

        for layer in self.fc_hidden:
            out = layer(out)
            out = self.act(out)

        out = self.fc_out(out)
        return out


class LatentODE(nn.Module):
    """Latent Neural ODE.

    Implements Latent Neural ODE described in https://arxiv.org/abs/1907.03907.
    Model consists of a GRUODE encoder, Neural ODE, and NN Decoder which
    is configured and trained as a VAE.
    Attributes:
        dec (nn.Module): Decoder module.
        enc (nn.Module): Encoder module.
        nodef (nn.Module): Neural ODE module.
    """

    def __init__(self, enc, nodef, dec):
        """Initialize latent neural ODE.
        Args:
            dec (nn.Module): Decoder module.
            enc (nn.Module): Encoder module.
            nodef (nn.Module): Neural ODE module.
        """
        super().__init__()

        self.enc = enc
        self.nodef = nodef
        self.dec = dec

    def get_latent_initial_state(self, x, ts):
        """Compute latent parameters.

        Allows masking via a 2D binary array of shape (B x T).

        Args:
            x (torch.Tensor): Data points.
            ts (torch.Tensor): Timepoints of observations.
            mask (torch.Tensor, optional): Masking array.
        Returns:
            torch.Tensor, torch.Tensor: Latent mean and logvar parameters.
        """
        obs = torch.flip(x, [1])
        rev_ts = torch.flip(ts, [0])

        out = self.enc.forward(obs, rev_ts)

        qz0_mean = out[:, :out.size(1) // 2]
        qz0_logvar = out[:, out.size(1) // 2:]

        return qz0_mean, qz0_logvar

    def reparameterize(self, qz0_mean, qz0_logvar):
        """Generate latent initial state from latent parameters.
        Use the reparameterization trick to enable backprop with low variance.

        Args:
            qz0_mean (torch.Tensor): Mean of latent distribution.
            qz0_logvar (torch.Tensor): Log variance of latent distribution
        Returns:
            (torch.Tensor, torch.Tensor): Latent state and noise sample.
        """
        epsilon = torch.randn(qz0_mean.size(), device=qz0_mean.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        return z0, epsilon

    def generate_from_latent(self, z0, ts, rtol=1e-3, atol=1e-4,
                             method='dopri5'):
        """Generate a latent trajectory from a latent initial state.
        Args:
            z0 (torch.Tensor): Latent initial state.
            ts (torch.Tensor): Timepoints of observations.
            rtol (float, optional): NODE ODE solver relative tolerance.
            atol (float, optional): NODE ODE solver absolute tolerance.
            method (str, optional): Type of ODE solver.
        Returns:
            torch.Tensor: Latent trajectory.
        """
        return self.nodef.forward(z0, ts, rtol, atol, method)

    def forward(self, x, ts, args):
        """Compute forward pass of Latent NODE.
        Args:
            x (torch.Tensor): Input data.
            ts (torch.Tensor): Time points of observations.
            args (dict): Forward arguments.
        Returns:
            (torch.Tensor, torch.Tensor, TODO, torch.Tensor, torch.Tensor):
                Reconstructed data, latent mean, latent logvar, sampled noise.
        """
        qz0_mean, qz0_logvar = self.get_latent_initial_state(x, ts)
        z0, epsilon = self.reparameterize(qz0_mean, qz0_logvar)

        pred_z = self.generate_from_latent(z0, ts, args['model_rtol'],
                                           args['model_atol'], args['method'])
        pred_x = self.dec(pred_z)

        return pred_x, z0, qz0_mean, qz0_logvar, epsilon

    def get_prediction(self, x, ts, rtol=1e-3, atol=1e-4, method='dopri5'):
        """Retrieve prediction from latent NODE output.
        Args:
            x (torch.Tensor): Data points.
            ts (torch.Tensor): Timepoints of observations.
            rtol (float, optional): NODE ODE solver relative tolerance.
            atol (float, optional): NODE ODE solver absolute tolerance.
            method (str, optional): Type of ODE solver.
        Returns:
            torch.Tensor: Reconstructed data points.
        """
        # TODO: add option to expand K / M
        args = {
            'model_rtol': rtol,
            'model_atol': atol,
            'method': method,
            'M': 1,
            'K': 1,
        }

        return self.forward(x, ts, args)[0]

    def get_elbo(self, x, pred_x, z0, qz0_mean, qz0_logvar, eps, args):
        """Compute the ELBO.
        Computes the evidence lower bound (ELBO) for a given prediction,
        ground truth, and latent initial state.
        Supports KL annealing, where the KL term can gradually be increased
        during training, as described in: https://arxiv.org/abs/1903.10145.

        Args:
            x (torch.Tensor): Input data.
            pred_x (torch.Tensor): Data reconstructed by latent NODE.
            z0 (torch.Tensor): Reparameterized latent initial states.
            qz0_mean (torch.Tensor): Latent initial state means.
            qz0_logvar (torch.Tensor): Latent initial state variances.
            eps (torch.Tensor): Reparameterization noise sample.
            args (dict): Additional arguments.
        Returns:
            torch.Tensor: ELBO score.
        """
        noise_std_ = torch.zeros(pred_x.size(), device=x.device) + args['l_std']
        noise_logvar = 2. * torch.log(noise_std_)

        logpx = log_normal_pdf(x, pred_x, noise_logvar).sum(-1).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size(), device=x.device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)

        return torch.mean(-logpx + analytic_kl, dim=0)
