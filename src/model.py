import torch.nn as nn
from base_model import GRU, ODEFuncNN, EncoderGRUODE, NeuralODE, DecoderNN

from base_model import LatentODE
from miw_latode import MIWLatentODE
from ciw_latode import CIWLatentODE
from piw_latode import PIWLatentODE


class LatentNeuralODEBuilder:
    """Construct Latent Neural NODE."""

    elbo_map = {
        'base': LatentODE,
        'iw': MIWLatentODE,
        'miw': MIWLatentODE,
        'ciw': CIWLatentODE,
        'piw': PIWLatentODE,
        'betavae': LatentODE,
    }

    def __init__(self, obs_dim, rec_latent_dim, node_latent_dim,
                 rec_gru_unit, rec_node_hidden, rec_node_layer, rec_node_act,
                 latent_node_hidden, latent_node_layer, latent_node_act,
                 dec_type, dec_hidden, dec_layer, dec_act, rec_out_hidden=100):
        """Initialize all sub-modules for Latent NODE.
        Notes on hyperparameter selection:
        The dimensionality of the encoder latent state should be >2x larger
        than decoder latent state. The decoder latent state should be close to
        the dimension of observed data.
        Args:
            obs_dim (int): Dimension of input data.
            rec_latent_dim (int): Dimension of encoder latent state.
            node_latent_dim (int): Dimension of node latent state.
            rec_gru_unit (int): Number of units in encoder GRU.
            rec_node_hidden (int): Number of hidden units in encoder NODE.
            rec_node_layer (int): Number of layers in encoder NODE.
            rec_node_act (str): Activations used by encoder NODE.
            latent_node_hidden (int): Number of hidden units in latent NODE.
            latent_node_layer (int): Number of layers in latent NODE.
            latent_node_act (str): Activations used by latent NODE.
            dec_type (str): Type of decoder network to use.
            dec_hidden (int): Number of hidden units in decoder NN.
            dec_layer (int): Number of layers in decoder NN.
            dec_act (string): Activation function used in decoder NN.
            rec_out_hidden (int): Number of hidden units in rec output NN.
        Raises:
            ValueError: Thrown when decoder type is unsupported.
        """
        enc_gru = GRU(rec_latent_dim, obs_dim, rec_gru_unit)

        enc_node = ODEFuncNN(rec_latent_dim * 2, rec_node_hidden,
                             rec_node_layer, rec_node_act)

        enc_out = nn.Sequential(
            nn.Linear(rec_latent_dim * 2, rec_out_hidden),
            nn.Tanh(),
            nn.Linear(rec_out_hidden, node_latent_dim * 2)
        )

        self.enc = EncoderGRUODE(rec_latent_dim, enc_gru, enc_node, enc_out)

        self.latent_node = NeuralODE(node_latent_dim, latent_node_hidden,
                                     latent_node_layer, latent_node_act)

        if dec_type == 'Linear':
            self.dec = nn.Linear(node_latent_dim, obs_dim)
        elif dec_type == 'NN':
            self.dec = DecoderNN(node_latent_dim, obs_dim,
                                 dec_hidden, dec_layer, dec_act)
        else:
            raise ValueError("Unknown or unsupported decoder type.")

    def build_latent_node(self, elbo_type):
        """Construct LatentNeuralODE with provided components.

        Args:
            elbo_type (str): The type of elbo to use.

        Returns:
            LatentODE: Constructed latent node.
        """
        try:
            model = self.elbo_map[elbo_type]
        except KeyError:
            raise ValueError("Unknown elbo type.")

        return model(self.enc, self.latent_node, self.dec)
