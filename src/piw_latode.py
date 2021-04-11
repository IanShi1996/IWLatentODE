import math
import torch

from base_model import LatentODE, log_normal_pdf


class PIWLatentODE(LatentODE):
    def __init__(self, enc, nodef, dec):
        super().__init__(enc, nodef, dec)
        raise NotImplementedError()
