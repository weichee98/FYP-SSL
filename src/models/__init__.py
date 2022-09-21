__all__ = [
    "FFN",
    "VAE_FFN",
    "VAECH",
    "VAECH_I",
    "VAECH_II",
    "VAESDR",
    "count_parameters",
]

import os
import sys
from torch.nn import Module

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .FFN import FFN
from .VAE_FFN import VAE_FFN
from .VAECH import VAECH
from .VAECH_I import VAECH_I
from .VAECH_II import VAECH_II
from .VAESDR import VAESDR


def count_parameters(model: Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
