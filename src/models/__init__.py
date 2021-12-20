import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GCN import *
from GNN import *
from FFN import *
from AE import *
from VAE import *
from VGAE import *
from DIVA import *
from VGAETS import *
from VAESDR import *
from VAECH import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
