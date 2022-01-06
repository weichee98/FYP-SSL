import os
import sys
import torch
import torch.nn.functional as F

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from models.VAESDR import VAESDR


class VAESDRII(VAESDR):
    def __init__(self, input_size, l1, emb_size, num_site):
        super().__init__(input_size, l1, emb_size, num_site)
        self.site_decoder1 = torch.nn.Linear(emb_size, l1)
        self.site_decoder2 = torch.nn.Linear(l1, input_size)

    def decode(self, z_res, z_site):
        x = self.site_decoder1(z_site)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x_site = self.site_decoder2(x)

        x = self.decoder1(z_res)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x_res = self.decoder2(x)

        x_mu = torch.tanh(x_site + x_res)
        x_res = torch.tanh(x_res)
        x_site = x_mu - x_res
        x_std = torch.exp(self.log_std)
        return x_mu, x_std, x_res, x_site
