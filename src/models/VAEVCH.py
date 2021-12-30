import os
import sys
import torch

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from models.VAECH import VAECH


class VCH(torch.nn.Module):
    def __init__(self, input_size, num_sites):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(input_size))

        self.age = torch.nn.Linear(1, input_size)
        self.age_norm = torch.nn.BatchNorm1d(1)
        self.age_log_std = torch.nn.Parameter(torch.tensor([[-1.0]]))

        self.gender = torch.nn.Linear(2, input_size)
        self.gender_log_std = torch.nn.Parameter(torch.tensor([[-1.0]]))

        self.batch_add = torch.nn.Linear(num_sites, input_size)
        self.batch_mul = torch.nn.Linear(num_sites, input_size)
        self.batch_add_log_std = torch.nn.Parameter(torch.tensor([[-1.0]]))
        self.batch_mul_log_std = torch.nn.Parameter(torch.tensor([[-1.0]]))

    def forward(self, x, age, gender, site):
        age_x = self.age(self.age_norm(age))
        gender_x = self.gender(gender)
        batch_add = self.batch_add(site)
        batch_mul = self.batch_mul(site)

        if self.training:
            q_age = torch.distributions.Normal(age_x, torch.exp(self.age_log_std))
            age_x = q_age.rsample()
            q_gender = torch.distributions.Normal(
                gender_x, torch.exp(self.gender_log_std)
            )
            gender_x = q_gender.rsample()
            q_batch_add = torch.distributions.Normal(
                batch_add, torch.exp(self.batch_add_log_std)
            )
            batch_add = q_batch_add.rsample()
            q_batch_mul = torch.distributions.Normal(
                batch_mul, torch.exp(self.batch_mul_log_std)
            )
            batch_add = q_batch_mul.rsample()

        batch_mul = torch.exp(batch_mul)
        error = (x - self.alpha - age_x - gender_x - batch_add) / batch_mul
        return error

    def inverse(self, error, age, gender, site):
        age_x = self.age(self.age_norm(age))
        gender_x = self.gender(gender)
        batch_add = self.batch_add(site)
        batch_mul = self.batch_mul(site)

        if self.training:
            q_age = torch.distributions.Normal(age_x, torch.exp(self.age_log_std))
            age_x = q_age.rsample()
            q_gender = torch.distributions.Normal(
                gender_x, torch.exp(self.gender_log_std)
            )
            gender_x = q_gender.rsample()
            q_batch_add = torch.distributions.Normal(
                batch_add, torch.exp(self.batch_add_log_std)
            )
            batch_add = q_batch_add.rsample()

        batch_mul = torch.exp(batch_mul)
        x = self.alpha + age_x + gender_x + batch_add + batch_mul * error
        return x


class VAEVCH(VAECH):
    def __init__(self, input_size, l1, l2, emb_size, l3, num_sites):
        """
        l1: number of nodes in the hidden layer of encoder and decoder
        emb_size: size of encoder output and decoder input
        l2, l3: number of nodes in the hidden layer of classifier
        """
        super().__init__(input_size, l1, l2, emb_size, l3, num_sites)
        self.ch = VCH(input_size, num_sites)
