import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import squareform
from captum.attr import IntegratedGradients

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.loss import GaussianKLDivLoss
from utils.metrics import ClassificationMetrics as CM
from models.base import SaliencyScoreForward


class CH(torch.nn.Module):
    def __init__(self, input_size, num_sites):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(input_size))
        self.age = torch.nn.Linear(1, input_size)
        self.age_norm = torch.nn.BatchNorm1d(1)
        self.gender = torch.nn.Linear(2, input_size)
        self.batch_add = torch.nn.Linear(num_sites, input_size)
        self.batch_mul = torch.nn.Linear(num_sites, input_size)

    def forward(self, x, age, gender, site):
        age_x = self.age(self.age_norm(age))
        gender_x = self.gender(gender)
        batch_add = self.batch_add(site)
        batch_mul = torch.exp(self.batch_mul(site))
        error = (x - self.alpha - age_x - gender_x - batch_add) / batch_mul
        return error

    def inverse(self, error, age, gender, site):
        age_x = self.age(self.age_norm(age))
        gender_x = self.gender(gender)
        batch_add = self.batch_add(site)
        batch_mul = torch.exp(self.batch_mul(site))
        x = self.alpha + age_x + gender_x + batch_add + batch_mul * error
        return x


_DROPOUT = 0.25


class VAECH(torch.nn.Module, SaliencyScoreForward):
    def __init__(self, input_size, l1, l2, emb_size, l3, num_sites):
        """
        l1: number of nodes in the hidden layer of encoder and decoder
        emb_size: size of encoder output and decoder input
        l2, l3: number of nodes in the hidden layer of classifier
        """
        super().__init__()
        self.ch = CH(input_size, num_sites)
        self.encoder1 = torch.nn.Linear(input_size, l1)
        self.encoder_mu = torch.nn.Linear(l1, emb_size)
        self.encoder_std = torch.nn.Linear(l1, emb_size)

        self.decoder1 = torch.nn.Linear(emb_size, l1)
        self.decoder2 = torch.nn.Linear(l1, input_size)

        self.cls1 = torch.nn.Linear(emb_size, l2)
        self.cls2 = torch.nn.Linear(l2, l3)
        self.cls3 = torch.nn.Linear(l3, 2)  # this is the head for disease class
        self.log_std = torch.nn.Parameter(torch.tensor([[0.0]]))

    def forward(self, x, age, gender, site):
        error = self.ch(x, age, gender, site)
        z_mu, z_log_std = self._encode(error)
        z_std = torch.exp(z_log_std)

        if self.training:
            q = torch.distributions.Normal(z_mu, z_std)
            z = q.rsample()
        else:
            z = z_mu

        error_mu = self._decode(z)
        x_mu = self.ch.inverse(error_mu, age, gender, site)
        x_std = torch.exp(self.log_std)

        y = self.cls1(z)
        y = F.relu(y)
        y = F.dropout(y, p=_DROPOUT, training=self.training)

        y = self.cls2(y)
        y = F.relu(y)
        y = F.dropout(y, p=_DROPOUT, training=self.training)

        y = self.cls3(y)
        y = F.softmax(y, dim=1)  # output for disease classification
        return y, x_mu, x_std, z, z_mu, z_std, error

    def _encode(self, x):
        x = self.encoder1(x)
        x = F.relu(x)
        x = F.dropout(x, p=_DROPOUT, training=self.training)

        mu = self.encoder_mu(x)
        mu = torch.tanh(mu)
        log_std = self.encoder_std(x)
        log_std = torch.tanh(log_std)
        return mu, log_std

    def _decode(self, x):
        x = self.decoder1(x)
        x = F.relu(x)
        x = F.dropout(x, p=_DROPOUT, training=self.training)

        x = self.decoder2(x)
        x = torch.tanh(x)
        return x

    def ss_forward(self, x, age, gender, site):
        return self.forward(x, age, gender, site)[0]

    def get_baselines_inputs(self, data):
        x, y = data.x, data.y
        age, gender, site = data.age, data.gender, data.d
        baselines = (
            x[y == 0].mean(dim=0).view(1, -1),
            age[y == 0].mean(dim=0).view(1, 1),
            gender[y == 0].mean(dim=0).view(1, 2),
            site[y == 0].mean(dim=0).view(1, site.size(1)),
        )
        inputs = (x[y == 1], age[y == 1], gender[y == 1], site[y == 1])
        return baselines, inputs

    def saliency_score(self, data):
        baselines, inputs = self.get_baselines_inputs(data)
        ig = IntegratedGradients(self.ss_forward, True)
        scores, _, _, _ = ig.attribute(inputs=inputs, baselines=baselines, target=1)

        scores = scores.detach().cpu().numpy()
        scores = np.array([squareform(score) for score in scores])
        return scores


def train_VAECH(
    device,
    model: VAECH,
    data,
    optimizer,
    labeled_idx,
    all_idx=None,
    gamma1=0,
    gamma2=0,
    weight=False,
):
    """
    all_idx: the indices of labeled and unlabeled data (exclude test indices)
    gamma1: float, the weightage of reconstruction loss
    gamma2: float, the weightage of regularizer (kl divergence)
    """
    model.to(device)
    model.train()
    optimizer.zero_grad()

    x = data.x.to(device)
    age = data.age.to(device)
    gender = data.gender.to(device)
    site = data.d.to(device)

    pred_y, x_mu, x_std, z, z_mu, z_std, error = model(x, age, gender, site)
    real_y = data.y[labeled_idx].to(device)
    if weight:
        _, counts = torch.unique(real_y, sorted=True, return_counts=True)
        weight = counts[[1, 0]] / counts.sum()
    else:
        weight = None

    cls_criterion = torch.nn.CrossEntropyLoss(weight=weight)
    gauss_criterion = torch.nn.GaussianNLLLoss(full=True)
    kl_criterion = GaussianKLDivLoss()
    ch_criterion = torch.nn.MSELoss()
    loss = cls_criterion(pred_y[labeled_idx], real_y)

    if all_idx is None:
        all_idx = labeled_idx
    x_std = x_std.expand(x_mu[all_idx].size())
    rc_loss = gauss_criterion(x[all_idx], x_mu[all_idx], x_std ** 2)
    kl = kl_criterion(
        z_mu[all_idx],
        z_std[all_idx] ** 2,
        torch.zeros_like(z_mu[all_idx]),
        torch.ones_like(z_std[all_idx]),
    )
    ch_loss = ch_criterion(error[all_idx], torch.zeros_like(error[all_idx]))
    loss += gamma1 * rc_loss + gamma2 * kl + ch_loss

    loss_val = loss.item()
    loss.backward()
    optimizer.step()

    accuracy = CM.accuracy(real_y, pred_y[labeled_idx])
    sensitivity = CM.tpr(real_y, pred_y[labeled_idx])
    specificity = CM.tnr(real_y, pred_y[labeled_idx])
    precision = CM.ppv(real_y, pred_y[labeled_idx])
    f1_score = CM.f1_score(real_y, pred_y[labeled_idx])
    metrics = {
        "sensitivity": sensitivity.item(),
        "specificity": specificity.item(),
        "f1": f1_score.item(),
        "precision": precision.item(),
    }
    return loss_val, accuracy.item(), metrics


def test_VAECH(device, model, data, test_idx):
    model.to(device)
    model.eval()

    pred_y, _, _, _, _, _, _ = model(
        data.x.to(device),
        data.age.to(device),
        data.gender.to(device),
        data.d.to(device),
    )
    real_y = data.y[test_idx].to(device)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(pred_y[test_idx], real_y)

    pred_y = pred_y.argmax(dim=1)[test_idx]
    accuracy = CM.accuracy(real_y, pred_y)
    sensitivity = CM.tpr(real_y, pred_y)
    specificity = CM.tnr(real_y, pred_y)
    precision = CM.ppv(real_y, pred_y)
    f1_score = CM.f1_score(real_y, pred_y)
    metrics = {
        "sensitivity": sensitivity.item(),
        "specificity": specificity.item(),
        "f1": f1_score.item(),
        "precision": precision.item(),
    }
    return loss.item(), accuracy.item(), metrics
