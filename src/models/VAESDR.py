import os
import sys
import torch
import torch.nn.functional as F

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.loss import GaussianKLDivLoss
from utils.metrics import ClassificationMetrics as CM
from models.base import SaliencyScoreForward


class VAESDR(torch.nn.Module, SaliencyScoreForward):
    def __init__(self, input_size, l1, l2, emb_size):
        """
        l1: number of nodes in the hidden layer of encoder and decoder
        emb_size: size of encoder output and decoder input
        l2, l3: number of nodes in the hidden layer of classifier
        """
        super().__init__()
        self.encoder1 = torch.nn.Linear(input_size, l1)
        self.encoder_mu = torch.nn.Linear(l1, emb_size)
        self.encoder_std = torch.nn.Linear(l1, emb_size)

        self.site_encoder = torch.nn.Linear(emb_size, emb_size)
        self.res_encoder = torch.nn.Linear(emb_size, emb_size)

        self.site_decoder1 = torch.nn.Linear(emb_size, l1)
        self.site_decoder2 = torch.nn.Linear(l1, input_size)
        self.res_decoder1 = torch.nn.Linear(emb_size, l1)
        self.res_decoder2 = torch.nn.Linear(l1, input_size)

        self.cls1 = torch.nn.Linear(emb_size, l2)
        self.cls2 = torch.nn.Linear(l2, 2)  # this is the head for disease class
        self.log_std = torch.nn.Parameter(torch.tensor([[0.0]]))

    def forward(self, x):
        z, z_mu, z_std, z_res, z_site = self._encode(x)
        x_mu, x_std, x_res, x_site = self._decode(z_res, z_site)
        y = self._cls(z_res)
        return y, (x_mu, x_std, x_res, x_site), (z, z_mu, z_std, z_res, z_site)

    def _cls(self, z_res):
        y = self.cls1(z_res)
        y = F.relu(y)
        y = F.dropout(y, p=0.5, training=self.training)

        y = self.cls2(y)
        y = F.softmax(y, dim=1)  # output for disease classification
        return y

    def _encode(self, x):
        x = self.encoder1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        z_mu = self.encoder_mu(x)
        z_mu = torch.tanh(z_mu)
        z_log_std = self.encoder_std(x)
        z_log_std = torch.tanh(z_log_std)
        z_std = torch.exp(z_log_std)

        if self.training:
            q = torch.distributions.Normal(z_mu, z_std)
            z = q.rsample()
        else:
            z = z_mu

        z_site = self.site_encoder(z)
        z_res = self.res_encoder(z)
        return z, z_mu, z_std, z_res, z_site

    def _decode(self, z_res, z_site):
        x = self.res_decoder1(z_res)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.res_decoder2(x)
        x_res = torch.tanh(x)

        x = self.site_decoder1(z_site)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.site_decoder2(x)
        x_site = torch.tanh(x)

        x_mu = x_res + x_site
        x_std = torch.exp(self.log_std)
        return x_mu, x_std, x_res, x_site

    def ss_forward(self, x):
        _, _, _, z_res, _ = self._encode(x)
        y = self._cls(z_res)
        return y


def train_VAESDR(
    device,
    model,
    data,
    optimizer,
    labeled_idx,
    all_idx=None,
    gamma1=0,
    gamma2=0,
    gamma3=0,
    gamma4=0,
    weight=False,
):
    """
    all_idx: the indices of labeled and unlabeled data (exclude test indices)
    gamma1: float, the weightage of reconstruction loss
    gamma2: float, the weightage of regularizer (kl divergence)
    gamma2: float, the weightage of regularizer (z vs z_site + z_res)
    gamma4: float, the weightage of second pass loss, using x_res
    """
    model.to(device)
    model.train()
    optimizer.zero_grad()

    real_y = data.y[labeled_idx].to(device)
    if weight:
        _, counts = torch.unique(real_y, sorted=True, return_counts=True)
        weight = counts[[1, 0]] / counts.sum()
    else:
        weight = None

    cls_criterion = torch.nn.CrossEntropyLoss(weight=weight)
    gauss_criterion = torch.nn.GaussianNLLLoss(full=True)
    kl_criterion = GaussianKLDivLoss()
    z_regularizer = torch.nn.MSELoss(reduction="sum")

    x = data.x.to(device)
    pred_y, (x_mu, x_std, x_res, _), (z, z_mu, z_std, z_res, z_site) = model(x)
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
    z_loss = z_regularizer(z[all_idx], z_res[all_idx] + z_site[all_idx]) / len(all_idx)
    total_loss = loss + gamma1 * rc_loss + gamma2 * kl + gamma3 * z_loss

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

    if gamma4 > 0:
        pred_y, (x_mu, x_std, _, _), (z, z_mu, z_std, z_res, z_site) = model(x_res)
        loss = cls_criterion(pred_y[labeled_idx], real_y)
        if all_idx is None:
            all_idx = labeled_idx
        x_std = x_std.expand(x_mu[all_idx].size())
        rc_loss = gauss_criterion(x_res[all_idx], x_mu[all_idx], x_std ** 2)
        kl = kl_criterion(
            z_mu[all_idx],
            z_std[all_idx] ** 2,
            torch.zeros_like(z_mu[all_idx]),
            torch.ones_like(z_std[all_idx]),
        )
        z_loss = (
            z_regularizer(z_site[all_idx], torch.zeros_like(z_site[all_idx]))
            + z_regularizer(z[all_idx], z_res[all_idx])
        ) / (2.0 * len(all_idx))
        total_loss += gamma4 * (loss + gamma1 * rc_loss + gamma2 * kl + gamma3 * z_loss)

    loss_val = total_loss.item()
    total_loss.backward()
    optimizer.step()

    return loss_val, accuracy.item(), metrics


def test_VAESDR(device, model: VAESDR, data, test_idx):
    model.to(device)
    model.eval()

    pred_y = model.ss_forward(data.x.to(device))
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
