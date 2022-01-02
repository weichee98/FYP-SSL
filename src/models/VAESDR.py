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
    def __init__(self, input_size, l1, emb_size, num_site):
        super().__init__()
        self.encoder1 = torch.nn.Linear(input_size, l1)
        self.encoder_mu = torch.nn.Linear(l1, emb_size)
        self.encoder_std = torch.nn.Linear(l1, emb_size)

        self.site_encoder = torch.nn.Linear(emb_size, emb_size)
        self.res_encoder = torch.nn.Linear(emb_size, emb_size)

        self.decoder1 = torch.nn.Linear(emb_size, l1)
        self.decoder2 = torch.nn.Linear(l1, input_size)
        self.log_std = torch.nn.Parameter(torch.tensor([[0.0]]))

        self.site_cls = torch.nn.Linear(emb_size, num_site)
        self.disease_cls = torch.nn.Linear(emb_size, 2)

        self.site_dis = torch.nn.Linear(emb_size, num_site)
        self.disease_dis = torch.nn.Linear(emb_size, 2)

    def forward(self, x):
        z, z_mu, z_std, z_res, z_site = self.encode(x)
        x_mu, x_std, x_res, x_site = self.decode(z_res, z_site)
        y = self.classify_disease(z_res)
        d = self.classify_site(z_site)
        dis_y = self.discriminate_disease(z_site)
        dis_d = self.discriminate_site(z_res)
        return (
            y,
            d,
            dis_y,
            dis_d,
            (x_mu, x_std, x_res, x_site),
            (z, z_mu, z_std, z_res, z_site),
        )

    def classify_disease(self, z_res):
        y = self.disease_cls(z_res)
        # y = F.relu(y)
        # y = F.dropout(y, p=0.5, training=self.training)
        # y = self.disease_cls2(y)
        y = F.softmax(y, dim=1)
        return y

    def classify_site(self, z_site):
        d = self.site_cls(z_site)
        d = F.softmax(d, dim=1)
        return d

    def discriminate_disease(self, z_site):
        y = self.disease_dis(z_site)
        y = F.softmax(y, dim=1)
        return y

    def discriminate_site(self, z_res):
        d = self.site_dis(z_res)
        d = F.softmax(d, dim=1)
        return d

    def encode(self, x):
        x = self.encoder1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        z_mu = self.encoder_mu(x)
        z_log_std = self.encoder_std(x)
        z_mu = torch.tanh(z_mu)
        z_log_std = torch.tanh(z_log_std)
        z_std = torch.exp(z_log_std)

        if self.training:
            q = torch.distributions.Normal(z_mu, z_std)
            z = q.rsample()
        else:
            z = z_mu

        z_site = torch.tanh(self.site_encoder(z))
        z_res = torch.tanh(self.res_encoder(z))
        return z, z_mu, z_std, z_res, z_site

    def decode(self, z_res, z_site):
        x = self.decoder1(z_site)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x_site = self.decoder2(x)

        x = self.decoder1(z_res)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x_res = self.decoder2(x)

        x_mu = torch.tanh(x_site + x_res)
        x_res = torch.tanh(x_res)
        x_site = x_mu - x_res
        x_std = torch.exp(self.log_std)
        return x_mu, x_std, x_res, x_site

    def ss_forward(self, x):
        _, _, _, z_res, _ = self.encode(x)
        y = self.classify_disease(z_res)
        return y

    def get_optimizer(model, param):
        model_optim = torch.optim.Adam(
            map(
                lambda p: p[1],
                filter(
                    lambda p: p[1].requires_grad
                    and "dis" not in p[0]
                    and "cls" not in p[0],
                    model.named_parameters(),
                ),
            ),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        disease_dis_optim = torch.optim.Adam(
            map(
                lambda p: p[1],
                filter(
                    lambda p: p[1].requires_grad and "disease_dis" in p[0],
                    model.named_parameters(),
                ),
            ),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        site_dis_optim = torch.optim.Adam(
            map(
                lambda p: p[1],
                filter(
                    lambda p: p[1].requires_grad and "site_dis" in p[0],
                    model.named_parameters(),
                ),
            ),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        disease_cls_optim = torch.optim.Adam(
            map(
                lambda p: p[1],
                filter(
                    lambda p: p[1].requires_grad and "disease_cls" in p[0],
                    model.named_parameters(),
                ),
            ),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        site_cls_optim = torch.optim.Adam(
            map(
                lambda p: p[1],
                filter(
                    lambda p: p[1].requires_grad and "site_cls" in p[0],
                    model.named_parameters(),
                ),
            ),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        optimizer = (
            model_optim,
            disease_cls_optim,
            site_cls_optim,
            disease_dis_optim,
            site_dis_optim,
        )
        return optimizer


def _entropy_loss(pred_y):
    uni_dist = torch.ones(pred_y.size(0), device=pred_y.device) / pred_y.size(1)
    max_entropy = -uni_dist.log()
    entropy = torch.sum(-pred_y * pred_y.log(), dim=1)
    return torch.mean(max_entropy - entropy)


def train_VAESDR(
    device,
    model: VAESDR,
    data,
    optimizer,
    labeled_idx,
    all_idx=None,
    gamma1=0,
    gamma2=0,
    gamma3=0,
    gamma4=0,
    gamma5=0,
    weight=False,
):
    """
    all_idx: the indices of labeled and unlabeled data (exclude test indices)
    gamma1: float, the weightage of reconstruction loss
    gamma2: float, the weightage of regularizer (kl divergence)
    gamma3: float, the weightage of regularizer (z = z_site + z_res)
    gamma4: float, the weightage of discriminator loss
    gamma5: float, the weightage of second pass loss, using x_res
    """
    model.to(device)
    model.train()

    (
        model_optim,
        disease_cls_optim,
        site_cls_optim,
        disease_dis_optim,
        site_dis_optim,
    ) = optimizer

    real_y = data.y[labeled_idx].to(device)
    real_d = data.d.to(device)
    if weight:
        _, counts = torch.unique(real_y, sorted=True, return_counts=True)
        weight = counts[[1, 0]] / counts.sum()
    else:
        weight = None
    if all_idx is None:
        all_idx = labeled_idx

    cls_criterion = torch.nn.CrossEntropyLoss(weight=weight)
    d_criterion = torch.nn.CrossEntropyLoss(weight=None)
    gauss_criterion = torch.nn.GaussianNLLLoss(full=True)
    kl_criterion = GaussianKLDivLoss()
    z_regularizer = torch.nn.GaussianNLLLoss(full=True)
    x = data.x.to(device)

    site_cls_optim.zero_grad()
    disease_cls_optim.zero_grad()
    site_dis_optim.zero_grad()
    disease_dis_optim.zero_grad()
    _, _, _, z_res, z_site = model.encode(x)
    z_res, z_site = z_res.detach(), z_site.detach()
    disc_d = model.discriminate_site(z_res)
    disc_y = model.discriminate_disease(z_site)
    pred_d = model.classify_site(z_site)
    pred_y = model.classify_disease(z_res)
    site_dis_loss = d_criterion(disc_d[all_idx], real_d[all_idx])
    disease_dis_loss = cls_criterion(disc_y[labeled_idx], real_y)
    site_cls_loss = d_criterion(pred_d[all_idx], real_d[all_idx])
    disease_cls_loss = cls_criterion(pred_y[labeled_idx], real_y)
    if gamma5 > 0:
        _, _, x_res, _ = model.decode(z_res, z_site)
        _, _, _, z_res, z_site = model.encode(x_res)
        z_res, z_site = z_res.detach(), z_site.detach()
        disc_d = model.discriminate_site(z_res)
        disc_y = model.discriminate_disease(z_site)
        pred_d = model.classify_site(z_site)
        pred_y = model.classify_disease(z_res)
        site_dis_loss += gamma5 * d_criterion(disc_d[all_idx], real_d[all_idx])
        disease_dis_loss += gamma5 * cls_criterion(disc_y[labeled_idx], real_y)
        site_cls_loss += gamma5 * d_criterion(pred_d[all_idx], real_d[all_idx])
        disease_cls_loss += gamma5 * cls_criterion(pred_y[labeled_idx], real_y)
    site_dis_loss.backward()
    disease_dis_loss.backward()
    site_cls_loss.backward()
    disease_cls_loss.backward()
    site_dis_optim.step()
    disease_dis_optim.step()
    site_cls_optim.step()
    disease_cls_optim.step()

    model_optim.zero_grad()
    (
        pred_y,
        pred_d,
        disc_y,
        disc_d,
        (x_mu, x_std, x_res, _),
        (z, z_mu, z_std, z_res, z_site),
    ) = model(x)
    loss = cls_criterion(pred_y[labeled_idx], real_y)
    x_std = x_std.expand(x_mu[all_idx].size())
    rc_loss = gauss_criterion(x[all_idx], x_mu[all_idx], x_std ** 2)
    kl = kl_criterion(
        z_mu[all_idx],
        z_std[all_idx] ** 2,
        torch.zeros_like(z_mu[all_idx]),
        torch.ones_like(z_std[all_idx]),
    )
    z_loss = z_regularizer(
        z_res[all_idx] + z_site[all_idx], z_mu[all_idx], z_std[all_idx] ** 2
    )
    d_loss = d_criterion(pred_d[all_idx], real_d[all_idx])
    dis_loss = (_entropy_loss(disc_d[all_idx]) + _entropy_loss(disc_y[labeled_idx])) / 2
    total_loss = (
        loss
        + d_loss
        + gamma1 * rc_loss
        + gamma2 * kl
        + gamma3 * z_loss
        + gamma4 * dis_loss
    )

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

    if gamma5 > 0:
        (
            pred_y,
            pred_d,
            disc_y,
            disc_d,
            (x_mu, x_std, x_res_2, _),
            (z, z_mu, z_std, z_res, z_site),
        ) = model(x_res)
        disc_y = model.discriminate_disease(z_site)
        disc_d = model.discriminate_site(z_res)
        pred_d = model.classify_site(z_site)
        loss = cls_criterion(pred_y[labeled_idx], real_y)
        if all_idx is None:
            all_idx = labeled_idx
        x_std = x_std.expand(x_mu[all_idx].size())
        rc_loss = (
            gauss_criterion(x_res[all_idx], x_mu[all_idx], x_std ** 2,)
            + gauss_criterion(x_res[all_idx], x_res_2[all_idx], x_std ** 2,)
        ) / 2
        kl = kl_criterion(
            z_mu[all_idx],
            z_std[all_idx] ** 2,
            torch.zeros_like(z_mu[all_idx]),
            torch.ones_like(z_std[all_idx]),
        )
        z_loss = (
            z_regularizer(z_res[all_idx], z_mu[all_idx], z_std[all_idx] ** 2)
            + z_regularizer(
                z_site[all_idx], torch.zeros_like(z_mu[all_idx]), z_std[all_idx] ** 2
            )
        ) / 2
        d_loss = _entropy_loss(pred_d[all_idx])
        dis_loss = (
            _entropy_loss(disc_d[all_idx]) + _entropy_loss(disc_y[labeled_idx])
        ) / 2
        total_loss += gamma5 * (
            loss
            + d_loss
            + gamma1 * rc_loss
            + gamma2 * kl
            + gamma3 * z_loss
            + gamma4 * dis_loss
        )

    loss_val = total_loss.item()
    total_loss.backward()
    model_optim.step()

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
