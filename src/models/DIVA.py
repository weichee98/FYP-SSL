import os
import sys
import torch
import torch.nn.functional as F

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.metrics import ClassificationMetrics as CM


class decoder(torch.nn.Module):

    def __init__(self, emb_dim, hidden_dim, output_dim):
        super().__init__()
        self.decoder1 = torch.nn.Linear(emb_dim, hidden_dim)
        self.decoder2 = torch.nn.Linear(hidden_dim, output_dim)
        self.log_std = torch.nn.Parameter(torch.tensor([[0.0]]))

    def forward(self, z):
        z = self.decoder1(z)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        x_mu = self.decoder2(z)
        x_mu = torch.tanh(x_mu)
        x_std = torch.exp(self.log_std)

        return x_mu, x_std.expand(x_mu.size())


class pz(torch.nn.Module):

    def __init__(self, y_dim, hidden_dim, z_dim):
        super().__init__()
        self.y_dim = y_dim
        self.linear1 = torch.nn.Linear(y_dim, hidden_dim)
        self.mu = torch.nn.Linear(hidden_dim, z_dim)
        self.log_std = torch.nn.Linear(hidden_dim, z_dim)

    def forward(self, y):
        if y.ndim == 1:
            y = F.one_hot(y, num_classes=self.y_dim).float()

        h = self.linear1(y)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        z_mu = self.mu(h)
        z_mu = torch.tanh(z_mu)

        z_log_std = self.log_std(h)
        z_log_std = torch.tanh(z_log_std)
        z_std = torch.exp(z_log_std)

        return z_mu, z_std


class encoder(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.encoder1 = torch.nn.Linear(input_dim, hidden_dim)
        self.mu = torch.nn.Linear(hidden_dim, z_dim)
        self.log_std = torch.nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        x = self.encoder1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        z_mu = self.mu(x)
        z_mu = torch.tanh(z_mu)

        z_log_std = self.log_std(x)
        z_log_std = torch.tanh(z_log_std)
        z_std = torch.exp(z_log_std)

        return z_mu, z_std


class classifier(torch.nn.Module):

    def __init__(self, z_dim, hidden_dim, y_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(z_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, y_dim)

    def forward(self, z):
        z = self.linear1(z)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        y = self.linear2(z)
        y = F.softmax(y, dim=1)
        return y


class DIVA(torch.nn.Module):

    def __init__(self, input_size, z_dim, d_dim, hidden1, hidden2):
        super().__init__()

        self.zx = encoder(input_size, hidden1, z_dim)
        self.zy = encoder(input_size, hidden1, z_dim)
        self.zd = encoder(input_size, hidden1, z_dim)

        self.d = classifier(z_dim, hidden2, d_dim)
        self.y = classifier(z_dim, hidden2, 2)
        self.pzd = pz(d_dim, hidden2, z_dim)
        self.pzy = pz(2, hidden2, z_dim)

        self.px = decoder(z_dim, hidden1, input_size)


    def forward(self, x, d=None, y=None):
        zx_mu, zx_std = self.zx(x)
        zd_mu, zd_std = self.zd(x)
        zy_mu, zy_std = self.zy(x)

        qzd = torch.distributions.Normal(zd_mu, zd_std)
        zd = qzd.rsample()
        qzy = torch.distributions.Normal(zy_mu, zy_std)
        zy = qzy.rsample()
        qzx = torch.distributions.Normal(zx_mu, zx_std)
        zx = qzx.rsample()

        z = zd + zy + zx
        x_hat_mu, x_hat_std = self.px(z)
        p_xhat = torch.distributions.Normal(x_hat_mu, x_hat_std)

        if d is not None:
            pzd_mu, pzd_std = self.pzd(d)
            pzd = torch.distributions.Normal(pzd_mu, pzd_std)
        else:
            pzd = None

        if y is not None:
            pzy_mu, pzy_std = self.pzy(y)
            pzy = torch.distributions.Normal(pzy_mu, pzy_std)
        else:
            pzy = None

        pzx = torch.distributions.Normal(
            torch.zeros_like(zx_mu), torch.ones_like(zx_std)
        )

        d_hat = self.d(zd_mu)
        y_hat = self.y(zy_mu)

        return y_hat, d_hat, p_xhat, qzx, pzx, zx, qzy, pzy, zy, qzd, pzd, zd


def train_DIVA(
        device, model, data, optimizer, labeled_idx, 
        all_idx=None, beta_klzd=1, beta_klzx=1, beta_klzy=1, 
        beta_d=1, beta_y=1, beta_recon=1
    ):
    model.to(device)
    model.train()
    optimizer.zero_grad()

    x = data.x.to(device)
    real_d = data.d.to(device)
    real_y = data.y.to(device)

    (pred_y, pred_d, p_xhat, 
    qzx, pzx, zx, qzy, pzy, zy, 
    qzd, pzd, zd) = model(x, real_d, real_y)

    ce_y = F.cross_entropy(pred_y[labeled_idx], real_y[labeled_idx])
    ce_d = F.cross_entropy(pred_d[all_idx], real_d[all_idx])
    recon_loss = -torch.sum(p_xhat.log_prob(x), dim=1)[all_idx].mean()

    kl_zx = torch.sum(qzx.log_prob(zx) - pzx.log_prob(zx), dim=1)[all_idx].mean()
    kl_zy = torch.sum(qzy.log_prob(zy) - pzy.log_prob(zy), dim=1)[all_idx].mean()
    kl_zd = torch.sum(qzd.log_prob(zd) - pzd.log_prob(zd), dim=1)[all_idx].mean()

    loss = (
        beta_y * ce_y + beta_d * ce_d + beta_recon * recon_loss + \
        beta_klzd * kl_zd + beta_klzx * kl_zx + beta_klzy * kl_zy
    )
    loss_val = loss.item()
    loss.backward()
    optimizer.step()

    accuracy = CM.accuracy(real_y[labeled_idx], pred_y[labeled_idx])
    return loss_val, accuracy.item()


def test_DIVA(device, model, data, test_idx):
    model.to(device)
    model.eval()

    x = data.x[test_idx].to(device)
    real_y = data.y[test_idx].to(device)
    res = model(x)
    pred_y = res[0]

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(pred_y, real_y)

    pred_y = pred_y.argmax(dim=1)
    accuracy = CM.accuracy(real_y, pred_y)
    sensitivity = CM.tpr(real_y, pred_y)
    specificity = CM.tnr(real_y, pred_y)
    precision = CM.ppv(real_y, pred_y)
    f1_score = CM.f1_score(real_y, pred_y)
    metrics = {
        "sensitivity": sensitivity.item(),
        "specificity": specificity.item(),
        "f1": f1_score.item(),
        "precision": precision.item()
    }
    return loss.item(), accuracy.item(), metrics
