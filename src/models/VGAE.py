import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.loss import GaussianKLDivLoss
from utils.metrics import CummulativeClassificationMetrics


_DROPOUT = 0.1


class GlobalAttentionPooling(torch.nn.Module):

    def __init__(self, input_size, l1):
        super().__init__()
        self.gate1 = GraphConv(input_size, l1)
        self.gate2 = GraphConv(l1, 1)

    def forward(self, x, adj_t):
        x1 = self.gate1(x, adj_t)
        x1 = F.leaky_relu(x1, 0.2)
        x1 = F.dropout(x1, p=_DROPOUT, training=self.training)

        gate = self.gate2(x1, adj_t)
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
        gate = F.softmax(gate, dim=0)
        out = torch.matmul(gate.t(), x)
        return out


class VGAE(torch.nn.Module):
    
    def __init__(self, input_size, emb1, emb2, l1):
        super().__init__()
        self.encoder1 = GraphConv(input_size, emb1)
        self.encoder_mu = GraphConv(emb1, emb2)
        self.encoder_std = torch.nn.Linear(emb1, emb2)
        
        self.pool = GlobalAttentionPooling(emb2, l1)
        self.cls1 = torch.nn.Linear(emb2, l1)
        self.cls2 = torch.nn.Linear(l1, 2)
        self.log_std = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, adj_t):
        z_mu, z_log_std = self._encode(x, adj_t)
        z_std = torch.exp(z_log_std)

        if self.training:
            q = torch.distributions.Normal(z_mu, z_std)
            z = q.rsample()
        else:
            z = z_mu

        w_mu = self._decode(z, adj_t)
        w_std = torch.exp(self.log_std)

        y = self.pool(z, adj_t)
        y = self.cls1(y)
        y = F.leaky_relu(y, 0.2)
        y = F.dropout(y, p=_DROPOUT, training=self.training)

        y = self.cls2(y)
        y = F.softmax(y, dim=1)
        return y, w_mu, w_std, z, z_mu, z_std

    def _encode(self, x, adj_t):
        x = self.encoder1(x, adj_t)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, p=_DROPOUT, training=self.training)

        mu = self.encoder_mu(x, adj_t)
        mu = torch.tanh(mu)
        log_std = self.encoder_std(x)
        log_std = torch.tanh(log_std)
        return mu, log_std

    def _decode(self, x, adj_t):
        row, col, _ = adj_t.coo()
        x = F.cosine_similarity(x[row], x[col])
        return x


def train_VGAE(
        device, model, labeled_dl, unlabeled_dl, optimizer, 
        gamma1=0, gamma2=0
    ):
    model.to(device)
    model.train()
    optimizer.zero_grad()

    cls_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    gauss_criterion = torch.nn.GaussianNLLLoss(full=True, reduction="sum")
    kl_criterion = GaussianKLDivLoss(reduction="sum")
    ccm = CummulativeClassificationMetrics()

    def _step(data, labeled=True):
        x = data.x.to(device)
        adj_t = data.adj_t.to(device)

        pred_y, w_mu, w_std, z, z_mu, z_std = model(x, adj_t)
        real_y = data.y.to(device)
        _, _, value = adj_t.coo()

        if labeled:
            cls_loss = cls_criterion(pred_y, real_y)
            ccm.update_batch(real_y, pred_y)
        else:
            cls_loss = None
        w_std = w_std.expand(w_mu.size())
        rc_loss = gauss_criterion(value, w_mu, w_std ** 2)
        kl = kl_criterion(z_mu, z_std ** 2, torch.zeros_like(z_mu), torch.ones_like(z_std))
        return cls_loss, rc_loss, kl

    loss_val = 0
    n_labeled = len(labeled_dl)
    n_unlabeled = 0. if unlabeled_dl is None else len(unlabeled_dl)
    n_all = n_labeled + n_unlabeled

    for data in labeled_dl:
        cls_loss, rc_loss, kl = _step(data, True)
        loss = cls_loss / n_labeled + \
            (gamma1 * rc_loss + gamma2 * kl) / n_all
        loss_val += loss.item()
        loss.backward()

    if unlabeled_dl is not None:
        for data in unlabeled_dl:
            _, rc_loss, kl = _step(data, False)
            loss = (gamma1 * rc_loss + gamma2 * kl) / n_all
            loss_val += loss.item()
            loss.backward()

    optimizer.step()
    acc_val = ccm.accuracy.item()
    metrics = {
        "sensitivity": ccm.tpr.item(),
        "specificity": ccm.tnr.item(),
        "f1": ccm.f1_score.item(),
        "precision": ccm.ppv.item()
    }
    return loss_val, acc_val, metrics


def test_VGAE(device, model, test_dl):
    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    ccm = CummulativeClassificationMetrics()

    def _step(data):
        x = data.x.to(device)
        adj_t = data.adj_t.to(device)

        pred_y, _, _, _, _, _ = model(x, adj_t)
        real_y = data.y.to(device)
        loss = criterion(pred_y, real_y)
        ccm.update_batch(real_y, pred_y)
        return loss

    loss_val = 0
    n = len(test_dl)
    for data in test_dl:
        loss = _step(data)
        loss_val += loss.item() / n

    acc_val = ccm.accuracy.item()
    sensitivity = ccm.tpr
    specificity = ccm.tnr
    precision = ccm.ppv
    f1_score = ccm.f1_score
    metrics = {
        "sensitivity": sensitivity.item(),
        "specificity": specificity.item(),
        "f1": f1_score.item(),
        "precision": precision.item()
    }
    return loss_val, acc_val, metrics