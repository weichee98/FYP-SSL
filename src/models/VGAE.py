import os
import sys
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn import GraphConv

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.loss import GaussianKLDivLoss
from utils.metrics import CummulativeClassificationMetrics


class GlobalAttentionPooling(torch.nn.Module):

    def __init__(self, input_size, l1):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, l1)
        self.gate = GraphConv(l1, 1)

    def forward(self, x, edge_index, edge_weight, batch, size=None):
        x1 = self.linear(x)
        gate = self.gate(x1, edge_index, edge_weight)
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)
        return out


class VGAE(torch.nn.Module):
    
    def __init__(self, input_size, emb1, emb2, l1):
        super().__init__()
        self.encoder1 = GraphConv(input_size, emb1)
        self.encoder_mu = GraphConv(emb1, emb2)
        self.encoder_std = torch.nn.Linear(emb1, emb2)
        self.pool = GlobalAttentionPooling(emb2, l1)
        self.cls = torch.nn.Linear(emb2, 2)  # this is the head for disease class
        self.log_std = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, edge_index, edge_weight, batch):
        z_mu, z_log_std = self._encode(x, edge_index, edge_weight)
        z_std = torch.exp(z_log_std)

        if self.training:
            q = torch.distributions.Normal(z_mu, z_std)
            z = q.rsample()
        else:
            z = z_mu

        w_mu = self._decode(z, edge_index)
        w_std = torch.exp(self.log_std)

        y = self.pool(z, edge_index, edge_weight, batch)
        y = self.cls(y)
        y = F.softmax(y, dim=1) # output for disease classification
        return y, w_mu, w_std, z, z_mu, z_std

    def _encode(self, x, edge_index, edge_weight):
        x = self.encoder1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        mu = self.encoder_mu(x, edge_index, edge_weight)
        mu = torch.tanh(mu)
        log_std = self.encoder_std(x)
        log_std = torch.tanh(log_std)
        return mu, log_std

    def _decode(self, x, edge_index):
        x = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]])
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

    def _batch_step(batch, labeled=True):
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_weight = batch.edge_attr.to(device)
        batch_idx = batch.batch.to(device)

        pred_y, w_mu, w_std, z, z_mu, z_std = model(
            x, edge_index, edge_weight, batch_idx
        )
        real_y = batch.y.to(device)

        if labeled:
            cls_loss = cls_criterion(pred_y, real_y)
            ccm.update_batch(real_y, pred_y)
        else:
            cls_loss = None
        w_std = w_std.expand(w_mu.size())
        rc_loss = gauss_criterion(edge_weight, w_mu, w_std ** 2)
        kl = kl_criterion(z_mu, z_std ** 2, torch.zeros_like(z_mu), torch.ones_like(z_std))
        return cls_loss, rc_loss, kl

    loss_val = 0
    n_labeled = len(labeled_dl.dataset)
    n_unlabeled = 0. if unlabeled_dl is None else len(unlabeled_dl.dataset)
    n_all = n_labeled + n_unlabeled

    for batch in labeled_dl:
        cls_loss, rc_loss, kl = _batch_step(batch, True)
        loss = cls_loss / n_labeled + \
            (gamma1 * rc_loss + gamma2 * kl) / n_all
        loss_val += loss.item()
        loss.backward()

    if unlabeled_dl is not None:
        for batch in unlabeled_dl:
            _, rc_loss, kl = _batch_step(batch, False)
            loss = (gamma1 * rc_loss + gamma2 * kl) / n_all
            loss_val += loss.item()
            loss.backward()

    optimizer.step()
    acc_val = ccm.accuracy.item()
    return loss_val, acc_val


def test_VGAE(device, model, test_dl):
    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    ccm = CummulativeClassificationMetrics()

    def _batch_step(batch):
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_weight = batch.edge_attr.to(device)
        batch_idx = batch.batch.to(device)

        pred_y, _, _, _, _, _ = model(
            x, edge_index, edge_weight, batch_idx
        )
        real_y = batch.y.to(device)
        loss = criterion(pred_y, real_y)
        ccm.update_batch(real_y, pred_y)
        return loss

    loss_val = 0
    n = len(test_dl.dataset)
    for batch in test_dl:
        loss = _batch_step(batch)
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