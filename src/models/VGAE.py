import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool

__dir__ = os.path.dirname(os.path.dirname(__file__))
sys.path.append(__dir__)

from loss import GaussianKLDivLoss


class VGAE(torch.nn.Module):
    
    def __init__(self, input_size, hidden, emb1, emb2, l1):
        super().__init__()
        self.emb = torch.nn.Embedding(input_size, hidden, max_norm=1)
        self.encoder1 = GraphConv(hidden, emb1)
        self.encoder_mu = GraphConv(emb1, emb2)
        self.encoder_std = GraphConv(emb1, emb2)

        self.cls1 = torch.nn.Linear(emb2, l1)
        self.cls2 = torch.nn.Linear(l1, 2)  # this is the head for disease class
        self.log_std = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.emb(x)
        x = F.dropout(x, p=0.5, training=self.training)

        z_mu, z_log_std = self._encode(x, edge_index, edge_weight)
        z_std = torch.exp(z_log_std)
        q = torch.distributions.Normal(z_mu, z_std)
        z = q.rsample()

        w_mu = self._decode(z, edge_index)
        w_std = torch.exp(self.log_std)

        y = global_mean_pool(z_mu, batch)  
        y = self.cls1(y)
        y = F.relu(y)
        y = F.dropout(y, p=0.5, training=self.training)

        y = self.cls2(y)
        y = F.softmax(y, dim=1) # output for disease classification
        return y, w_mu, w_std, z, z_mu, z_std

    def _encode(self, x, edge_index, edge_weight):
        x = self.encoder1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        mu = self.encoder_mu(x, edge_index, edge_weight)
        mu = torch.tanh(mu)
        log_std = self.encoder_std(x, edge_index, edge_weight)
        log_std = torch.tanh(log_std)
        return mu, log_std

    def _decode(self, x, edge_index):
        x = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]])
        # x = x[edge_index[0]] * x[edge_index[1]]
        # x = torch.tanh(x.sum(dim=1))
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

    def _batch_step(batch, labeled=True):
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_weight = batch.edge_attr.to(device)
        batch_idx = batch.batch.to(device)

        x = x.argmax(dim=1)
        pred_y, w_mu, w_std, z, z_mu, z_std = model(
            x, edge_index, edge_weight, batch_idx
        )
        real_y = batch.y.to(device)

        if labeled:
            cls_loss = cls_criterion(pred_y, real_y)
            pred = pred_y.argmax(dim=1)
            correct = pred == real_y
            accuracy = correct.float().sum()
        else:
            cls_loss = None
            accuracy = None
        w_std = w_std.expand(w_mu.size())
        rc_loss = -torch.sum((edge_weight / 2 + 0.5) * torch.log(w_mu / 2 + 0.5))
        # rc_loss = gauss_criterion(edge_weight, w_mu, w_std ** 2)
        kl = kl_criterion(z_mu, z_std ** 2, torch.zeros_like(z_mu), torch.ones_like(z_std))
        return cls_loss, rc_loss, kl, accuracy

    loss_val = 0
    acc_val = 0
    n_labeled = len(labeled_dl.dataset)
    n_unlabeled = 0. if unlabeled_dl is None else len(unlabeled_dl.dataset)
    n_all = n_labeled + n_unlabeled

    for batch in labeled_dl:
        cls_loss, rc_loss, kl, accuracy = _batch_step(batch, True)
        loss = cls_loss / n_labeled + \
            (gamma1 * rc_loss + gamma2 * kl) / n_all
        loss_val += loss.item()
        acc_val += accuracy.item() / n_labeled
        loss.backward()

    if unlabeled_dl is not None:
        for batch in unlabeled_dl:
            _, rc_loss, kl, _ = _batch_step(batch, False)
            loss = (gamma1 * rc_loss + gamma2 * kl) / n_all
            loss_val += loss.item()
            loss.backward()

    optimizer.step()
    return loss_val, acc_val


def test_VGAE(device, model, test_dl):
    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _batch_step(batch):
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_weight = batch.edge_attr.to(device)
        batch_idx = batch.batch.to(device)

        x = x.argmax(dim=1)
        pred_y, _, _, _, _, _ = model(
            x, edge_index, edge_weight, batch_idx
        )
        real_y = batch.y.to(device)
        loss = criterion(pred_y, real_y)
        pred = pred_y.argmax(dim=1)
        correct = pred == real_y
        accuracy = correct.float().sum()
        return loss, accuracy

    loss_val = 0
    acc_val = 0
    n = len(test_dl.dataset)
    for batch in test_dl:
        loss, accuracy = _batch_step(batch)
        loss_val += loss.item() / n
        acc_val += accuracy.item() / n

    return loss_val, acc_val