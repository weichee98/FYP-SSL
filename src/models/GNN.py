import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.metrics import CummulativeClassificationMetrics
from models.base import GraphSaliencyScoreForward


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


class GNN(torch.nn.Module, GraphSaliencyScoreForward):
    def __init__(self, input_size, emb1, emb2, l1):
        super().__init__()
        self.conv1 = GraphConv(input_size, emb1)
        self.conv2 = GraphConv(emb1, emb2)
        self.pool = GlobalAttentionPooling(emb2, l1)
        self.cls1 = torch.nn.Linear(emb2, l1)
        self.cls2 = torch.nn.Linear(l1, 2)  # this is the head for disease class

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, p=_DROPOUT, training=self.training)

        x = self.conv2(x, adj_t)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, p=_DROPOUT, training=self.training)

        x = self.pool(x, adj_t)
        x = self.cls1(x)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, p=_DROPOUT, training=self.training)

        x = self.cls2(x)
        x = F.softmax(x, dim=1)  # output for disease classification
        return x

    def ss_forward(self, x, adj_t):
        return self.forward(x, adj_t)


def train_GNN(device, model, train_dl, optimizer, weight=False):
    model.to(device)
    model.train()
    optimizer.zero_grad()

    if weight:
        all_y = torch.cat([data.y for data in train_dl], dim=0)
        _, counts = torch.unique(all_y, sorted=True, return_counts=True)
        weight = counts[[1, 0]] / counts.sum()
    else:
        weight = None

    cls_criterion = torch.nn.CrossEntropyLoss(weight=weight, reduction="sum")
    ccm = CummulativeClassificationMetrics()

    def _step(data):
        x = data.x.to(device)
        adj_t = data.adj_t.to(device)

        pred_y = model(x, adj_t)
        real_y = data.y.to(device)
        loss = cls_criterion(pred_y, real_y)

        ccm.update_batch(real_y, pred_y)
        return loss

    loss_val = 0
    n = len(train_dl)

    for data in train_dl:
        loss = _step(data)
        loss = loss / n
        loss_val += loss.item()
        loss.backward()

    optimizer.step()
    acc_val = ccm.accuracy.item()
    metrics = {
        "sensitivity": ccm.tpr.item(),
        "specificity": ccm.tnr.item(),
        "f1": ccm.f1_score.item(),
        "precision": ccm.ppv.item(),
    }
    return loss_val, acc_val, metrics


def test_GNN(device, model, test_dl):
    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    ccm = CummulativeClassificationMetrics()

    def _step(data):
        x = data.x.to(device)
        adj_t = data.adj_t.to(device)

        pred_y = model(x, adj_t)
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
        "precision": precision.item(),
    }
    return loss_val, acc_val, metrics
