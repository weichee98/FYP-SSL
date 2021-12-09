import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_sparse import SparseTensor
from scipy.spatial.distance import squareform
from captum.attr import IntegratedGradients

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.metrics import ClassificationMetrics as CM
from models.base import SaliencyScoreForward


class ChebGCN(torch.nn.Module, SaliencyScoreForward):
    def __init__(self, input_size, hidden, emb1, emb2, K):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden)
        self.conv1 = ChebConv(hidden, emb1, K=K)
        self.conv2 = ChebConv(emb1, emb2, K=K)
        self.linear2 = torch.nn.Linear(emb2, 2)  # this is the head for disease class

    def forward(self, x, adj_t):
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.linear2(x)
        x = F.softmax(x, dim=1)
        return x

    def ss_forward(self, x, adj_t):
        return self.forward(x, adj_t)

    def get_baselines_inputs(self, data):
        x, y = data.x, data.y
        baselines = x[y == 0].mean(dim=0).view(1, -1)
        inputs = x[y == 1]

        adj_t = data.adj_t.to_dense()
        baselines_adj_t = SparseTensor.from_dense(
            torch.ones((1, 1)).to(adj_t.device), has_value=True
        )
        inputs_adj_t = adj_t[y == 1, :]
        inputs_adj_t = inputs_adj_t[:, y == 1]
        inputs_adj_t = SparseTensor.from_dense(inputs_adj_t, has_value=True)
        return (baselines, baselines_adj_t), (inputs, inputs_adj_t)


def train_GCN(device, model, data, optimizer, train_idx, weight=False):
    model.to(device)
    model.train()
    optimizer.zero_grad()  # Clear gradients

    pred_y = model(data.x.to(device), data.adj_t.to(device))
    real_y = data.y[train_idx].to(device)
    if weight:
        _, counts = torch.unique(real_y, sorted=True, return_counts=True)
        weight = counts[[1, 0]] / counts.sum()
    else:
        weight = None
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    loss = criterion(pred_y[train_idx], real_y)
    loss_val = loss.item()
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    accuracy = CM.accuracy(real_y, pred_y[train_idx])
    sensitivity = CM.tpr(real_y, pred_y[train_idx])
    specificity = CM.tnr(real_y, pred_y[train_idx])
    precision = CM.ppv(real_y, pred_y[train_idx])
    f1_score = CM.f1_score(real_y, pred_y[train_idx])
    metrics = {
        "sensitivity": sensitivity.item(),
        "specificity": specificity.item(),
        "f1": f1_score.item(),
        "precision": precision.item(),
    }
    return loss_val, accuracy.item(), metrics


def test_GCN(device, model, data, test_idx):
    model.to(device)
    model.eval()

    pred_y = model(data.x.to(device), data.adj_t.to(device))
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
