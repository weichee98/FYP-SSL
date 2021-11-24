import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.metrics import ClassificationMetrics as CM


class ChebGCN(torch.nn.Module):
    
    def __init__(self, input_size, hidden, emb1, emb2, K):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden)
        self.conv1 = ChebConv(hidden, emb1, K=K)
        self.conv2 = ChebConv(emb1, emb2, K=K)
        self.linear2 = torch.nn.Linear(emb2, 2) # this is the head for disease class
            
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


def train_GCN(device, model, data, optimizer, train_idx):
    model.to(device)
    model.train()
    optimizer.zero_grad()  # Clear gradients

    pred_y = model(data.x.to(device), data.adj_t.to(device))
    real_y = data.y[train_idx].to(device)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(pred_y[train_idx], real_y)
    loss_val = loss.item()
    loss.backward()     # Derive gradients.
    optimizer.step()    # Update parameters based on gradients.
    
    accuracy = CM.accuracy(real_y, pred_y[train_idx])
    return loss_val, accuracy.item()


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
        "precision": precision.item()
    }
    return loss.item(), accuracy.item(), metrics