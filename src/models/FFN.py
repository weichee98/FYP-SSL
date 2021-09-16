import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.loss import LaplacianRegularization
from utils.metrics import ClassificationMetrics as CM


class FFN(torch.nn.Module):
    
    def __init__(self, input_size, l1, l2, l3):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, l3)
        self.linear4 = torch.nn.Linear(l3, 2) # this is the head for disease class
            
    def forward(self, x): # full batch
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.linear2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.linear3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.linear4(x)
        x = F.softmax(x, dim=1) # output for disease classification        
        return x


def get_laplacian_regularization(device, data, laplacian_idx, y):
    laplacian_idx = torch.tensor(laplacian_idx).long()
    edge_index, edge_weights = subgraph(
        subset=laplacian_idx,
        edge_index = data.edge_index,
        edge_attr = data.edge_attr,
        relabel_nodes=True
    )
    criterion = LaplacianRegularization(normalization="sym", p=2)
    loss = criterion(edge_index.to(device), edge_weights.to(device), y[laplacian_idx])
    return loss


def train_FFN(device, model, data, optimizer, labeled_idx, 
              all_idx=None, gamma_lap=0):
    """
    all_idx: the indices of labeled and unlabeled data
                   (exclude test indices)
    gamma_lap: float, the weightage of laplacian regularization
    """
    model.to(device)
    model.train()
    optimizer.zero_grad()  # Clear gradients

    pred_y = model(data.x.to(device))
    real_y = data.y[labeled_idx].to(device)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(pred_y[labeled_idx], real_y)
    
    if all_idx is not None and gamma_lap > 0:
        loss += gamma_lap * get_laplacian_regularization(
            device, data, all_idx, pred_y
        )
    
    loss_val = loss.item()
    loss.backward()     # Derive gradients.
    optimizer.step()    # Update parameters based on gradients.
    
    accuracy = CM.accuracy(real_y, pred_y[labeled_idx])
    return loss_val, accuracy.item()


def test_FFN(device, model, data, test_idx):
    model.to(device)
    model.eval()

    pred_y = model(data.x.to(device))
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