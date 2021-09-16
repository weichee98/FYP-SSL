import os
import sys
import torch
import torch.nn.functional as F

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.metrics import ClassificationMetrics as CM


class AE(torch.nn.Module):
    
    def __init__(self, input_size, l1, l2, l3, emb_size):
        """
        l1: number of nodes in the hidden layer of encoder and decoder
        emb_size: size of encoder output and decoder input
        l2, l3: number of nodes in the hidden layer of classifier
        """
        super().__init__()
        self.encoder1 = torch.nn.Linear(input_size, l1)
        self.encoder2 = torch.nn.Linear(l1, emb_size)
        self.decoder1 = torch.nn.Linear(emb_size, l1)
        self.decoder2 = torch.nn.Linear(l1, input_size)
        self.cls1 = torch.nn.Linear(emb_size, l2)
        self.cls2 = torch.nn.Linear(l2, l3)
        self.cls3 = torch.nn.Linear(l3, 2) # this is the head for disease class

    def forward(self, x): # full batch
        emb = self._encode(x)
        x_ = self._decode(emb)

        y = self.cls1(emb)
        y = F.relu(y)
        y = F.dropout(y, p=0.5, training=self.training)

        y = self.cls2(y)
        y = F.relu(y)
        y = F.dropout(y, p=0.5, training=self.training)

        y = self.cls3(y)
        y = F.softmax(y, dim=1) # output for disease classification
        return y, x_

    def _encode(self, x):
        x = self.encoder1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.encoder2(x)
        x = F.relu(x)
        return x

    def _decode(self, x):
        x = self.decoder1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.decoder2(x)
        x = torch.tanh(x)
        return x


def train_AE(
        device, model, data, optimizer, labeled_idx, 
        all_idx=None, gamma=0
    ):
    """
    all_idx: the indices of labeled and unlabeled data (exclude test indices)
    gamma: float, the weightage of reconstruction loss
    """
    model.to(device)
    model.train()
    optimizer.zero_grad()  # Clear gradients

    cls_criterion = torch.nn.CrossEntropyLoss()

    x = data.x.to(device)
    pred_y, pred_x = model(x)
    real_y = data.y[labeled_idx].to(device)
    loss = cls_criterion(pred_y[labeled_idx], real_y)

    if all_idx is None:
        all_idx = labeled_idx

    rc_loss = (pred_x[all_idx] - x[all_idx]) ** 2
    rc_loss = rc_loss.sum(dim=1).mean()
    loss += gamma * rc_loss
    
    loss_val = loss.item()
    loss.backward()     # Derive gradients.
    optimizer.step()    # Update parameters based on gradients.
    
    accuracy = CM.accuracy(real_y, pred_y[labeled_idx])
    return loss_val, accuracy.item()


def test_AE(device, model, data, test_idx):
    model.to(device)
    model.eval()

    pred_y, _ = model(data.x.to(device))
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