import torch
import torch.nn.functional as F


class AE(torch.nn.Module):
    
    def __init__(self, input_size, l1, l2, emb_size, l3, l4):
        """
        l1, l2: number of nodes in the hidden layer of encoder and decoder
        emb_size: size of encoder output and decoder input
        l3, l4: number of nodes in the hidden layer of classifier
        """
        super().__init__()
        self.encoder1 = torch.nn.Linear(input_size, l1)
        self.encoder2 = torch.nn.Linear(l1, l2)
        self.encoder3 = torch.nn.Linear(l2, emb_size)
        self.decoder1 = torch.nn.Linear(emb_size, l2)
        self.decoder2 = torch.nn.Linear(l2, l1)
        self.decoder3 = torch.nn.Linear(l1, input_size)
        self.cls1 = torch.nn.Linear(emb_size, l3)
        self.cls2 = torch.nn.Linear(l3, l4)
        self.cls3 = torch.nn.Linear(l4, 2) # this is the head for disease class

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
        y = F.softmax(x, dim=1) # output for disease classification
        return y, x_

    def _encode(self, x):
        x = self.encoder1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.encoder2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.encoder3(x)
        return x

    def _decode(self, x):
        x = self.decoder1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.decoder2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.decoder3(x)
        x_max = x.abs().max(dim=1)
        x = x / x_max
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
    rc_criterion = torch.nn.MSELoss()

    x = data.x.to(device)
    pred_y, pred_x = model(x)
    real_y = data.y[labeled_idx].to(device)
    cls_loss = cls_criterion(pred_y[labeled_idx], real_y)
    loss = cls_loss

    if all_idx is None:
        all_idx = labeled_idx
    if gamma > 0:
        rc_loss = rc_criterion(pred_x[all_idx], x[all_idx])
        loss += gamma * rc_loss
    
    loss_val = loss.item()
    loss.backward()     # Derive gradients.
    optimizer.step()    # Update parameters based on gradients.
    
    pred = pred_y.argmax(dim=1)
    correct = pred[labeled_idx] == real_y
    accuracy = correct.float().mean()
    return loss_val, accuracy.item()


def test_AE(device, model, data, test_idx):
    model.to(device)
    model.eval()

    pred_y, _ = model(data.x.to(device))
    real_y = data.y[test_idx].to(device)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(pred_y[test_idx], real_y)
    
    pred = pred_y.argmax(dim=1)
    correct = pred[test_idx] == real_y
    accuracy = correct.float().mean()
    return loss.item(), accuracy.item()