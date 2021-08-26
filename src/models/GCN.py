import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class ChebGCN(torch.nn.Module):
    
    def __init__(self, input_size, hidden, emb1, emb2, K):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden)
        self.conv1 = ChebConv(hidden, emb1, K=K)
        self.conv2 = ChebConv(emb1, emb2, K=K)
        self.linear2 = torch.nn.Linear(emb2, 2) # this is the head for disease class
            
    def forward(self, x, edge_index, edge_weight):
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.linear2(x)
        x = F.softmax(x, dim=1)
        return x


def train_GCN(device, model, data, optimizer, train_idx):
    model.to(device)
    model.train()
    optimizer.zero_grad()  # Clear gradients

    pred_y = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
    real_y = data.y[train_idx].to(device)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(pred_y[train_idx], real_y)
    loss_val = loss.item()
    loss.backward()     # Derive gradients.
    optimizer.step()    # Update parameters based on gradients.
    
    pred = pred_y.argmax(dim=1)
    correct = pred[train_idx] == real_y
    accuracy = correct.float().mean()
    return loss_val, accuracy.item()


def test_GCN(device, model, data, test_idx):
    model.to(device)
    model.eval()

    pred_y = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
    real_y = data.y[test_idx].to(device)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(pred_y[test_idx], real_y)
    
    pred = pred_y.argmax(dim=1)
    correct = pred[test_idx] == real_y
    accuracy = correct.float().mean()
    return loss.item(), accuracy.item()