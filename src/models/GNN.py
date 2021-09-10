import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool


class GNN(torch.nn.Module):
    
    def __init__(self, input_size, hidden, emb1, emb2, l1):
        super().__init__()
        self.emb = torch.nn.Embedding(input_size, hidden, max_norm=1)
        self.conv1 = GraphConv(hidden, emb1)
        self.conv2 = GraphConv(emb1, emb2)

        self.cls1 = torch.nn.Linear(emb2, l1)
        self.cls2 = torch.nn.Linear(l1, 2) # this is the head for disease class

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.emb(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = global_mean_pool(x, batch)  
        x = self.cls1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.cls2(x)
        x = F.softmax(x, dim=1) # output for disease classification
        return x

    def get_embeddings(self, x):
        return self.emb(x)


def train_GNN(device, model, train_dl, optimizer, gamma=0):
    model.to(device)
    model.train()
    optimizer.zero_grad()

    cls_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    gauss_criterion = torch.nn.GaussianNLLLoss(full=True, reduction="sum")

    def _batch_step(batch):
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_weight = batch.edge_attr.to(device)
        batch_idx = batch.batch.to(device)

        x = x.argmax(dim=1)
        pred_y = model(x, edge_index, edge_weight, batch_idx)
        real_y = batch.y.to(device)
        loss = cls_criterion(pred_y, real_y)

        if gamma > 0:
            z = model.get_embeddings(x)
            sim = F.cosine_similarity(z[edge_index[0]], z[edge_index[1]])
            loss += gauss_criterion(edge_weight, sim, torch.ones_like(sim))

        pred = pred_y.argmax(dim=1)
        correct = pred == real_y
        accuracy = correct.float().sum()
        return loss, accuracy

    loss_val = 0
    acc_val = 0
    n = len(train_dl.dataset)

    for batch in train_dl:
        loss, accuracy = _batch_step(batch)
        loss = loss / n
        accuracy = accuracy / n
        loss_val += loss.item()
        acc_val += accuracy.item()
        loss.backward()

    optimizer.step()
    return loss_val, acc_val


def test_GNN(device, model, test_dl):
    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _batch_step(batch):
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_weight = batch.edge_attr.to(device)
        batch_idx = batch.batch.to(device)

        x = x.argmax(dim=1)
        pred_y = model(x, edge_index, edge_weight, batch_idx)
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