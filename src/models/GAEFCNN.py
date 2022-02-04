import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.metrics import ClassificationMetrics as CM
from utils.loss import GaussianKLDivLoss


def threshold(adj_t, tau):
    with torch.no_grad():
        adj = adj_t.to_dense().abs()
        q = torch.quantile(adj, q=tau)
        adj = (adj >= q).type(adj.dtype)
        adj_t = SparseTensor.from_dense(adj, has_value=False)
        return adj_t


class GCN(torch.nn.Module):
    def __init__(self, input_size=116, emb1=64, emb2=16, tau=0.0):
        super().__init__()
        self.encoder1 = GCNConv(input_size, emb1)
        if emb2 > 0:
            self.encoder2 = GCNConv(emb1, emb2)
        else:
            self.encoder2 = None
        self.tau = tau

    def forward(self, x, adj_t):
        if self.tau > 0.0:
            adj_t = threshold(adj_t, 1.0 - self.tau)
        z = self.encoder1(x, adj_t)
        z = F.relu(z)
        if self.encoder2 is not None:
            z = self.encoder2(z, adj_t)
            z = F.relu(z)
        return z

    def get_optimizer(model, lr=0.0001, lmbda=0.0001):
        return torch.optim.Adam(
            map(
                lambda p: p[1],
                filter(lambda p: p[1].requires_grad, model.named_parameters(),),
            ),
            lr=lr,
            weight_decay=lmbda,
        )


class GCNAE(torch.nn.Module):
    def __init__(self, input_size=116, emb1=64, emb2=16, tau=0.25):
        super().__init__()
        self.encoder = GCN(input_size, emb1, emb2)
        self.tau = tau

    def forward(self, x, adj_t):
        adj_t = threshold(adj_t, 1.0 - self.tau)
        z = self.encode(x, adj_t)
        A = self.decode(z, adj_t)
        return z, A

    def encode(self, x, adj_t):
        z = self.encoder(x, adj_t)
        return z

    def decode(self, z, adj_t):
        row, col, _ = adj_t.coo()
        A = (z[row] * z[col]).sum(dim=1)
        A = torch.sigmoid(A)
        return A

    def get_optimizer(model, lr=0.0001, lmbda=0.0001):
        return torch.optim.Adam(
            map(
                lambda p: p[1],
                filter(lambda p: p[1].requires_grad, model.named_parameters(),),
            ),
            lr=lr,
            weight_decay=lmbda,
        )


class VGCNAE(GCNAE):
    def __init__(self, input_size=116, emb1=64, emb2=16, tau=0.25):
        super().__init__(input_size, emb1, emb2, tau)
        self.log_std_encoder = GCN(input_size, emb1, emb2)

    def forward(self, x, adj_t):
        adj_t = threshold(adj_t, 1.0 - self.tau)
        z_mu, z_std = self.encode(x, adj_t)
        if self.training:
            q = torch.distributions.Normal(z_mu, z_std)
            z = q.rsample()
        else:
            z = z_mu
        A = self.decode(z, adj_t)
        return z, A, z_mu, z_std

    def encode(self, x, adj_t):
        z_mu = self.encoder(x, adj_t)
        z_log_std = self.log_std_encoder(x, adj_t)
        z_std = torch.exp(z_log_std)
        return z_mu, z_std


class GFCNN(torch.nn.Module):
    def __init__(self, input_size=16, num_nodes=116, l1=256, l2=256, l3=128):
        super().__init__()
        layers = [
            torch.nn.Linear(input_size * num_nodes, l1),
            torch.nn.ReLU(),
            torch.nn.Linear(l1, l2),
        ]
        if l3 > 0:
            layers += [
                torch.nn.ReLU(),
                torch.nn.Linear(l2, l3),
                torch.nn.ReLU(),
                torch.nn.Linear(l3, 2),
            ]
        else:
            layers += [
                torch.nn.ReLU(),
                torch.nn.Linear(l2, 2),
            ]
        self.clf = torch.nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 2:
            x = torch.flatten(x).unsqueeze(0)
        elif x.ndim > 2:
            x = torch.flatten(x, start_dim=1)
        else:
            raise ValueError("invalid x with shape {}".format(x.size()))

        y = self.clf(x)
        y = F.softmax(y, dim=1)
        return y

    def get_optimizer(model, lr=0.0001, lmbda=0.0001):
        return torch.optim.Adam(
            map(
                lambda p: p[1],
                filter(lambda p: p[1].requires_grad, model.named_parameters(),),
            ),
            lr=lr,
            weight_decay=lmbda,
        )


class GAEFCNN(torch.nn.Module):

    def __init__(self, gae, fcnn: GFCNN):
        super().__init__()
        self.gae = gae
        self.fcnn = fcnn

    def forward(self, x, adj_t):
        if isinstance(self.gae, GCN):
            z = self.gae(x, adj_t)
            out = list()
        elif isinstance(self.gae, (VGCNAE, GCNAE)):
            z, *out = self.gae(x, adj_t)
        else:
            raise TypeError("invalid gae of type {}".format(self.gae.__class__))
        y = self.fcnn(z)
        return y, z, *out


def train_GCNAE(device, model: GCNAE, train_dl, optimizer):
    model.to(device)
    model.train()
    optimizer.zero_grad()

    kl_criterion = GaussianKLDivLoss(reduction="mean")

    def _step(data):
        x = data.x.to(device)
        adj_t = data.adj_t.to(device)

        if isinstance(model, VGCNAE):
            _, A, z_mu, z_std = model(x, adj_t)
        elif isinstance(model, GCNAE):
            _, A = model(x, adj_t)
        else:
            raise TypeError("invalid model type {}".format(model.__class__))

        loss = -A.log().mean()
        if isinstance(model, VGCNAE):
            loss += kl_criterion(
                z_mu, z_std ** 2, torch.zeros_like(z_mu), torch.ones_like(z_std)
            )
        return loss

    loss_val = 0
    n = len(train_dl)
    for data in train_dl:
        loss = _step(data) / n
        loss_val += loss.item()
        loss.backward()
    optimizer.step()
    return loss_val


def test_GCNAE(device, model: GCNAE, test_dl):
    model.to(device)
    model.eval()

    def _step(data):
        x = data.x.to(device)
        adj_t = data.adj_t.to(device)

        if isinstance(model, VGCNAE):
            _, A, *_ = model(x, adj_t)
        elif isinstance(model, GCNAE):
            _, A = model(x, adj_t)
        else:
            raise TypeError("invalid model type {}".format(model.__class__))

        loss = -A.log().mean()
        return loss

    loss_val = 0
    n = len(test_dl)
    for data in test_dl:
        loss = _step(data) / n
        loss_val += loss.item()
    return loss_val


def train_GFCNN(device, model: GFCNN, gae: GCNAE, train_dl, optimizer):
    gae.to(device)
    gae.train()
    model.to(device)
    model.train()
    optimizer.zero_grad()

    cls_criterion = torch.nn.CrossEntropyLoss()
    z = torch.stack(
        tuple(gae(data.x.to(device), data.adj_t.to(device))[0] for data in train_dl), dim=0
    )
    z = z.detach()
    real_y = torch.cat(tuple(data.y.to(device) for data in train_dl), dim=0)
    pred_y = model(z)
    loss = cls_criterion(pred_y, real_y)
    loss_val = loss.item()
    loss.backward()
    optimizer.step()

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
    return loss_val, accuracy.item(), metrics


def test_GFCNN(device, model: GFCNN, gae: GCNAE, test_dl):
    gae.to(device)
    gae.eval()
    model.to(device)
    model.eval()

    z = torch.stack(
        tuple(gae(data.x.to(device), data.adj_t.to(device))[0] for data in test_dl), dim=0
    )
    real_y = torch.cat(tuple(data.y.to(device) for data in test_dl), dim=0)
    pred_y = model(z)

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(pred_y, real_y)

    pred_y = pred_y.argmax(dim=1)
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


def train_GCNFCNN(
    device, gcn: GCN, model: GFCNN, train_dl, gcn_optimizer, model_optimizer
):
    gcn.to(device)
    gcn.train()
    model.to(device)
    model.train()
    gcn_optimizer.zero_grad()
    model_optimizer.zero_grad()

    cls_criterion = torch.nn.CrossEntropyLoss()
    z = torch.stack(
        tuple(gcn(data.x.to(device), data.adj_t.to(device)) for data in train_dl), dim=0
    )
    real_y = torch.cat(tuple(data.y.to(device) for data in train_dl), dim=0)
    pred_y = model(z)
    loss = cls_criterion(pred_y, real_y)
    loss_val = loss.item()
    loss.backward()
    gcn_optimizer.step()
    model_optimizer.step()

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
    return loss_val, accuracy.item(), metrics


def test_GCNFCNN(device, gcn: GCN, model: GFCNN, test_dl):
    gcn.to(device)
    gcn.eval()
    model.to(device)
    model.eval()

    z = torch.stack(
        tuple(gcn(data.x.to(device), data.adj_t.to(device)) for data in test_dl), dim=0
    )
    real_y = torch.cat(tuple(data.y.to(device) for data in test_dl), dim=0)
    pred_y = model(z)

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(pred_y, real_y)

    pred_y = pred_y.argmax(dim=1)
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

