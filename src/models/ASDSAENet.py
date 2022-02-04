import os
import sys
import torch
import torch.nn.functional as F

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.metrics import ClassificationMetrics as CM


class SAE(torch.nn.Module):
    def __init__(self, input_size=9500, emb=4975):
        super().__init__()
        self.encoder = torch.nn.Linear(input_size, emb)
        self.decoder = torch.nn.Linear(emb, input_size)

    def forward(self, x):
        z = self.encoder(x)
        z = F.relu(z)
        x_hat = self.decoder(z)
        x_hat = torch.tanh(x_hat)
        return z, x_hat

    def get_optimizer(model, lr=0.0001, lmbda=0.0001):
        return torch.optim.Adam(
            map(
                lambda p: p[1],
                filter(lambda p: p[1].requires_grad, model.named_parameters(),),
            ),
            lr=lr,
            weight_decay=lmbda,
        )


class MaskedSAE(SAE):
    def __init__(self, input_size=19000, emb=4975, mask_ratio=0.5):
        self.num_features = int(round(input_size * mask_ratio))
        super().__init__(self.num_features, emb)
        self.mask_fitted = False
        self.mask = torch.zeros(input_size, dtype=torch.bool, requires_grad=False)

    def forward(self, x):
        if not self.mask_fitted:
            raise Exception("call train_mask first")
        x = x[:, self.mask]
        z = self.encoder(x)
        z = F.relu(z)
        x_hat = self.decoder(z)
        x_hat = torch.tanh(x_hat)
        return z, x, x_hat

    def train_mask(self, x):
        n_smallest = int(self.num_features / 2)
        n_largest = self.num_features - n_smallest

        mean = torch.mean(x, dim=0)
        n_largest_idx = torch.topk(mean, n_largest)[1]
        n_smallest_idx = torch.topk(mean, n_smallest, largest=False)[1]
        self.mask[n_largest_idx] = True
        self.mask[n_smallest_idx] = True

        self.mask_fitted = True
        return self


class FCNN(torch.nn.Module):
    def __init__(self, input_size=4975, l1=2487, l2=500):
        super().__init__()
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(input_size, l1),
            torch.nn.ReLU(),
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, 2),
        )

    def forward(self, x):
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


class ASDSAENet(torch.nn.Module):
    def __init__(self, sae: SAE, cls: FCNN):
        super().__init__()
        self.sae = sae
        self.cls = cls

    def forward(self, x):
        z, *x_hat = self.sae(x)
        y = self.cls(z)
        return y, z, *x_hat


def train_SAE(device, model: SAE, data, optimizer, train_idx, beta=2, p=0.05):
    model.to(device)
    model.train()
    optimizer.zero_grad()

    x = data.x[train_idx].to(device)
    z, pred_x = model(x)

    rc_loss = (x - pred_x) ** 2
    rc_loss = rc_loss.sum(dim=1).mean()

    if beta > 0:
        p_hat = (z ** 2).mean()
        sparsity = torch.sum(
            p * torch.log(p / p_hat) + (1 - p) * torch.log((1 - p) / (1 - p_hat))
        )
        loss = rc_loss + beta * sparsity
    else:
        loss = rc_loss
    loss_val = loss.item()
    loss.backward()
    optimizer.step()

    return loss_val


def train_MaskedSAE(
    device, model: MaskedSAE, data, optimizer, train_idx, beta=2, p=0.05
):
    model.to(device)
    model.train()
    optimizer.zero_grad()

    x = data.x[train_idx].to(device)
    if not model.mask_fitted:
        model.train_mask(x)
    z, masked_x, pred_x = model(x)

    rc_loss = (masked_x - pred_x) ** 2
    rc_loss = rc_loss.sum(dim=1).mean()

    if beta > 0:
        p_hat = z.mean(dim=0) ** 2
        sparsity = torch.sum(
            p * torch.log(p / p_hat) + (1 - p) * torch.log((1 - p) / (1 - p_hat))
        )
        loss = rc_loss + beta * sparsity
    else:
        loss = rc_loss
    loss_val = loss.item()
    loss.backward()
    optimizer.step()

    return loss_val


def test_SAE(device, model: SAE, data, test_idx):
    model.to(device)
    model.eval()

    x = data.x[test_idx].to(device)
    _, pred_x = model(x)
    rc_loss = (x - pred_x) ** 2
    rc_loss = rc_loss.sum(dim=1).mean()

    return rc_loss.item()


def test_MaskedSAE(device, model: MaskedSAE, data, test_idx):
    model.to(device)
    model.eval()

    x = data.x[test_idx].to(device)
    _, masked_x, pred_x = model(x)
    rc_loss = (masked_x - pred_x) ** 2
    rc_loss = rc_loss.sum(dim=1).mean()

    return rc_loss.item()


def train_FCNN(device, model: FCNN, sae: SAE, data, optimizer, train_idx):
    sae.to(device)
    sae.eval()
    model.to(device)
    model.train()
    optimizer.zero_grad()

    cls_criterion = torch.nn.CrossEntropyLoss()

    x = data.x[train_idx].to(device)
    real_y = data.y[train_idx].to(device)

    z, *_ = sae(x)
    z = z.detach()
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


def test_FCNN(device, model: FCNN, sae: SAE, data, test_idx):
    sae.to(device)
    sae.eval()
    model.to(device)
    model.eval()

    x = data.x[test_idx].to(device)
    real_y = data.y[test_idx].to(device)
    z, *_ = sae(x)
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
