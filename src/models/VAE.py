import torch
import torch.nn.functional as F


class VAE(torch.nn.Module):
    
    def __init__(self, input_size, l1, l2, emb_size, l3, l4):
        """
        l1, l2: number of nodes in the hidden layer of encoder and decoder
        emb_size: size of encoder output and decoder input
        l3, l4: number of nodes in the hidden layer of classifier
        """
        super().__init__()
        self.encoder1 = torch.nn.Linear(input_size, l1)
        self.encoder2 = torch.nn.Linear(l1, l2)
        self.encoder_mu = torch.nn.Linear(l2, emb_size)
        self.encoder_std = torch.nn.Linear(l2, emb_size)

        self.decoder1 = torch.nn.Linear(emb_size, l2)
        self.decoder2 = torch.nn.Linear(l2, l1)
        self.decoder_mu = torch.nn.Linear(l1, input_size)
        self.decoder_std = torch.nn.Linear(l1, input_size)

        self.cls1 = torch.nn.Linear(emb_size, l3)
        self.cls2 = torch.nn.Linear(l3, l4)
        self.cls3 = torch.nn.Linear(l4, 2) # this is the head for disease class

    def forward(self, x):
        z_mu, z_log_std = self._encode(x)
        z_std = torch.exp(z_log_std)
        q = torch.distributions.Normal(z_mu, z_std)
        z = q.rsample()

        x_mu, x_log_std = self._decode(z)
        x_std = torch.exp(x_log_std)

        y = self.cls1(z_mu)
        y = F.relu(y)
        y = F.dropout(y, p=0.5, training=self.training)

        y = self.cls2(y)
        y = F.relu(y)
        y = F.dropout(y, p=0.5, training=self.training)

        y = self.cls3(y)
        y = F.softmax(y, dim=1) # output for disease classification
        return y, x_mu, x_std, z, z_mu, z_std

    def _encode(self, x):
        x = self.encoder1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.encoder2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        mu = self.encoder_mu(x)
        log_std = self.encoder_std(x)
        log_std = torch.tanh(log_std)
        return mu, log_std

    def _decode(self, x):
        x = self.decoder1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.decoder2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        mu = self.decoder_mu(x)
        mu = torch.tanh(mu)
        log_std = self.decoder_std(x)
        log_std = torch.tanh(log_std)
        return mu, log_std


def train_VAE(
        device, model, data, optimizer, labeled_idx, 
        all_idx=None, gamma1=0, gamma2=0
    ):
    """
    all_idx: the indices of labeled and unlabeled data (exclude test indices)
    gamma1: float, the weightage of reconstruction loss
    gamma2: float, the weightage of regularizer (kl divergence)
    """
    model.to(device)
    model.train()
    optimizer.zero_grad()

    cls_criterion = torch.nn.CrossEntropyLoss()
    gauss_criterion = torch.nn.GaussianNLLLoss(full=True)

    x = data.x.to(device)
    pred_y, x_mu, x_std, z, z_mu, z_std = model(x)
    real_y = data.y[labeled_idx].to(device)

    loss = cls_criterion(pred_y[labeled_idx], real_y)

    if all_idx is None:
        all_idx = labeled_idx
    rc_loss = gauss_criterion(x[all_idx], x_mu[all_idx], x_std[all_idx] ** 2)
    kl = torch.sum(z_std ** 2 + z_mu ** 2 - z_std.log() - 1, dim=1).mean()
    loss += gamma1 * rc_loss + gamma2 * kl

    loss_val = loss.item()
    loss.backward()
    optimizer.step()
    
    pred = pred_y.argmax(dim=1)
    correct = pred[labeled_idx] == real_y
    accuracy = correct.float().mean()
    return loss_val, accuracy.item()


def test_VAE(device, model, data, test_idx):
    model.to(device)
    model.eval()

    pred_y, _, _, _, _, _ = model(data.x.to(device))
    real_y = data.y[test_idx].to(device)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(pred_y[test_idx], real_y)
    
    pred = pred_y.argmax(dim=1)
    correct = pred[test_idx] == real_y
    accuracy = correct.float().mean()
    return loss.item(), accuracy.item()