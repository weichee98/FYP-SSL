import os
import sys
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence
import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, Linear, ReLU, Softmax
from torch.optim import Optimizer, Adam
from torch.distributions import Normal
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.metrics import ClassificationMetrics as CM
from utils.loss import kl_divergence_loss
from models.base import GraphModelBase, ModelBase


def threshold(adj_t: SparseTensor, tau: float) -> SparseTensor:
    with torch.no_grad():
        adj = adj_t.to_dense().abs()
        q = torch.quantile(adj, q=tau)
        adj = (adj >= q).type(adj.dtype)
        adj_t = SparseTensor.from_dense(adj, has_value=False)
        return adj_t


class GCN(Module):
    def __init__(
        self,
        input_size: int = 116,
        emb1: int = 64,
        emb2: int = 16,
        tau: float = 0,
    ):
        super().__init__()
        self.encoder1 = GCNConv(input_size, emb1)
        if emb2 > 0:
            self.encoder2 = GCNConv(emb1, emb2)
        else:
            self.encoder2 = None
        self.tau = tau

    def forward(
        self, x: torch.Tensor, adj_t: SparseTensor
    ) -> Dict[str, torch.Tensor]:
        if self.tau > 0:
            adj_t = threshold(adj_t, 1.0 - self.tau)
        z = self.encoder1(x, adj_t)
        z = F.relu(z)
        if self.encoder2 is not None:
            z = self.encoder2(z, adj_t)
            z = F.relu(z)
        return {"z": z, "adj_t": adj_t}

    def get_optimizer(self, param: dict) -> Optimizer:
        optim = Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=param.get("lr", 0.0001),
            weight_decay=param.get("l2_reg", 0.0001),
        )
        return optim


class GAE(GraphModelBase):
    def __init__(
        self,
        input_size: int = 116,
        emb1: int = 64,
        emb2: int = 16,
        tau: int = 0.25,
        **kwargs
    ):
        super().__init__()
        self.encoder = GCN(input_size, emb1, emb2, tau)

    @staticmethod
    def state_dict_mapping() -> dict:
        return dict()

    def ss_forward(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def encode(
        self, x: torch.Tensor, adj_t: SparseTensor
    ) -> Dict[str, torch.Tensor]:
        return self.encoder(x, adj_t)

    def decode(self, z: torch.Tensor, adj_t: SparseTensor) -> torch.Tensor:
        row, col, _ = adj_t.coo()
        A = (z[row] * z[col]).sum(dim=1)
        A = torch.sigmoid(A)
        return A

    def forward(
        self, x: torch.Tensor, adj_t: SparseTensor
    ) -> Dict[str, torch.Tensor]:
        encodings = self.encode(x, adj_t)
        z, adj_t = encodings["z"], encodings["adj_t"]
        A = self.decode(z, adj_t)
        return {"z": z, "A": A}

    def get_optimizer(self, param: dict) -> Optimizer:
        optim = Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=param.get("lr", 0.0001),
            weight_decay=param.get("l2_reg", 0.0001),
        )
        return optim

    def _accumulate_step(
        self, x: torch.Tensor, adj_t: SparseTensor,
    ) -> Dict[str, torch.Tensor]:
        results: Dict[str, torch.Tensor] = self(x, adj_t)
        A = results["A"]
        rc_loss = -A.log().mean()
        losses = dict()
        losses["rc_loss"] = rc_loss
        losses["total_loss"] = rc_loss
        return losses

    def train_step(
        self,
        device: torch.device,
        labeled_data: Sequence[Data],
        unlabeled_data: Optional[Sequence[Data]],
        optimizer: Optimizer,
        hyperparameters: Dict[str, Any],
    ) -> Dict[str, float]:

        self.to(device)
        self.train()
        results = defaultdict(float)
        n = float(len(labeled_data))

        with torch.enable_grad():
            optimizer.zero_grad()
            for data in labeled_data:
                x: torch.Tensor = data.x
                adj_t: SparseTensor = data.adj_t
                x, adj_t = x.to(device), adj_t.to(device)

                losses = self._accumulate_step(x, adj_t)
                for k, v in losses.items():
                    results[k] += (v / n).item()
                total_loss = losses["total_loss"]
                total_loss.backward()
            optimizer.step()

        return results

    def test_step(
        self, device: torch.device, test_data: Sequence[Data]
    ) -> Dict[str, float]:
        self.to(device)
        self.eval()
        results = defaultdict(float)
        n = float(len(test_data))

        with torch.no_grad():
            for data in test_data:
                x: torch.Tensor = data.x
                adj_t: SparseTensor = data.adj_t
                x, adj_t = x.to(device), adj_t.to(device)

                losses = self._accumulate_step(x, adj_t)
                for k, v in losses.items():
                    results[k] += (v / n).item()
        return results

    def prepare_z_y(
        self, device: torch.device, data_ls: Sequence[Data]
    ) -> Data:
        self.to(device)
        self.eval()

        with torch.no_grad():
            z_ls = list()
            y_ls = list()
            for data in data_ls:
                x: torch.Tensor = data.x
                y: torch.Tensor = data.y
                adj_t: SparseTensor = data.adj_t
                x, y, adj_t = x.to(device), y.to(device), adj_t.to(device)
                z = self(x, adj_t)["z"]
                z_ls.append(z)
                y_ls.append(y)
            Z = torch.stack(z_ls, dim=0).detach()
            Y = torch.cat(y_ls, dim=0)

        reduced_data = Data(y=Y)
        reduced_data.z = Z
        return reduced_data


class GVAE(GAE):
    def __init__(
        self,
        input_size: int = 116,
        emb1: int = 64,
        emb2: int = 16,
        tau: float = 0.25,
        **kwargs
    ):
        super().__init__(input_size, emb1, emb2, tau)
        self.log_std_encoder = GCN(input_size, emb1, emb2)

    def encode(
        self, x: torch.Tensor, adj_t: SparseTensor
    ) -> Dict[str, torch.Tensor]:
        encodings = self.encoder(x, adj_t)
        z_mu = encodings["z"]
        adj_t = encodings["adj_t"]
        z_log_std: torch.Tensor = self.log_std_encoder(x, adj_t)
        z_std = z_log_std.exp()
        return {"adj_t": adj_t, "z_mu": z_mu, "z_std": z_std}

    def forward(
        self, x: torch.Tensor, adj_t: SparseTensor
    ) -> Dict[str, torch.Tensor]:
        encodings = self.encode(x, adj_t)
        z_mu = encodings["z_mu"]
        z_std = encodings["z_std"]
        adj_t = encodings["adj_t"]
        if self.training:
            q = Normal(z_mu, z_std)
            z = q.rsample()
        else:
            z = z_mu
        A = self.decode(z, adj_t)
        return {"z": z, "z_mu": z_mu, "z_std": z_std, "A": A}

    def _accumulate_step(
        self, x: torch.Tensor, adj_t: SparseTensor,
    ) -> Dict[str, torch.Tensor]:
        results: Dict[str, torch.Tensor] = self(x, adj_t)

        A = results["A"]
        rc_loss = -A.log().mean()

        z_mu, z_std = results["z_mu"], results["z_std"]
        kl_loss = kl_divergence_loss(
            z_mu, z_std ** 2, torch.zeros_like(z_mu), torch.ones_like(z_std),
        )
        total_loss = rc_loss + kl_loss

        losses = dict()
        losses["rc_loss"] = rc_loss
        losses["kl_loss"] = kl_loss
        losses["total_loss"] = total_loss
        return losses


class GFCNN(ModelBase):
    def __init__(
        self,
        input_size: int = 16,
        num_nodes: int = 116,
        clf_hidden_size: Sequence[int] = (256, 256, 128),
        clf_output_size: int = 2,
        **kwargs
    ):
        super().__init__()
        dimensions = (
            [input_size * num_nodes] + list(clf_hidden_size) + [clf_output_size]
        )
        layers = list()
        for i in range(len(dimensions) - 2):
            layers.append(Linear(dimensions[i], dimensions[i + 1]))
            layers.append(ReLU())
        else:
            layers.append(Linear(dimensions[-2], dimensions[-1]))
            layers.append(Softmax(dim=1))
        self.clf = Sequential(*layers)

    @staticmethod
    def state_dict_mapping() -> dict:
        return dict()

    def ss_forward(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = torch.flatten(z).unsqueeze(0)
        elif z.ndim > 2:
            z = torch.flatten(z, start_dim=1)
        else:
            raise ValueError("invalid z with shape {}".format(z.size()))
        y = self.clf(F.relu(z))
        return y

    def get_optimizer(self, param: dict) -> Optimizer:
        optim = Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=param.get("lr", 0.0001),
            weight_decay=param.get("l2_reg", 0.0001),
        )
        return optim

    def train_step(
        self,
        device: torch.device,
        labeled_data: Data,
        unlabeled_data: Optional[Data],
        optimizer: Optimizer,
        hyperparameters: Dict[str, Any],
    ) -> Dict[str, float]:
        self.to(device)
        self.train()
        with torch.enable_grad():
            optimizer.zero_grad()

            z: torch.Tensor = labeled_data.z
            real_y: torch.Tensor = labeled_data.y
            z, real_y = z.to(device), real_y.to(device)
            z = z.detach()

            pred_y = self(z)
            ce_loss = F.cross_entropy(pred_y, real_y)
            ce_loss.backward()
            optimizer.step()

            accuracy = CM.accuracy(real_y, pred_y)
            sensitivity = CM.tpr(real_y, pred_y)
            specificity = CM.tnr(real_y, pred_y)
            precision = CM.ppv(real_y, pred_y)
            f1_score = CM.f1_score(real_y, pred_y)
            metrics = {
                "ce_loss": ce_loss.item(),
                "accuracy": accuracy.item(),
                "sensitivity": sensitivity.item(),
                "specificity": specificity.item(),
                "f1": f1_score.item(),
                "precision": precision.item(),
            }
        return metrics

    def test_step(
        self, device: torch.device, test_data: Data
    ) -> Dict[str, float]:
        self.to(device)
        self.eval()

        with torch.no_grad():
            z: torch.Tensor = test_data.z
            real_y: torch.Tensor = test_data.y
            z, real_y = z.to(device), real_y.to(device)

            pred_y = self(z)
            ce_loss = F.cross_entropy(pred_y, real_y)

            accuracy = CM.accuracy(real_y, pred_y)
            sensitivity = CM.tpr(real_y, pred_y)
            specificity = CM.tnr(real_y, pred_y)
            precision = CM.ppv(real_y, pred_y)
            f1_score = CM.f1_score(real_y, pred_y)

            metrics = {
                "ce_loss": ce_loss.item(),
                "accuracy": accuracy.item(),
                "sensitivity": sensitivity.item(),
                "specificity": specificity.item(),
                "f1": f1_score.item(),
                "precision": precision.item(),
            }
        return metrics


class GAE_FCNN(Module):
    def __init__(self, gae: GAE, fcnn: GFCNN):
        super().__init__()
        self.gae = gae
        self.fcnn = fcnn

    def forward(
        self, x: torch.Tensor, adj_t: SparseTensor
    ) -> Dict[str, torch.Tensor]:
        gae_res: dict = self.gae(x, adj_t)
        z = gae_res["z"]
        y = self.fcnn(z)
        return {**gae_res, "y": y}


class GCN_FCNN(GraphModelBase):
    def __init__(self, gcn: GCN, fcnn: GFCNN):
        super().__init__()
        self.gcn = gcn
        self.fcnn = fcnn

    def forward(
        self, x: torch.Tensor, adj_t: SparseTensor
    ) -> Dict[str, torch.Tensor]:
        gcn_res: dict = self.gcn(x, adj_t)
        z = gcn_res["z"]
        y = self.fcnn(z)
        return {**gcn_res, "y": y}

    def get_optimizer(self, param: dict) -> Dict[str, Optimizer]:
        return {
            "gcn_optim": self.gcn.get_optimizer(param.get("gcn", dict())),
            "fcnn_optim": self.fcnn.get_optimizer(param.get("fcnn", dict())),
        }

    def train_step(
        self,
        device: torch.device,
        labeled_data: Sequence[Data],
        unlabeled_data: Optional[Sequence[Data]],
        optimizer: Dict[str, Optimizer],
        hyperparameters: Dict[str, Any],
    ) -> Dict[str, float]:

        self.to(device)
        self.train()
        gcn_optimizer, fcnn_optimzier = (
            optimizer["gcn_optim"],
            optimizer["fcnn_optim"],
        )

        with torch.enable_grad():
            gcn_optimizer.zero_grad()
            fcnn_optimzier.zero_grad()

            z_ls, y_ls = list(), list()
            for data in labeled_data:
                x: torch.Tensor = data.x
                y: torch.Tensor = data.y
                adj_t: SparseTensor = data.adj_t
                x, y, adj_t = x.to(device), y.to(device), adj_t.to(device)
                z = self(x, adj_t)["z"]
                z_ls.append(z)
                y_ls.append(y)
            z = torch.stack(z_ls, dim=0)
            real_y = torch.cat(y_ls, dim=0)

            pred_y = self.fcnn(z)
            ce_loss = F.cross_entropy(pred_y, real_y)
            ce_loss.backward()
            fcnn_optimzier.step()
            gcn_optimizer.step()

            accuracy = CM.accuracy(real_y, pred_y)
            sensitivity = CM.tpr(real_y, pred_y)
            specificity = CM.tnr(real_y, pred_y)
            precision = CM.ppv(real_y, pred_y)
            f1_score = CM.f1_score(real_y, pred_y)
            metrics = {
                "ce_loss": ce_loss.item(),
                "accuracy": accuracy.item(),
                "sensitivity": sensitivity.item(),
                "specificity": specificity.item(),
                "f1": f1_score.item(),
                "precision": precision.item(),
            }
        return metrics

    def test_step(
        self, device: torch.device, test_data: Sequence[Data]
    ) -> Dict[str, float]:
        self.to(device)
        self.eval()

        with torch.no_grad():
            z_ls, y_ls = list(), list()
            for data in test_data:
                x: torch.Tensor = data.x
                y: torch.Tensor = data.y
                adj_t: SparseTensor = data.adj_t
                x, y, adj_t = x.to(device), y.to(device), adj_t.to(device)
                z = self(x, adj_t)["z"]
                z_ls.append(z)
                y_ls.append(y)
            z = torch.stack(z_ls, dim=0)
            real_y = torch.cat(y_ls, dim=0)

            pred_y = self.fcnn(z)
            ce_loss = F.cross_entropy(pred_y, real_y)

            accuracy = CM.accuracy(real_y, pred_y)
            sensitivity = CM.tpr(real_y, pred_y)
            specificity = CM.tnr(real_y, pred_y)
            precision = CM.ppv(real_y, pred_y)
            f1_score = CM.f1_score(real_y, pred_y)
            metrics = {
                "ce_loss": ce_loss.item(),
                "accuracy": accuracy.item(),
                "sensitivity": sensitivity.item(),
                "specificity": specificity.item(),
                "f1": f1_score.item(),
                "precision": precision.item(),
            }
        return metrics
