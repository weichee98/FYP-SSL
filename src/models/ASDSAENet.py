import os
import sys
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, Tanh, Parameter, Linear, ReLU, Softmax
from torch.optim import Adam, Optimizer
from torch_geometric.data import Data

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from models.base import FeedForward, ModelBase
from utils.metrics import ClassificationMetrics as CM


class SAE(ModelBase):
    def __init__(self, input_size: int = 9500, emb_size: int = 4975):
        super().__init__()
        self.encoder = FeedForward(input_size, [], emb_size)
        self.decoder = FeedForward(emb_size, [], input_size, Tanh())

    @staticmethod
    def state_dict_mapping() -> dict:
        return dict()

    def ss_forward(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(F.relu(z))
        return {"z": z, "x_hat": x_hat}

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

            x: torch.Tensor = labeled_data.x
            x = x.to(device)

            result = self(x)
            pred_x = result["x_hat"]
            z: torch.Tensor = result["z"]

            beta = hyperparameters.get("beta", 2)
            p = hyperparameters.get("p", 0.05)
            eps = hyperparameters.get("eps", 1e-4)

            rc_loss = F.mse_loss(x, pred_x, reduction="sum")
            rc_loss = rc_loss.sum(dim=1).mean()

            p_hat = torch.maximum((z ** 2).mean(), eps)
            sparsity = torch.sum(
                p * torch.log(p / p_hat)
                + (1 - p) * torch.log((1 - p) / (1 - p_hat))
            )
            total_loss = rc_loss + beta * sparsity
            total_loss.backward()
            optimizer.step()

        return {"rc_loss": rc_loss.item(), "sparsity": sparsity.item()}

    def test_step(
        self, device: torch.device, test_data: Data
    ) -> Dict[str, float]:
        self.to(device)
        self.eval()

        with torch.enable_grad():
            x: torch.Tensor = test_data.x
            x = x.to(device)

            result = self(x)
            pred_x = result["x_hat"]

            rc_loss = F.mse_loss(x, pred_x, reduction="sum")
            rc_loss = rc_loss.sum(dim=1).mean()

        return {"rc_loss": rc_loss.item()}


class MaskedSAE(SAE):
    def __init__(
        self,
        input_size: int = 19000,
        emb_size: int = 4975,
        mask_ratio: float = 0.5,
        mask_fitted: bool = False,
    ):
        self.num_features = int(round(input_size * mask_ratio))
        super().__init__(self.num_features, emb_size)
        self.mask_fitted = mask_fitted
        self.mask = Parameter(
            torch.zeros(input_size, dtype=torch.bool), requires_grad=False
        )

    def fit_mask(self, x: torch.Tensor):
        n_smallest = int(self.num_features / 2)
        n_largest = self.num_features - n_smallest
        mask = torch.zeros(x.size(0), dtype=torch.bool)

        mean = x.mean(dim=0)
        n_largest_idx = torch.topk(mean, n_largest)[1]
        n_smallest_idx = torch.topk(mean, n_smallest, largest=False)[1]
        mask[n_largest_idx] = True
        mask[n_smallest_idx] = True

        self.mask = Parameter(mask, requires_grad=False)
        self.mask_fitted = True
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.mask_fitted:
            raise Exception("call train_mask first")
        x = x[:, self.mask]
        res = super().forward(x)
        res["masked_x"] = x
        return res

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
        x: torch.Tensor = labeled_data.x
        x = x.to(device)

        if not self.mask_fitted:
            self.fit_mask(x)

        with torch.enable_grad():
            optimizer.zero_grad()

            result = self(x)
            pred_x = result["x_hat"]
            masked_x = result["masked_x"]
            z: torch.Tensor = result["z"]

            beta = hyperparameters.get("beta", 2)
            p = hyperparameters.get("p", 0.05)
            eps = hyperparameters.get("eps", 1e-4)

            rc_loss = F.mse_loss(masked_x, pred_x, reduction="sum")
            rc_loss = rc_loss.sum(dim=1).mean()

            p_hat = torch.maximum((z ** 2).mean(), eps)
            sparsity = torch.sum(
                p * torch.log(p / p_hat)
                + (1 - p) * torch.log((1 - p) / (1 - p_hat))
            )
            total_loss = rc_loss + beta * sparsity
            total_loss.backward()
            optimizer.step()

        return {"rc_loss": rc_loss.item(), "sparsity": sparsity.item()}

    def test_step(
        self, device: torch.device, test_data: Data
    ) -> Dict[str, float]:
        self.to(device)
        self.eval()

        with torch.enable_grad():
            x: torch.Tensor = test_data.x
            x = x.to(device)

            result = self(x)
            pred_x = result["x_hat"]
            masked_x = result["masked_x"]

            rc_loss = F.mse_loss(masked_x, pred_x, reduction="sum")
            rc_loss = rc_loss.sum(dim=1).mean()

        return {"rc_loss": rc_loss.item()}


class FCNN(ModelBase):
    def __init__(self, input_size: int = 4975, hidden_1: int = 2487, hidden_2: int = 500, output_size: int = 2):
        super().__init__()
        self.clf = Sequential(
            Linear(input_size, hidden_1),
            ReLU(),
            Linear(hidden_1, hidden_2),
            ReLU(),
            Linear(hidden_2, output_size),
            Softmax()
        )

    @staticmethod
    def state_dict_mapping() -> dict:
        return dict()

    def ss_forward(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.clf(z)
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


class ASDSAENet(Module):
    def __init__(self, sae: SAE, cls: FCNN):
        super().__init__()
        self.sae = sae
        self.cls = cls

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae(x)["z"]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.sae.decode(z)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.cls(F.relu(z))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        ae_res: dict = self.sae(x)
        y = self.cls(F.relu(ae_res["z"]))
        return {**ae_res, "y": y}
