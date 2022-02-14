import os
import sys
import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict
from torch.optim import Optimizer
from torch_geometric.data import Data

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.metrics import ClassificationMetrics as CM
from models.ASDDiagNet import ASDDiagNet


class ASDSAENet(ASDDiagNet):
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

        labeled_x: torch.Tensor = labeled_data.x
        real_y: torch.Tensor = labeled_data.y
        labeled_x, real_y = (
            labeled_x.to(device),
            real_y.to(device),
        )

        if unlabeled_data is not None:
            unlabeled_x: torch.Tensor = unlabeled_data.x
            unlabeled_x = unlabeled_x.to(device)

        with torch.enable_grad():
            optimizer.zero_grad()

            labeled_res = self(labeled_x)
            labeled_z = labeled_res["z"]
            labeled_x_hat = labeled_res["x_hat"]
            pred_y = labeled_res["y"]
            if unlabeled_data is not None:
                unlabeled_res = self(unlabeled_res)
                unlabeled_z = unlabeled_res["z"]
                unlabeled_x_hat = unlabeled_res["x_hat"]
                z = torch.cat((labeled_z, unlabeled_z), dim=0)
                x = torch.cat((labeled_x, unlabeled_x), dim=0)
                x_hat = torch.cat((labeled_x_hat, unlabeled_x_hat), dim=0)
            else:
                z = labeled_z
                x = labeled_x
                x_hat = labeled_x_hat

            ce_loss = F.cross_entropy(pred_y, real_y)
            rc_loss = F.mse_loss(x_hat, x, reduction="none")
            rc_loss = rc_loss.sum(dim=1).mean()

            p = hyperparameters.get("p", 0.05)
            eps = hyperparameters.get("eps", 1e-4)
            p_hat = torch.maximum((z ** 2).mean(), torch.tensor(eps))
            sparsity = torch.sum(
                p * torch.log(p / p_hat)
                + (1 - p) * torch.log((1 - p) / (1 - p_hat))
            )

            gamma = hyperparameters.get("rc_loss", 1)
            beta = hyperparameters.get("beta", 2)
            total_loss = ce_loss + gamma * rc_loss + beta * sparsity
            total_loss.backward()
            optimizer.step()

        accuracy = CM.accuracy(real_y, pred_y)
        sensitivity = CM.tpr(real_y, pred_y)
        specificity = CM.tnr(real_y, pred_y)
        precision = CM.ppv(real_y, pred_y)
        f1_score = CM.f1_score(real_y, pred_y)
        metrics = {
            "ce_loss": ce_loss.item(),
            "rc_loss": rc_loss.item(),
            "sparsity": sparsity.item(),
            "accuracy": accuracy.item(),
            "sensitivity": sensitivity.item(),
            "specificity": specificity.item(),
            "f1": f1_score.item(),
            "precision": precision.item(),
        }
        return metrics
