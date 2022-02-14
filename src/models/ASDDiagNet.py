import os
import sys
import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict
from torch.nn import Tanh, Softmax
from torch.optim import Optimizer, Adam
from torch_geometric.data import Data

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.metrics import ClassificationMetrics as CM
from models.base import ModelBase, FeedForward


class ASDDiagNet(ModelBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        emb_size: int,
        clf_hidden_1: int,
        clf_hidden_2: int,
        clf_output_size: int = 2,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.encoder = FeedForward(
            input_size,
            [hidden_size] if hidden_size > 0 else [],
            emb_size,
            dropout=dropout,
        )
        self.decoder = FeedForward(
            emb_size,
            [hidden_size] if hidden_size > 0 else [],
            input_size,
            Tanh(),
            dropout=dropout,
        )
        self.classifier = FeedForward(
            emb_size,
            [h for h in [clf_hidden_1, clf_hidden_2] if h > 0],
            clf_output_size,
            Softmax(dim=1),
            dropout=dropout,
        )

    @staticmethod
    def state_dict_mapping() -> dict:
        return {
            "encoder1.weight": "encoder.0.0.weight",
            "encoder1.bias": "encoder.0.0.bias",
            "encoder2.weight": "encoder.1.weight",
            "encoder2.bias": "encoder.1.bias",
            "decoder1.weight": "decoder.0.0.weight",
            "decoder1.bias": "decoder.0.0.bias",
            "decoder2.weight": "decoder.1.weight",
            "decoder2.bias": "decoder.1.bias",
            "cls1.weight": "classifier.0.0.weight",
            "cls1.bias": "classifier.0.0.bias",
            "cls2.weight": "classifier.1.0.weight",
            "cls2.bias": "classifier.1.0.bias",
            "cls3.weight": "classifier.2.weight",
            "cls3.bias": "classifier.2.bias",
        }

    def get_optimizer(self, param: dict) -> Optimizer:
        optim = Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=param.get("lr", 0.0001),
            weight_decay=param.get("l2_reg", 0.0001),
        )
        return optim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(F.relu(z))

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(F.relu(z))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        returns:
            - classifier output
            - encoder output
            - decoder output
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        y = self.classify(z)
        return {"y": y, "x_hat": x_hat, "z": z}

    def ss_forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        y = self.classify(z)
        return y

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
            labeled_x_hat = labeled_res["x_hat"]
            pred_y = labeled_res["y"]
            if unlabeled_data is not None:
                unlabeled_res = self(unlabeled_res)
                unlabeled_x_hat = unlabeled_res["x_hat"]
                x = torch.cat((labeled_x, unlabeled_x), dim=0)
                x_hat = torch.cat((labeled_x_hat, unlabeled_x_hat), dim=0)
            else:
                x = labeled_x
                x_hat = labeled_x_hat

            ce_loss = F.cross_entropy(pred_y, real_y)
            rc_loss = F.mse_loss(x_hat, x, reduction="none")
            rc_loss = rc_loss.sum(dim=1).mean()

            gamma = hyperparameters.get("rc_loss", 1)
            total_loss = ce_loss + gamma * rc_loss
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
            x: torch.Tensor = test_data.x
            real_y: torch.Tensor = test_data.y
            x, real_y = x.to(device), real_y.to(device)

            result = self(x)
            pred_y = result["y"]
            x_hat = result["x_hat"]

            rc_loss = F.mse_loss(x_hat, x, reduction="none")
            rc_loss = rc_loss.sum(dim=1).mean()
            ce_loss = F.cross_entropy(pred_y, real_y)

            accuracy = CM.accuracy(real_y, pred_y)
            sensitivity = CM.tpr(real_y, pred_y)
            specificity = CM.tnr(real_y, pred_y)
            precision = CM.ppv(real_y, pred_y)
            f1_score = CM.f1_score(real_y, pred_y)

            metrics = {
                "ce_loss": ce_loss.item(),
                "rc_loss": rc_loss.item(),
                "accuracy": accuracy.item(),
                "sensitivity": sensitivity.item(),
                "specificity": specificity.item(),
                "f1": f1_score.item(),
                "precision": precision.item(),
            }
        return metrics


if __name__ == "__main__":
    model = ASDDiagNet.load_from_state_dict(
        "/data/yeww0006/FYP-SSL/.archive/exp20_ABIDE_WHOLE/ssl_ABIDE_1639615017/models/1639616389.pt",
        dict(
            input_size=34716,
            hidden_size=300,
            emb_size=150,
            clf_hidden_1=50,
            clf_hidden_2=30,
        ),
    )
    print(model)

    x = torch.randn((10, 34716))
    res = model(x)
    for k, v in res.items():
        print("{}: {}".format(k, v.size()))

