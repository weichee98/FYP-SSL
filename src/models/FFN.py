import os
import sys
import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict
from torch.nn import Softmax
from torch.optim import Optimizer
from torch_geometric.data import Data

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.metrics import ClassificationMetrics as CM
from models.base import ModelBase, FeedForward


class FFN(ModelBase):
    def __init__(
        self,
        input_size: int,
        hidden_1: int,
        hidden_2: int,
        hidden_3: int,
        output_size: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.classifier = FeedForward(
            input_size,
            [h for h in [hidden_1, hidden_2, hidden_3] if h > 0],
            output_size,
            Softmax(),
            dropout=dropout,
        )

    @staticmethod
    def state_dict_mapping() -> dict:
        return {
            "linear1.weight": "classifier.0.0.weight",
            "linear1.bias": "classifier.0.0.bias",
            "linear2.weight": "classifier.1.0.weight",
            "linear2.bias": "classifier.1.0.bias",
            "linear3.weight": "classifier.2.0.weight",
            "linear3.bias": "classifier.2.0.bias",
            "linear4.weight": "classifier.3.weight",
            "linear4.bias": "classifier.3.bias",
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = self.classifier(x)
        return {"y": y}

    def ss_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["y"]

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
        real_y: torch.Tensor = labeled_data.y
        x, real_y = (
            x.to(device),
            real_y.to(device),
        )

        with torch.enable_grad():
            optimizer.zero_grad()
            pred_y = self.ss_forward(x)
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
            x: torch.Tensor = test_data.x
            real_y: torch.Tensor = test_data.y
            x, real_y = x.to(device), real_y.to(device)

            pred_y = self.ss_forward(x)
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


if __name__ == "__main__":
    model = FFN.load_from_state_dict(
        "/data/yeww0006/FYP-SSL/.archive/exp20_ABIDE_WHOLE/ssl_ABIDE_1639612566/models/1639612623.pt",
        dict(input_size=34716, hidden_1=150, hidden_2=50, hidden_3=30,),
    )
    print(model)

    x = torch.randn((10, 34716))
    res = model(x)
    for k, v in res.items():
        print("{}: {}".format(k, v.size()))
