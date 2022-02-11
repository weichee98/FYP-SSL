import os
import sys
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, OrderedDict, Tuple
from torch.nn import Module, Linear, Parameter, BatchNorm1d
from torch.optim import Optimizer
from torch_geometric.data import Data

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.loss import kl_divergence_loss
from utils.metrics import ClassificationMetrics as CM
from models.base import ModelBase
from models.VAE_FFN import VAE_FFN


def init_zero(linear_layer: Linear):
    linear_layer.weight.data.fill_(0.0)
    linear_layer.bias.data.fill_(0.0)


class CH(Module):
    def __init__(self, input_size: int, num_sites: int):
        super().__init__()
        self.alpha = Parameter(torch.zeros(input_size))
        self.age_norm = BatchNorm1d(1)
        self.age = Linear(1, input_size)
        self.gender = Linear(2, input_size)
        self.gamma = Linear(num_sites, input_size)
        self.delta = Linear(num_sites, input_size)
        init_zero(self.age)
        init_zero(self.gender)
        init_zero(self.gamma)
        init_zero(self.delta)

    def forward(
        self,
        x: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        site: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        age_x = self.age(self.age_norm(age))
        gender_x = self.gender(gender)
        gamma = self.gamma(site)
        delta = torch.exp(self.delta(site))
        eps = (x - self.alpha - age_x - gender_x - gamma) / delta
        x_ch = self.alpha + age_x + gender_x + eps
        return {
            "x_ch": x_ch,
            "alpha": self.alpha,
            "age": age_x,
            "gender": gender_x,
            "eps": eps,
            "gamma": gamma,
            "delta": delta,
        }

    def inverse(
        self,
        x_ch: torch.Tensor,
        age_x: torch.Tensor,
        gender_x: torch.Tensor,
        gamma: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        eps = x_ch - self.alpha - age_x - gender_x
        return self.inverse_eps(eps, age_x, gender_x, gamma, delta)

    def inverse_eps(
        self,
        eps: torch.Tensor,
        age_x: torch.Tensor,
        gender_x: torch.Tensor,
        gamma,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        x = self.alpha + age_x + gender_x + gamma + delta * eps
        return x


class VAECH(ModelBase):
    def __init__(
        self,
        num_sites: int,
        input_size: int,
        hidden_size: int,
        emb_size: int,
        clf_hidden_1: int,
        clf_hidden_2: int,
        clf_output_size: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.ch = CH(input_size, num_sites)
        self.vae_ffn = VAE_FFN(
            input_size,
            hidden_size,
            emb_size,
            clf_hidden_1,
            clf_hidden_2,
            clf_output_size,
            dropout=dropout,
        )

    @staticmethod
    def state_dict_mapping() -> dict:
        return {
            "ch.batch_add.weight": "ch.gamma.weight",
            "ch.batch_add.bias": "ch.gamma.bias",
            "ch.batch_mul.weight": "ch.delta.weight",
            "ch.batch_mul.bias": "ch.delta.bias",
            "log_std": "vae_ffn.decoder.log_std",
            "encoder1.weight": "vae_ffn.encoder.hidden.0.0.weight",
            "encoder1.bias": "vae_ffn.encoder.hidden.0.0.bias",
            "encoder_mu.weight": "vae_ffn.encoder.mu.weight",
            "encoder_mu.bias": "vae_ffn.encoder.mu.bias",
            "encoder_std.weight": "vae_ffn.encoder.log_std.weight",
            "encoder_std.bias": "vae_ffn.encoder.log_std.bias",
            "decoder1.weight": "vae_ffn.decoder.decoder.0.0.weight",
            "decoder1.bias": "vae_ffn.decoder.decoder.0.0.bias",
            "decoder2.weight": "vae_ffn.decoder.decoder.1.weight",
            "decoder2.bias": "vae_ffn.decoder.decoder.1.bias",
            "cls1.weight": "vae_ffn.classifier.0.0.weight",
            "cls1.bias": "vae_ffn.classifier.0.0.bias",
            "cls2.weight": "vae_ffn.classifier.1.0.weight",
            "cls2.bias": "vae_ffn.classifier.1.0.bias",
            "cls3.weight": "vae_ffn.classifier.2.weight",
            "cls3.bias": "vae_ffn.classifier.2.bias",
            "vae.log_std": "vae_ffn.decoder.log_std",
            "vae.encoder1.weight": "vae_ffn.encoder.hidden.0.0.weight",
            "vae.encoder1.bias": "vae_ffn.encoder.hidden.0.0.bias",
            "vae.encoder_mu.weight": "vae_ffn.encoder.mu.weight",
            "vae.encoder_mu.bias": "vae_ffn.encoder.mu.bias",
            "vae.encoder_std.weight": "vae_ffn.encoder.log_std.weight",
            "vae.encoder_std.bias": "vae_ffn.encoder.log_std.bias",
            "vae.decoder1.weight": "vae_ffn.decoder.decoder.0.0.weight",
            "vae.decoder1.bias": "vae_ffn.decoder.decoder.0.0.bias",
            "vae.decoder2.weight": "vae_ffn.decoder.decoder.1.weight",
            "vae.decoder2.bias": "vae_ffn.decoder.decoder.1.bias",
            "vae.cls1.weight": "vae_ffn.classifier.0.0.weight",
            "vae.cls1.bias": "vae_ffn.classifier.0.0.bias",
            "vae.cls2.weight": "vae_ffn.classifier.1.0.weight",
            "vae.cls2.bias": "vae_ffn.classifier.1.0.bias",
            "vae.cls3.weight": "vae_ffn.classifier.2.weight",
            "vae.cls3.bias": "vae_ffn.classifier.2.bias",
        }

    @classmethod
    def update_old_parameters(
        cls,
        old_state_dict: OrderedDict[str, torch.Tensor],
        model_params: Dict[str, Any],
    ) -> OrderedDict[str, torch.Tensor]:
        log_std: torch.Tensor = old_state_dict["vae_ffn.decoder.log_std"]
        old_state_dict["vae_ffn.decoder.log_std"] = log_std.expand(
            1, int(model_params.get("input_size", 1))
        )
        return old_state_dict

    def combat(
        self,
        x: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        site: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.ch(x, age, gender, site)

    def forward(
        self,
        x: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        site: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ch_res = self.ch(x, age, gender, site)
        vae_res = self.vae_ffn(ch_res["eps"])

        eps_mu = vae_res["x_mu"]
        vae_res["eps_mu"] = eps_mu
        vae_res["x_mu"] = self.ch.inverse_eps(
            eps_mu,
            ch_res["age"],
            ch_res["gender"],
            ch_res["gamma"],
            ch_res["delta"],
        )
        return {**ch_res, **vae_res}

    def ss_forward(self, eps: torch.Tensor) -> torch.Tensor:
        y = self.vae_ffn.ss_forward(eps)
        return y

    def get_baselines_inputs(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = data.x, data.y
        age, gender, site = data.age, data.gender, data.d
        ch_res = self.combat(x, age, gender, site)
        eps: torch.Tensor = ch_res["eps"]
        baselines = eps[y == 0].mean(dim=0).view(1, -1)
        inputs = eps[y == 1]
        return baselines, inputs

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
        labeled_age: torch.Tensor = labeled_data.age
        labeled_gender: torch.Tensor = labeled_data.gender
        labeled_site: torch.Tensor = labeled_data.d
        real_y: torch.Tensor = labeled_data.y
        labeled_x, labeled_age, labeled_gender, labeled_site, real_y = (
            labeled_x.to(device),
            labeled_age.to(device),
            labeled_gender.to(device),
            labeled_site.to(device),
            real_y.to(device),
        )

        if unlabeled_data is not None:
            unlabeled_x: torch.Tensor = unlabeled_data.x
            unlabeled_age: torch.Tensor = unlabeled_data.age
            unlabeled_gender: torch.Tensor = unlabeled_data.gender
            unlabeled_site: torch.Tensor = unlabeled_data.d
            unlabeled_x, unlabeled_age, unlabeled_gender, unlabeled_site = (
                unlabeled_x.to(device),
                unlabeled_age.to(device),
                unlabeled_gender.to(device),
                unlabeled_site.to(device),
            )

        with torch.enable_grad():
            optimizer.zero_grad()

            labeled_res = self(
                labeled_x, labeled_age, labeled_gender, labeled_site
            )
            pred_y = labeled_res["y"]
            labeled_x_mu = labeled_res["x_mu"]
            labeled_x_std = labeled_res["x_std"]
            labeled_z_mu = labeled_res["z_mu"]
            labeled_z_std = labeled_res["z_std"]
            labeled_eps = labeled_res["eps"]
            if unlabeled_data is not None:
                unlabeled_res = self(
                    unlabeled_x, unlabeled_age, unlabeled_gender, unlabeled_site
                )
                unlabeled_x_mu = unlabeled_res["x_mu"]
                unlabeled_x_std = unlabeled_res["x_std"]
                unlabeled_z_mu = unlabeled_res["z_mu"]
                unlabeled_z_std = unlabeled_res["z_std"]
                unlabeled_eps = unlabeled_res["eps"]
                x = torch.cat((labeled_x, unlabeled_x), dim=0)
                x_mu = torch.cat((labeled_x_mu, unlabeled_x_mu), dim=0)
                x_std = torch.cat((labeled_x_std, unlabeled_x_std), dim=0)
                z_mu = torch.cat((labeled_z_mu, unlabeled_z_mu), dim=0)
                z_std = torch.cat((labeled_z_std, unlabeled_z_std), dim=0)
                eps = torch.cat((labeled_eps, unlabeled_eps), dim=0)
            else:
                x = labeled_x
                x_mu = labeled_x_mu
                x_std = labeled_x_std
                z_mu = labeled_z_mu
                z_std = labeled_z_std
                eps = labeled_eps

            ce_loss = F.cross_entropy(pred_y, real_y)
            rc_loss = F.gaussian_nll_loss(x_mu, x, x_std, full=True)
            kl_loss = kl_divergence_loss(
                z_mu,
                z_std ** 2,
                torch.zeros_like(z_mu),
                torch.ones_like(z_std),
            )
            ch_loss = (eps ** 2).mean()

            gamma1 = hyperparameters.get("rc_loss", 1)
            gamma2 = hyperparameters.get("kl_loss", 1)
            gamma3 = hyperparameters.get("ch_loss", 1)
            total_loss = (
                ce_loss + gamma1 * rc_loss + gamma2 * kl_loss + gamma3 * ch_loss
            )
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
            "kl_loss": kl_loss.item(),
            "ch_loss": ch_loss.item(),
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
            age: torch.Tensor = test_data.age
            gender: torch.Tensor = test_data.gender
            site: torch.Tensor = test_data.d
            real_y: torch.Tensor = test_data.y
            x, age, gender, site, real_y = (
                x.to(device),
                age.to(device),
                gender.to(device),
                site.to(device),
                real_y.to(device),
            )

            res = self(x, age, gender, site)
            pred_y = res["y"]
            x_mu = res["x_mu"]
            x_std = res["x_std"]
            z_mu = res["z_mu"]
            z_std = res["z_std"]
            eps: torch.Tensor = res["eps"]

            ce_loss = F.cross_entropy(pred_y, real_y)
            rc_loss = F.gaussian_nll_loss(x_mu, x, x_std, full=True)
            kl_loss = kl_divergence_loss(
                z_mu,
                z_std ** 2,
                torch.zeros_like(z_mu),
                torch.ones_like(z_std),
            )
            ch_loss = (eps ** 2).mean()

        accuracy = CM.accuracy(real_y, pred_y)
        sensitivity = CM.tpr(real_y, pred_y)
        specificity = CM.tnr(real_y, pred_y)
        precision = CM.ppv(real_y, pred_y)
        f1_score = CM.f1_score(real_y, pred_y)
        metrics = {
            "ce_loss": ce_loss.item(),
            "rc_loss": rc_loss.item(),
            "kl_loss": kl_loss.item(),
            "ch_loss": ch_loss.item(),
            "accuracy": accuracy.item(),
            "sensitivity": sensitivity.item(),
            "specificity": specificity.item(),
            "f1": f1_score.item(),
            "precision": precision.item(),
        }
        return metrics


if __name__ == "__main__":
    model = VAECH.load_from_state_dict(
        "/data/yeww0006/FYP-SSL/.archive/exp20_ABIDE_WHOLE/ssl_ABIDE_1641361495/models/1641372616.pt",
        dict(
            num_sites=20,
            input_size=34716,
            hidden_size=300,
            emb_size=150,
            clf_hidden_1=50,
            clf_hidden_2=30,
        ),
    )

    print(model)

    x = torch.randn((10, 34716))
    age = torch.randn((10, 1))
    gender = torch.randn((10, 2))
    site = torch.randn((10, 20))
    res = model(x, age, gender, site)
    for k, v in res.items():
        print("{}: {}".format(k, v.size()))
