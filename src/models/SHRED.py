import os
import sys
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional
from torch.nn.utils import clip_grad_norm_
from torch.nn import Module, Linear, Parameter, BatchNorm1d
from torch.optim import Optimizer
from torch_geometric.data import Data

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.loss import kl_divergence_loss
from utils.metrics import ClassificationMetrics as CM
from models.base import ModelBase
from models.EDC_VAE import EDC_VAE


def init_zero(linear_layer: Linear):
    linear_layer.weight.data.fill_(0.0)
    linear_layer.bias.data.fill_(0.0)


class CH(Module):
    def __init__(self, input_size: int, num_sites: int):
        super().__init__()
        self.num_sites = num_sites
        self.alpha = Parameter(torch.zeros(input_size))
        self.age_norm = BatchNorm1d(1)
        self.age = Linear(1, input_size)
        self.gender = Linear(1, input_size)
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
        if site.ndim == 2:
            pass
        elif site.ndim == 1:
            site = F.one_hot(site, num_classes=self.num_sites).float()
        else:
            raise ValueError("invalid shape for site: {}".format(site.size()))

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


class SHRED(ModelBase):
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
        **kwargs
    ):
        super().__init__()
        self.ch = CH(input_size, num_sites)
        self.edc_vae = EDC_VAE(
            input_size,
            hidden_size,
            emb_size,
            clf_hidden_1,
            clf_hidden_2,
            clf_output_size,
            dropout=dropout,
        )

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
        vae_res = self.edc_vae(ch_res["x_ch"])

        x_ch_mu = vae_res["x_mu"]
        vae_res["x_ch_mu"] = x_ch_mu
        vae_res["x_mu"] = self.ch.inverse(
            x_ch_mu,
            ch_res["age"],
            ch_res["gender"],
            ch_res["gamma"],
            ch_res["delta"],
        )
        return {**ch_res, **vae_res}

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
            alpha: torch.Tensor = labeled_res["alpha"]
            labeled_age_x = labeled_res["age"]
            labeled_gender_x = labeled_res["gender"]
            pred_y = labeled_res["y"]
            labeled_x_mu = labeled_res["x_mu"]
            labeled_x_std = labeled_res["x_std"]
            labeled_z_mu = labeled_res["z_mu"]
            labeled_z_std = labeled_res["z_std"]
            labeled_eps = labeled_res["eps"]
            labeled_gamma = labeled_res["gamma"]
            labeled_delta = labeled_res["delta"]
            if unlabeled_data is not None:
                unlabeled_res = self(
                    unlabeled_x, unlabeled_age, unlabeled_gender, unlabeled_site
                )
                unlabeled_age_x = unlabeled_res["age"]
                unlabeled_gender_x = unlabeled_res["gender"]
                unlabeled_x_mu = unlabeled_res["x_mu"]
                unlabeled_x_std = unlabeled_res["x_std"]
                unlabeled_z_mu = unlabeled_res["z_mu"]
                unlabeled_z_std = unlabeled_res["z_std"]
                unlabeled_eps = unlabeled_res["eps"]
                unlabeled_gamma = unlabeled_res["gamma"]
                unlabeled_delta = unlabeled_res["delta"]
                age_x = torch.cat((labeled_age_x, unlabeled_age_x), dim=0)
                gender_x = torch.cat(
                    (labeled_gender_x, unlabeled_gender_x), dim=0
                )
                x = torch.cat((labeled_x, unlabeled_x), dim=0)
                x_mu = torch.cat((labeled_x_mu, unlabeled_x_mu), dim=0)
                x_std = torch.cat((labeled_x_std, unlabeled_x_std), dim=0)
                z_mu = torch.cat((labeled_z_mu, unlabeled_z_mu), dim=0)
                z_std = torch.cat((labeled_z_std, unlabeled_z_std), dim=0)
                eps = torch.cat((labeled_eps, unlabeled_eps), dim=0)
                gamma = torch.cat((labeled_gamma, unlabeled_gamma), dim=0)
                delta = torch.cat((labeled_delta, unlabeled_delta), dim=0)
            else:
                age_x = labeled_age_x
                gender_x = labeled_gender_x
                x = labeled_x
                x_mu = labeled_x_mu
                x_std = labeled_x_std
                z_mu = labeled_z_mu
                z_std = labeled_z_std
                eps = labeled_eps
                gamma = labeled_gamma
                delta = labeled_delta

            ce_loss = F.cross_entropy(pred_y, real_y)
            rc_loss = F.gaussian_nll_loss(x_mu, x, x_std ** 2, full=True)
            kl_loss = kl_divergence_loss(
                z_mu,
                z_std ** 2,
                torch.zeros_like(z_mu),
                torch.ones_like(z_std),
            )

            stand_mean = alpha.expand(age_x.size()) + age_x + gender_x
            ch_loss = F.gaussian_nll_loss(
                gamma, x - stand_mean, delta ** 2, full=True
            )
            alpha_loss = F.gaussian_nll_loss(
                stand_mean,
                x,
                x.var(dim=0, keepdim=True).expand(x.size()),
                full=True,
            )

            gamma1 = hyperparameters.get("rc_loss", 1)
            gamma2 = hyperparameters.get("kl_loss", 1)
            gamma3 = hyperparameters.get("ch_loss", 1)
            use_alpha_loss = hyperparameters.get("alpha_loss", True)

            if use_alpha_loss:
                total_loss = (
                    ce_loss
                    + gamma1 * rc_loss
                    + gamma2 * kl_loss
                    + gamma3 * (ch_loss + alpha_loss)
                )
            else:
                total_loss = (
                    ce_loss
                    + gamma1 * rc_loss
                    + gamma2 * kl_loss
                    + gamma3 * ch_loss
                )
            total_loss.backward()
            clip_grad_norm_(self.parameters(), 5.0)
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
            "alpha_loss": alpha_loss.item(),
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
            alpha: torch.Tensor = res["alpha"]
            age_x: torch.Tensor = res["age"]
            gender_x = res["gender"]
            eps: torch.Tensor = res["eps"]
            gamma: torch.Tensor = res["gamma"]
            delta: torch.Tensor = res["delta"]

            ce_loss = F.cross_entropy(pred_y, real_y)
            rc_loss = F.gaussian_nll_loss(x_mu, x, x_std ** 2, full=True)
            kl_loss = kl_divergence_loss(
                z_mu,
                z_std ** 2,
                torch.zeros_like(z_mu),
                torch.ones_like(z_std),
            )

            stand_mean = alpha.expand(age_x.size()) + age_x + gender_x
            ch_loss = F.gaussian_nll_loss(
                gamma, x - stand_mean, delta ** 2, full=True
            )
            alpha_loss = F.gaussian_nll_loss(
                stand_mean,
                x,
                x.var(dim=0, keepdim=True).expand(x.size()),
                full=True,
            )

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
            "alpha_loss": alpha_loss.item(),
            "accuracy": accuracy.item(),
            "sensitivity": sensitivity.item(),
            "specificity": specificity.item(),
            "f1": f1_score.item(),
            "precision": precision.item(),
        }
        return metrics
