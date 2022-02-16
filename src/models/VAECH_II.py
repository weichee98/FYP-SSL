import os
import sys
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from torch.optim import Optimizer, Adam
from torch_geometric.data import Data

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.loss import kl_divergence_loss
from utils.metrics import ClassificationMetrics as CM
from models.VAECH_I import VAECH_I


class VAECHOptimizer:
    def __init__(self, ch_optim: Optimizer, vae_optim: Optimizer):
        self.ch_optim = ch_optim
        self.vae_optim = vae_optim

    def zero_grad(self):
        self.ch_optim.zero_grad()
        self.vae_optim.zero_grad()

    def step(self):
        self.vae_optim.step()
        self.ch_optim.step()


class VAECH_II(VAECH_I):
    def get_optimizer(
        self, param: Dict[str, Any]
    ) -> Tuple[Optimizer, Optimizer]:
        vae_param: dict = param.get("vae_ffn", dict())
        ch_param: dict = param.get("ch", dict())
        vae_optim = Adam(
            filter(lambda p: p.requires_grad, self.vae_ffn.parameters()),
            lr=vae_param.get("lr", 0.0001),
            weight_decay=vae_param.get("l2_reg", 0.0),
        )
        ch_optim = Adam(
            filter(lambda p: p.requires_grad, self.ch.parameters()),
            lr=ch_param.get("lr", 0.005),
            weight_decay=ch_param.get("l2_reg", 0.0),
        )
        return vae_optim, ch_optim

    def train_step(
        self,
        device: torch.device,
        labeled_data: Data,
        unlabeled_data: Optional[Data],
        optimizer: Tuple[Optimizer, Optimizer],
        hyperparameters: Dict[str, Any],
    ) -> Dict[str, float]:
        self.to(device)
        self.train()
        vae_optim, ch_optim = optimizer

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
            x = torch.cat((labeled_x, unlabeled_x), dim=0)
        else:
            x = labeled_x

        with torch.enable_grad():

            """
            train CH component
            """
            ch_optim.zero_grad()

            labeled_ch_res = self.combat(
                labeled_x, labeled_age, labeled_gender, labeled_site
            )
            alpha = labeled_ch_res["alpha"]
            labeled_eps = labeled_ch_res["eps"]
            if unlabeled_data is not None:
                unlabeled_ch_res = self.combat(
                    unlabeled_x, unlabeled_age, unlabeled_gender, unlabeled_site
                )
                unlabeled_eps = unlabeled_ch_res["eps"]
                eps = torch.cat((labeled_eps, unlabeled_eps), dim=0)
            else:
                eps = labeled_eps

            ch_loss = (eps ** 2).sum(dim=1).mean()
            alpha_loss = F.mse_loss(alpha, x.mean(dim=0), reduction="sum")

            use_alpha_loss = hyperparameters.get("alpha_loss", True)
            gamma3 = hyperparameters.get("ch_loss", 1)
            if use_alpha_loss:
                total_loss = gamma3 * (ch_loss + alpha_loss)
            else:
                total_loss = gamma3 * ch_loss
            total_loss.backward()
            ch_optim.step()

            """
            train VAE component
            """
            vae_optim.zero_grad()

            labeled_ch_res = self.combat(
                labeled_x, labeled_age, labeled_gender, labeled_site
            )
            labeled_x_ch = labeled_ch_res["x_ch"]
            labeled_x_ch = labeled_x_ch.detach()
            if unlabeled_data is not None:
                unlabeled_ch_res = self.combat(
                    unlabeled_x, unlabeled_age, unlabeled_gender, unlabeled_site
                )
                unlabeled_x_ch = unlabeled_ch_res["x_ch"]
                unlabeled_x_ch = unlabeled_x_ch.detach()

            labeled_vae_res = self.vae_ffn(labeled_x_ch)
            pred_y = labeled_vae_res["y"]
            labeled_x_ch_mu = labeled_vae_res["x_mu"]
            labeled_x_std = labeled_vae_res["x_std"]
            labeled_z_mu = labeled_vae_res["z_mu"]
            labeled_z_std = labeled_vae_res["z_std"]
            if unlabeled_data is not None:
                unlabeled_vae_res = self.vae_ffn(unlabeled_x_ch)
                unlabeled_x_ch_mu = unlabeled_vae_res["x_mu"]
                unlabeled_x_std = unlabeled_vae_res["x_std"]
                unlabeled_z_mu = unlabeled_vae_res["z_mu"]
                unlabeled_z_std = unlabeled_vae_res["z_std"]
                x_ch = torch.cat((labeled_x_ch, unlabeled_x_ch), dim=0)
                x_ch_mu = torch.cat((labeled_x_ch_mu, unlabeled_x_ch_mu), dim=0)
                x_std = torch.cat((labeled_x_std, unlabeled_x_std), dim=0)
                z_mu = torch.cat((labeled_z_mu, unlabeled_z_mu), dim=0)
                z_std = torch.cat((labeled_z_std, unlabeled_z_std), dim=0)
            else:
                x_ch = labeled_x_ch
                x_ch_mu = labeled_x_ch_mu
                x_std = labeled_x_std
                z_mu = labeled_z_mu
                z_std = labeled_z_std

            ce_loss = F.cross_entropy(pred_y, real_y)
            rc_loss = F.gaussian_nll_loss(x_ch_mu, x_ch, x_std, full=True)
            kl_loss = kl_divergence_loss(
                z_mu,
                z_std ** 2,
                torch.zeros_like(z_mu),
                torch.ones_like(z_std),
            )
            gamma1 = hyperparameters.get("rc_loss", 1)
            gamma2 = hyperparameters.get("kl_loss", 1)
            total_loss = ce_loss + gamma1 * rc_loss + gamma2 * kl_loss
            total_loss.backward()
            vae_optim.step()

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
            x_ch = res["x_ch"]
            x_ch_mu = res["x_ch_mu"]
            x_std = res["x_std"]
            z_mu = res["z_mu"]
            z_std = res["z_std"]
            alpha = res["alpha"]
            eps: torch.Tensor = res["eps"]

            ce_loss = F.cross_entropy(pred_y, real_y)
            rc_loss = F.gaussian_nll_loss(x_ch_mu, x_ch, x_std, full=True)
            kl_loss = kl_divergence_loss(
                z_mu,
                z_std ** 2,
                torch.zeros_like(z_mu),
                torch.ones_like(z_std),
            )
            ch_loss = (
                F.mse_loss(alpha, x.mean(dim=0), reduction="sum")
                + (eps ** 2).sum(dim=1).mean()
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
            "accuracy": accuracy.item(),
            "sensitivity": sensitivity.item(),
            "specificity": specificity.item(),
            "f1": f1_score.item(),
            "precision": precision.item(),
        }
        return metrics


if __name__ == "__main__":
    model = VAECH_II.load_from_state_dict(
        "/data/yeww0006/FYP-SSL/.archive/exp06_rerun_ffn/ssl_ABIDE_1644486310/models/1644490844.pt",
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
