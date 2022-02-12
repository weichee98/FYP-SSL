import os
import sys
import torch
from functools import reduce
import torch.nn.functional as F
from typing import Any, Dict, Optional, OrderedDict
from torch.nn import Linear, Tanh, Softmax
from torch.distributions import Normal
from torch.optim import Optimizer, Adam
from torch_geometric.data import Data

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from utils.loss import entropy_loss, kl_divergence_loss
from utils.metrics import ClassificationMetrics as CM
from models.base import (
    FeedForward,
    ModelBase,
    VariationalDecoder,
    VariationalEncoder,
)


class VAESDR(ModelBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        emb_size: int,
        num_sites: int,
        clf_output_size: int = 2,
        dropout: float = 0.5,
        share_decoder: bool = False,
        **kwargs
    ):
        super().__init__()
        self.encoder = VariationalEncoder(
            input_size,
            [hidden_size] if hidden_size > 0 else [],
            emb_size,
            dropout,
        )
        self.site_encoder = Linear(emb_size, emb_size)
        self.disease_encoder = Linear(emb_size, emb_size)

        self.decoder = VariationalDecoder(
            emb_size,
            [hidden_size] if hidden_size > 0 else [],
            input_size,
            Tanh(),
            dropout=dropout,
        )
        if share_decoder:
            self.site_decoder = None
        else:
            self.site_decoder = FeedForward(
                emb_size,
                [hidden_size] if hidden_size > 0 else [],
                input_size,
                Tanh(),
                dropout=dropout,
            )

        self.site_cls = FeedForward(
            emb_size, [], num_sites, Softmax(), dropout=dropout
        )
        self.disease_cls = FeedForward(
            emb_size, [], clf_output_size, Softmax(), dropout=dropout
        )
        self.site_dis = FeedForward(
            emb_size, [], num_sites, Softmax(), dropout=dropout
        )
        self.disease_dis = FeedForward(
            emb_size, [], clf_output_size, Softmax(), dropout=dropout
        )

    @staticmethod
    def state_dict_mapping() -> dict:
        return {
            "log_std": "decoder.log_std",
            "encoder1.weight": "encoder.hidden.0.0.weight",
            "encoder1.bias": "encoder.hidden.0.0.bias",
            "encoder_mu.weight": "encoder.mu.weight",
            "encoder_mu.bias": "encoder.mu.bias",
            "encoder_std.weight": "encoder.log_std.weight",
            "encoder_std.bias": "encoder.log_std.bias",
            "res_encoder.weight": "disease_encoder.weight",
            "res_encoder.bias": "disease_encoder.bias",
            "decoder1.weight": "decoder.decoder.0.0.weight",
            "decoder1.bias": "decoder.decoder.0.0.bias",
            "decoder2.weight": "decoder.decoder.1.weight",
            "decoder2.bias": "decoder.decoder.1.bias",
            "site_decoder1.weight": "site_decoder.0.0.weight",
            "site_decoder1.bias": "site_decoder.0.0.bias",
            "site_decoder2.weight": "site_decoder.1.weight",
            "site_decoder2.bias": "site_decoder.1.bias",
            "site_cls.weight": "site_cls.0.weight",
            "site_cls.bias": "site_cls.0.bias",
            "disease_cls.weight": "disease_cls.0.weight",
            "disease_cls.bias": "disease_cls.0.bias",
            "site_dis.weight": "site_dis.0.weight",
            "site_dis.bias": "site_dis.0.bias",
            "disease_dis.weight": "disease_dis.0.weight",
            "disease_dis.bias": "disease_dis.0.bias",
        }

    @classmethod
    def update_old_parameters(
        cls,
        old_state_dict: OrderedDict[str, torch.Tensor],
        model_params: Dict[str, Any],
    ) -> OrderedDict[str, torch.Tensor]:
        log_std: torch.Tensor = old_state_dict["decoder.log_std"]
        old_state_dict["decoder.log_std"] = log_std.expand(
            1, int(model_params.get("input_size", 1))
        )
        return old_state_dict

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_mu, z_std = self.encoder(x)
        return {"z_mu": z_mu, "z_std": z_std}

    def split_encoding(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = F.relu(z)
        z_disease = self.disease_encoder(z)
        z_site = self.site_encoder(z)
        return {"z_disease": z_disease, "z_site": z_site}

    def decode(
        self, z_disease: torch.Tensor, z_site: torch.Tensor
    ) -> Dict[Any, torch.Tensor]:
        x_disease, x_std = self.decoder(F.relu(z_disease))
        if self.site_decoder is not None:
            x_site = self.site_decoder(F.relu(z_site))
        else:
            x_site, _ = self.decoder(F.relu(z_site))
        x_mu = x_disease + x_site
        return {
            "x_mu": x_mu,
            "x_std": x_std,
            "x_disease": x_disease,
            "x_site": x_site,
        }

    def classify_disease(self, z_disease: torch.Tensor) -> torch.Tensor:
        y = self.disease_cls(F.relu(z_disease))
        return y

    def classify_site(self, z_site: torch.Tensor) -> torch.Tensor:
        d = self.site_cls(F.relu(z_site))
        return d

    def discriminate_disease(self, z_site: torch.Tensor) -> torch.Tensor:
        y = self.disease_dis(F.relu(z_site))
        return y

    def discriminate_site(self, z_disease: torch.Tensor) -> torch.Tensor:
        d = self.site_dis(F.relu(z_disease))
        return d

    def get_all_encodings(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        all_encodings = dict()

        encode_res = self.encode(x)
        all_encodings.update(encode_res)
        z_mu, z_std = encode_res["z_mu"], encode_res["z_std"]
        if self.training:
            q = Normal(z_mu, z_std)
            z = q.rsample()
        else:
            z = z_mu
        all_encodings["z"] = z

        split_encode_res = self.split_encoding(z)
        all_encodings.update(split_encode_res)
        return all_encodings

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        final_result = dict()

        all_encodings = self.get_all_encodings(x)
        final_result.update(all_encodings)
        z_disease, z_site = (
            all_encodings["z_disease"],
            all_encodings["z_site"],
        )

        decode_res = self.decode(z_disease, z_site)
        final_result.update(decode_res)

        final_result["y_hat"] = self.classify_disease(z_disease)
        final_result["d_hat"] = self.classify_site(z_site)
        final_result["y_dis"] = self.discriminate_disease(z_site)
        final_result["d_dis"] = self.discriminate_site(z_disease)
        return final_result

    def ss_forward(self, x):
        encode_res = self.encode(x)
        split_encode_res = self.split_encoding(encode_res["z_mu"])
        cls_res = self.classify_disease(split_encode_res["z_disease"])
        return cls_res["y"]

    def get_optimizer(self, param: Dict[str, Any]) -> Dict[str, Optimizer]:
        model_optim = Adam(
            reduce(
                lambda x, y: x + y,
                map(
                    list,
                    [
                        filter(
                            lambda p: p.requires_grad, self.encoder.parameters()
                        ),
                        filter(
                            lambda p: p.requires_grad,
                            self.site_encoder.parameters(),
                        ),
                        filter(
                            lambda p: p.requires_grad,
                            self.disease_encoder.parameters(),
                        ),
                        filter(
                            lambda p: p.requires_grad, self.decoder.parameters()
                        ),
                    ],
                ),
            ),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        disease_dis_optim = Adam(
            filter(lambda p: p.requires_grad, self.disease_dis.parameters()),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        site_dis_optim = Adam(
            filter(lambda p: p.requires_grad, self.site_dis.parameters()),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        disease_cls_optim = Adam(
            filter(lambda p: p.requires_grad, self.disease_cls.parameters()),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        site_cls_optim = Adam(
            filter(lambda p: p.requires_grad, self.site_cls.parameters()),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        optimizer = dict(
            model_optim=model_optim,
            disease_cls_optim=disease_cls_optim,
            site_cls_optim=site_cls_optim,
            disease_dis_optim=disease_dis_optim,
            site_dis_optim=site_dis_optim,
        )
        return optimizer

    def _classifier_discriminator_step(
        self,
        site_cls_optim: Optimizer,
        disease_cls_optim: Optimizer,
        site_dis_optim: Optimizer,
        disease_dis_optim: Optimizer,
        labeled_x: torch.Tensor,
        real_d: torch.Tensor,
        real_y: torch.Tensor,
        unlabeled_x: Optional[torch.Tensor] = None,
        gamma5: float = 0,
    ) -> Dict[str, Any]:
        """
        Train classifier and discriminator
        """
        site_cls_optim.zero_grad()
        disease_cls_optim.zero_grad()
        site_dis_optim.zero_grad()
        disease_dis_optim.zero_grad()
        losses = dict()

        labeled_encodings = self.get_all_encodings(labeled_x)
        labeled_z_disease = labeled_encodings["z_disease"].detach()
        labeled_z_site = labeled_encodings["z_site"].detach()
        disc_y = self.discriminate_disease(labeled_z_site)
        pred_y = self.classify_disease(labeled_z_disease)
        labeled_disc_d = self.discriminate_site(labeled_z_disease)
        labeled_pred_d = self.classify_site(labeled_z_site)
        if unlabeled_x is not None:
            unlabeled_encodings = self.get_all_encodings(unlabeled_x)
            unlabeled_z_disease = unlabeled_encodings["z_disease"].detach()
            unlabeled_z_site = unlabeled_encodings["z_site"].detach()
            unlabeled_disc_d = self.discriminate_site(unlabeled_z_disease)
            unlabeled_pred_d = self.classify_site(unlabeled_z_site)
            disc_d = torch.cat((labeled_disc_d, unlabeled_disc_d), dim=0)
            pred_d = torch.cat((labeled_pred_d, unlabeled_pred_d), dim=0)
        else:
            disc_d = labeled_disc_d
            pred_d = labeled_pred_d

        site_dis_loss = F.cross_entropy(disc_d, real_d)
        disease_dis_loss = F.cross_entropy(disc_y, real_y)
        site_cls_loss = F.cross_entropy(pred_d, real_d)
        disease_cls_loss = F.cross_entropy(pred_y, real_y)
        losses.update(
            dict(
                site_dis_loss=site_dis_loss.item(),
                disease_dis_loss=disease_dis_loss.item(),
                site_cls_loss=site_cls_loss.item(),
                disease_cls_loss=disease_cls_loss.item(),
            )
        )

        if gamma5 > 0:
            labeled_decodings = self.decode(labeled_z_disease, labeled_z_site)
            labeled_x_disease = labeled_decodings["x_disease"]
            labeled_encodings = self.get_all_encodings(labeled_x_disease)
            labeled_z_disease = labeled_encodings["z_disease"].detach()
            labeled_z_site = labeled_encodings["z_site"].detach()
            disc_y = self.discriminate_disease(labeled_z_site)
            pred_y = self.classify_disease(labeled_z_disease)
            labeled_disc_d = self.discriminate_site(labeled_z_disease)
            labeled_pred_d = self.classify_site(labeled_z_site)
            if unlabeled_x is not None:
                unlabeled_decodings = self.decode(
                    unlabeled_z_disease, unlabeled_z_site
                )
                unlabeled_x_disease = unlabeled_decodings["x_disease"]
                unlabeled_encodings = self.get_all_encodings(
                    unlabeled_x_disease
                )
                unlabeled_z_disease = unlabeled_encodings["z_disease"].detach()
                unlabeled_z_site = unlabeled_encodings["z_site"].detach()
                unlabeled_disc_d = self.discriminate_site(unlabeled_z_disease)
                unlabeled_pred_d = self.classify_site(unlabeled_z_site)
                disc_d = torch.cat((labeled_disc_d, unlabeled_disc_d), dim=0)
                pred_d = torch.cat((labeled_pred_d, unlabeled_pred_d), dim=0)
            else:
                disc_d = labeled_disc_d
                pred_d = labeled_pred_d

            site_dis_loss_2 = F.cross_entropy(disc_d, real_d)
            disease_dis_loss_2 = F.cross_entropy(disc_y, real_y)
            site_cls_loss_2 = F.cross_entropy(pred_d, real_d)
            disease_cls_loss_2 = F.cross_entropy(pred_y, real_y)
            losses.update(
                dict(
                    site_dis_loss_2=site_dis_loss_2.item(),
                    disease_dis_loss_2=disease_dis_loss_2.item(),
                    site_cls_loss_2=site_cls_loss_2.item(),
                    disease_cls_loss_2=disease_cls_loss_2.item(),
                )
            )

            site_dis_loss += gamma5 * site_dis_loss_2
            disease_dis_loss += gamma5 * disease_dis_loss_2
            site_cls_loss += gamma5 * site_cls_loss_2
            disease_cls_loss += gamma5 * disease_cls_loss_2

        site_dis_loss.backward()
        disease_dis_loss.backward()
        site_cls_loss.backward()
        disease_cls_loss.backward()
        site_dis_optim.step()
        disease_dis_optim.step()
        site_cls_optim.step()
        disease_cls_optim.step()

        return losses

    def _backbone_step(
        self,
        model_optim: Optimizer,
        labeled_x: torch.Tensor,
        real_y: torch.Tensor,
        real_d: torch.Tensor,
        unlabeled_x: Optional[torch.Tensor] = None,
        gamma1: float = 1,
        gamma2: float = 1,
        gamma3: float = 1,
        gamma4: float = 0,
        gamma5: float = 0,
    ) -> Dict[str, Any]:
        model_optim.zero_grad()
        metrics = dict()

        labeled_res: Dict[str, torch.Tensor] = self(labeled_x)
        pred_y = labeled_res["y_hat"]
        labeled_pred_d = labeled_res["d_hat"]
        disc_y = labeled_res["y_dis"]
        labeled_disc_d = labeled_res["d_dis"]
        labeled_x_mu = labeled_res["x_mu"]
        labeled_x_std = labeled_res["x_std"]
        labeled_z_mu = labeled_res["z_mu"]
        labeled_z_std = labeled_res["z_std"]
        labeled_z = labeled_res["z"]
        labeled_z_site = labeled_res["z_site"]
        labeled_z_disease = labeled_res["z_disease"]
        labeled_x_disease = labeled_res["x_disease"].detach()

        if unlabeled_x is not None:
            unlabeled_res: Dict[str, torch.Tensor] = self(unlabeled_x)
            unlabeled_pred_d = unlabeled_res["d_hat"]
            unlabeled_disc_d = unlabeled_res["d_dis"]
            unlabeled_x_mu = unlabeled_res["x_mu"]
            unlabeled_x_std = unlabeled_res["x_std"]
            unlabeled_z_mu = unlabeled_res["z_mu"]
            unlabeled_z_std = unlabeled_res["z_std"]
            unlabeled_z = unlabeled_res["z"]
            unlabeled_z_site = unlabeled_res["z_site"]
            unlabeled_z_disease = unlabeled_res["z_disease"]
            unlabeled_x_disease = unlabeled_res["x_disease"].detach()
            x = torch.cat((labeled_x, unlabeled_x), dim=0)
            pred_d = torch.cat((labeled_pred_d, unlabeled_pred_d), dim=0)
            disc_d = torch.cat((labeled_disc_d, unlabeled_disc_d), dim=0)
            x_mu = torch.cat((labeled_x_mu, unlabeled_x_mu), dim=0)
            x_std = torch.cat((labeled_x_std, unlabeled_x_std), dim=0)
            z_mu = torch.cat((labeled_z_mu, unlabeled_z_mu), dim=0)
            z_std = torch.cat((labeled_z_std, unlabeled_z_std), dim=0)
            z = torch.cat((labeled_z, unlabeled_z), dim=0)
            z_site = torch.cat((labeled_z_site, unlabeled_z_site), dim=0)
            z_disease = torch.cat(
                (labeled_z_disease, unlabeled_z_disease), dim=0
            )
            x_disease = torch.cat(
                (labeled_x_disease, unlabeled_x_disease), dim=0
            )
        else:
            x = labeled_x
            pred_d = labeled_pred_d
            disc_d = labeled_disc_d
            x_mu = labeled_x_mu
            x_std = labeled_x_std
            z_mu = labeled_z_mu
            z_std = labeled_z_std
            z = labeled_z
            z_site = labeled_z_site
            z_disease = labeled_z_disease
            x_disease = labeled_x_disease

        ce_y_loss = F.cross_entropy(pred_y, real_y)
        ce_d_loss = F.cross_entropy(pred_d, real_d)
        rc_x_loss = F.gaussian_nll_loss(x_mu, x, x_std ** 2)
        rc_z_loss = F.gaussian_nll_loss(z, z_disease + z_site, z_std ** 2)
        kl_loss = kl_divergence_loss(
            z_mu, z_std ** 2, torch.zeros_like(z_mu), torch.ones_like(z_std),
        )
        dis_loss = (entropy_loss(disc_d) + entropy_loss(disc_y)) / 2.0
        total_loss = (
            ce_y_loss
            + ce_d_loss
            + gamma1 * rc_x_loss
            + gamma2 * rc_z_loss
            + gamma3 * kl_loss
            + gamma4 * dis_loss
        )

        accuracy = CM.accuracy(real_y, pred_y)
        sensitivity = CM.tpr(real_y, pred_y)
        specificity = CM.tnr(real_y, pred_y)
        precision = CM.ppv(real_y, pred_y)
        f1_score = CM.f1_score(real_y, pred_y)

        metrics.update(
            {
                "ce_y_loss": ce_y_loss.item(),
                "ce_d_loss": ce_d_loss.item(),
                "rc_x_loss": rc_x_loss.item(),
                "rc_z_loss": rc_z_loss.item(),
                "kl_loss": kl_loss.item(),
                "discriminator_loss": dis_loss.item(),
                "accuracy": accuracy.item(),
                "sensitivity": sensitivity.item(),
                "specificity": specificity.item(),
                "f1": f1_score.item(),
                "precision": precision.item(),
            }
        )

        if gamma5 > 0:
            labeled_res = self(labeled_x_disease)
            pred_y = labeled_res["y_hat"]
            labeled_pred_d = labeled_res["d_hat"]
            disc_y = labeled_res["y_dis"]
            labeled_disc_d = labeled_res["d_dis"]
            labeled_x_mu = labeled_res["x_mu"]
            labeled_x_std = labeled_res["x_std"]
            labeled_z_mu = labeled_res["z_mu"]
            labeled_z_std = labeled_res["z_std"]
            labeled_z = labeled_res["z"]
            labeled_z_site = labeled_res["z_site"]
            labeled_z_disease = labeled_res["z_disease"]
            labeled_x_disease_2 = labeled_res["x_disease"]
            if unlabeled_x is not None:
                unlabeled_res = self(unlabeled_x_disease)
                unlabeled_pred_d = unlabeled_res["d_hat"]
                unlabeled_disc_d = unlabeled_res["d_dis"]
                unlabeled_x_mu = unlabeled_res["x_mu"]
                unlabeled_x_std = unlabeled_res["x_std"]
                unlabeled_z_mu = unlabeled_res["z_mu"]
                unlabeled_z_std = unlabeled_res["z_std"]
                unlabeled_z = unlabeled_res["z"]
                unlabeled_z_site = unlabeled_res["z_site"]
                unlabeled_z_disease = unlabeled_res["z_disease"]
                unlabeled_x_disease_2 = unlabeled_res["x_disease"]
                pred_d = torch.cat((labeled_pred_d, unlabeled_pred_d), dim=0)
                disc_d = torch.cat((labeled_disc_d, unlabeled_disc_d), dim=0)
                x_mu = torch.cat((labeled_x_mu, unlabeled_x_mu), dim=0)
                x_std = torch.cat((labeled_x_std, unlabeled_x_std), dim=0)
                z_mu = torch.cat((labeled_z_mu, unlabeled_z_mu), dim=0)
                z_std = torch.cat((labeled_z_std, unlabeled_z_std), dim=0)
                z = torch.cat((labeled_z, unlabeled_z), dim=0)
                z_site = torch.cat((labeled_z_site, unlabeled_z_site), dim=0)
                z_disease = torch.cat(
                    (labeled_z_disease, unlabeled_z_disease), dim=0
                )
                x_disease_2 = torch.cat(
                    (labeled_x_disease_2, unlabeled_x_disease_2), dim=0
                )
            else:
                pred_d = labeled_pred_d
                disc_d = labeled_disc_d
                x_mu = labeled_x_mu
                x_std = labeled_x_std
                z_mu = labeled_z_mu
                z_std = labeled_z_std
                z = labeled_z
                z_site = labeled_z_site
                z_disease = labeled_z_disease
                x_disease_2 = labeled_x_disease_2

            ce_y_loss_2 = F.cross_entropy(pred_y, real_y)
            ce_d_loss_2 = entropy_loss(pred_d)
            rc_x_loss_2 = (
                F.gaussian_nll_loss(x_mu, x_disease, x_std ** 2)
                + F.gaussian_nll_loss(x_disease_2, x_disease, x_std ** 2)
            ) / 2.0
            rc_z_loss_2 = (
                F.gaussian_nll_loss(z, z_disease, z_std ** 2)
                + F.gaussian_nll_loss(
                    torch.zeros_like(z_site), z_site, z_std ** 2
                )
            ) / 2.0
            kl_loss_2 = kl_divergence_loss(
                z_mu,
                z_std ** 2,
                torch.zeros_like(z_mu),
                torch.ones_like(z_std),
            )
            dis_loss_2 = (entropy_loss(disc_d) + entropy_loss(disc_y)) / 2.0
            total_loss += gamma5 * (
                ce_y_loss_2
                + ce_d_loss_2
                + gamma1 * rc_x_loss_2
                + gamma2 * rc_z_loss_2
                + gamma3 * kl_loss_2
                + gamma4 * dis_loss_2
            )

            accuracy_2 = CM.accuracy(real_y, pred_y)
            sensitivity_2 = CM.tpr(real_y, pred_y)
            specificity_2 = CM.tnr(real_y, pred_y)
            precision_2 = CM.ppv(real_y, pred_y)
            f1_score_2 = CM.f1_score(real_y, pred_y)

            metrics.update(
                {
                    "ce_y_loss_2": ce_y_loss_2.item(),
                    "ce_d_loss_2": ce_d_loss_2.item(),
                    "rc_x_loss_2": rc_x_loss_2.item(),
                    "rc_z_loss_2": rc_z_loss_2.item(),
                    "kl_loss_2": kl_loss_2.item(),
                    "discriminator_loss_2": dis_loss_2.item(),
                    "accuracy_2": accuracy_2.item(),
                    "sensitivity_2": sensitivity_2.item(),
                    "specificity_2": specificity_2.item(),
                    "f1_2": f1_score_2.item(),
                    "precision_2": precision_2.item(),
                }
            )

        total_loss.backward()
        model_optim.step()

    def train_step(
        self,
        device: torch.device,
        labeled_data: Data,
        unlabeled_data: Optional[Data],
        optimizer: Dict[str, Optimizer],
        hyperparameters: Dict[str, Any],
    ) -> Dict[str, float]:

        self.to(device)
        self.train()

        model_optim = optimizer["model_optim"]
        disease_cls_optim = optimizer["disease_cls_optim"]
        site_cls_optim = optimizer["site_cls_optim"]
        disease_dis_optim = optimizer["disease_dis_optim"]
        site_dis_optim = optimizer["site_dis_optim"]

        labeled_x: torch.Tensor = labeled_data.x
        real_y: torch.Tensor = labeled_data.y
        labeled_real_d: torch.Tensor = labeled_data.d
        labeled_x, real_y, labeled_real_d = (
            labeled_x.to(device),
            real_y.to(device),
            labeled_real_d.to(device),
        )

        if unlabeled_data is not None:
            unlabeled_x: torch.Tensor = unlabeled_data.x
            unlabeled_real_d: torch.Tensor = unlabeled_data.d
            unlabeled_x, unlabeled_real_d = (
                unlabeled_x.to(device),
                unlabeled_real_d.to(device),
            )
            real_d = torch.cat((labeled_real_d, unlabeled_real_d), dim=0)
        else:
            real_d = labeled_real_d

        gamma1 = hyperparameters.get("rc_x_loss", 1)
        gamma2 = hyperparameters.get("rc_z_loss", 1)
        gamma3 = hyperparameters.get("kl_loss", 1)
        gamma4 = hyperparameters.get("discriminator_loss", 0)
        gamma5 = hyperparameters.get("second_pass_loss", 0)
        metrics = dict()

        with torch.enable_grad():
            """
            Train classifier and discriminator
            """
            partial_losses = self._classifier_discriminator_step(
                site_cls_optim,
                disease_cls_optim,
                site_dis_optim,
                disease_dis_optim,
                labeled_x,
                real_d,
                real_y,
                unlabeled_x if unlabeled_data is not None else None,
                gamma5,
            )
            metrics.update(partial_losses)

            """
            Train Backbone
            """
            backbone_metrics = self._backbone_step(
                model_optim,
                labeled_x,
                real_y,
                real_d,
                unlabeled_x if unlabeled_data is not None else None,
                gamma1,
                gamma2,
                gamma3,
                gamma4,
                gamma5,
            )
            metrics.update(backbone_metrics)

        return metrics

    def test_step(
        self, device: torch.device, test_data: Data
    ) -> Dict[str, float]:
        self.to(device)
        self.eval()

        with torch.no_grad():
            x: torch.Tensor = test_data.x
            real_y: torch.Tensor = test_data.y
            real_d: torch.Tensor = test_data.d
            x, real_y, real_d = (
                x.to(device),
                real_y.to(device),
                real_d.to(device),
            )

            result: Dict[str, torch.Tensor] = self(x)
            pred_y = result["y_hat"]
            pred_d = result["d_hat"]
            disc_y = result["y_dis"]
            disc_d = result["d_dis"]
            x_mu = result["x_mu"]
            x_std = result["x_std"]
            z_mu = result["z_mu"]
            z_std = result["z_std"]
            z = result["z"]
            z_site = result["z_site"]
            z_disease = result["z_disease"]

            ce_y_loss = F.cross_entropy(pred_y, real_y)
            ce_d_loss = F.cross_entropy(pred_d, real_d)
            rc_x_loss = F.gaussian_nll_loss(x_mu, x, x_std ** 2)
            rc_z_loss = F.gaussian_nll_loss(z, z_disease + z_site, z_std ** 2)
            kl_loss = kl_divergence_loss(
                z_mu,
                z_std ** 2,
                torch.zeros_like(z_mu),
                torch.ones_like(z_std),
            )
            dis_loss = (entropy_loss(disc_d) + entropy_loss(disc_y)) / 2.0

            accuracy = CM.accuracy(real_y, pred_y)
            sensitivity = CM.tpr(real_y, pred_y)
            specificity = CM.tnr(real_y, pred_y)
            precision = CM.ppv(real_y, pred_y)
            f1_score = CM.f1_score(real_y, pred_y)

            metrics = {
                "ce_loss": ce_y_loss.item(),
                "ce_y_loss": ce_y_loss.item(),
                "ce_d_loss": ce_d_loss.item(),
                "rc_x_loss": rc_x_loss.item(),
                "rc_z_loss": rc_z_loss.item(),
                "kl_loss": kl_loss.item(),
                "discriminator_loss": dis_loss.item(),
                "accuracy": accuracy.item(),
                "sensitivity": sensitivity.item(),
                "specificity": specificity.item(),
                "f1": f1_score.item(),
                "precision": precision.item(),
            }
        return metrics


if __name__ == "__main__":
    # l1 = list(
    #     torch.load(
    #         "/data/yeww0006/FYP-SSL/.archive/exp20_ABIDE_WHOLE/ssl_ABIDE_1641297403/models/1641297649.pt"
    #     ).keys()
    # )

    # l2 = list(VAESDR(34716, 300, 150, 20).state_dict().keys())
    # print(l1)
    # print(l2)
    # print(dict(zip(l1, l2)))

    model = VAESDR.load_from_state_dict(
        "/data/yeww0006/FYP-SSL/.archive/exp20_ABIDE_WHOLE/ssl_ABIDE_1641297403/models/1641297649.pt",
        dict(
            num_sites=20,
            input_size=34716,
            hidden_size=300,
            emb_size=150,
            share_decoder=False,
        ),
    )
    print(model)

    x = torch.randn((10, 34716))
    res = model(x)
    for k, v in res.items():
        print("{}: {}".format(k, v.size()))

    model = VAESDR.load_from_state_dict(
        "/data/yeww0006/FYP-SSL/.archive/exp20_ABIDE_WHOLE/ssl_ABIDE_1639715346/models/1639716052.pt",
        dict(
            num_sites=20,
            input_size=34716,
            hidden_size=300,
            emb_size=150,
            share_decoder=True,
        ),
    )
    print(model)

    x = torch.randn((10, 34716))
    res = model(x)
    for k, v in res.items():
        print("{}: {}".format(k, v.size()))
