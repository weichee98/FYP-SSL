import os
import sys
import torch
from typing import Dict, Tuple
from torch_geometric.data import Data

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)

from models.VAECH import VAECH


class VAECH_I(VAECH):
    def forward(
        self,
        x: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        site: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ch_res = self.ch(x, age, gender, site)
        vae_res = self.vae_ffn(ch_res["x_ch"])

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

    def ss_forward(self, x_ch: torch.Tensor) -> torch.Tensor:
        y = self.vae_ffn.ss_forward(x_ch)
        return y

    def get_baselines_inputs(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = data.x, data.y
        age, gender, site = data.age, data.gender, data.d
        ch_res = self.combat(x, age, gender, site)
        x_ch: torch.Tensor = ch_res["x_ch"]
        baselines = x_ch[y == 0].mean(dim=0).view(1, -1)
        inputs = x_ch[y == 1]
        return baselines, inputs


if __name__ == "__main__":
    model = VAECH_I.load_from_state_dict(
        "/data/yeww0006/FYP-SSL/.archive/exp06_rerun_ffn/ssl_ABIDE_1644540995/models/1644541153.pt",
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
