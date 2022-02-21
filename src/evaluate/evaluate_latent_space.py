import logging
import os
import ast
import sys
import torch
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from typing import Any, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import on_error
from models.base import ModelBase
from factory import SingleStageFrameworkFactory
from data import Dataset, ModelBaseDataloader
from evaluate.plot_utils import PlotLatentSpace


def load_model(param: Dict[str, Any]) -> ModelBase:
    model_cls: ModelBase = SingleStageFrameworkFactory.get_model_class(
        param["model_name"]
    )
    model = model_cls.load_from_state_dict(
        param["model_path"], ast.literal_eval(param["model_params"])
    )
    model.eval()
    return model


def parse_sites(site):
    if isinstance(site, float) and np.isnan(site):
        return None
    try:
        return ast.literal_eval(site)
    except:
        return site


def read_csv(input_dir):
    main_folder = os.path.abspath(input_dir)
    if not os.path.exists(main_folder):
        return

    for sub_folder, _, _ in os.walk(main_folder):
        if "deprecated" in sub_folder:
            continue
        base_sub_folder = os.path.basename(sub_folder)
        csv_path = os.path.join(sub_folder, "{}.csv".format(base_sub_folder))
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path).dropna(subset=["model_path"])
        if df.empty:
            continue
        try:
            df["labeled_sites"] = df["labeled_sites"].apply(parse_sites)
        except Exception as e:
            logging.error("{}: {}".format(sub_folder, e))
        try:
            df["unlabeled_sites"] = df["unlabeled_sites"].apply(parse_sites)
        except Exception as e:
            logging.error("{}: {}".format(sub_folder, e))

        output_dir = os.path.join(os.path.dirname(sub_folder), "latent_space")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        yield output_dir, df


def get_name(
    dataset, labeled_sites, unlabeled_sites, model_name, ssl, harmonize
) -> str:
    name_components = [dataset]

    if isinstance(labeled_sites, (list, tuple)):
        name_components.append("-".join(labeled_sites))
    else:
        name_components.append(str(labeled_sites))
    if isinstance(unlabeled_sites, (list, tuple)):
        name_components.append("-".join(unlabeled_sites))
    else:
        name_components.append(str(unlabeled_sites))

    name_components.append(model_name)
    name_components.append("SSL" if ssl else "SL")
    if harmonize:
        name_components.append("ComBat")
    return "_".join(name_components)


def create_subplots(num_rows):
    ncol = int(np.sqrt(num_rows))
    nrow = int(np.ceil(num_rows / float(ncol)))

    def ax_iterator():
        if nrow == 1 and ncol == 1:
            yield None
        elif nrow == 1 or ncol == 1:
            for i in range(max(nrow, ncol)):
                yield i
        else:
            for r in range(nrow):
                for c in range(ncol):
                    yield r, c

    f, ax = plt.subplots(nrow, ncol, figsize=(8 * ncol, 8 * nrow))
    return f, ax, ax_iterator


@on_error(None, True)
def plot_latent_space(
    output_dir: str, df: pd.DataFrame, data_dict: Dict[str, ModelBaseDataloader]
):
    assert df["model_name"].nunique() == 1
    assert df["dataset"].nunique() == 1
    assert df["labeled_sites"].nunique(dropna=False) == 1
    assert df["unlabeled_sites"].nunique(dropna=False) == 1
    assert df["ssl"].nunique() == 1
    assert df["harmonize"].nunique() == 1

    model_name = df["model_name"].iloc[0]
    ssl = df["ssl"].iloc[0]

    dataset = df["dataset"].iloc[0]
    harmonize = df["harmonize"].iloc[0]
    dataloader = data_dict[dataset, harmonize]

    labeled_sites = df["labeled_sites"].iloc[0]
    unlabeled_sites = df["unlabeled_sites"].iloc[0]
    sites = dataloader.sites

    title = "{} | SSL: {} | ComBat: {}\n{} - Labeled: {} | Unlabeled: {}\n".format(
        model_name, ssl, harmonize, dataset, labeled_sites, unlabeled_sites
    )

    data = dataloader.load_all_data()["data"]
    labeled_idx = torch.tensor(
        np.isin(sites, labeled_sites)
        if isinstance(labeled_sites, (tuple, list))
        else (sites == labeled_sites)
        if labeled_sites is not None
        else np.ones(sites.shape, dtype=bool)
    )
    unlabeled_idx = (
        torch.tensor(
            np.isin(sites, unlabeled_sites)
            if isinstance(unlabeled_sites, (tuple, list))
            else (sites == unlabeled_sites)
            if unlabeled_sites is not None
            else np.ones(sites.shape, dtype=bool)
        )
        & ~labeled_idx
    )
    y: torch.Tensor = data.y

    param_list = df.sort_values(by="accuracy", ascending=False).to_dict(
        "records"
    )
    f, ax, ax_iterator = create_subplots(min(25, len(df)))

    for i, ax_idx in enumerate(ax_iterator()):
        if i >= len(param_list):
            ax[ax_idx].axis("off")
            continue

        param = param_list[i]
        model = load_model(param)
        ls_encoding = model.get_latent_space_encoding(data)
        emb = ls_encoding["x"]
        x_dict = dict()
        x_dict_kwargs = dict()

        labeled_control = emb[labeled_idx & (y == 0)]
        labeled_disease = emb[labeled_idx & (y == 1)]
        unlabeled_control = emb[unlabeled_idx & (y == 0)]
        unlabeled_disease = emb[unlabeled_idx & (y == 1)]
        x_dict["unlabeled control"] = unlabeled_control
        x_dict["unlabeled diseased"] = unlabeled_disease
        x_dict["{} control".format(labeled_sites)] = labeled_control
        x_dict["{} diseased".format(labeled_sites)] = labeled_disease
        x_dict_kwargs["{} control".format(labeled_sites)] = {
            "color": "tab:blue",
            "marker": "^",
        }
        x_dict_kwargs["{} diseased".format(labeled_sites)] = {
            "color": "tab:orange",
            "marker": "^",
        }
        x_dict_kwargs["unlabeled control"] = {
            "color": "deepskyblue",
            "marker": ".",
        }
        x_dict_kwargs["unlabeled diseased"] = {
            "color": "gold",
            "marker": ".",
        }

        if ax_idx is None:
            axis = ax
        else:
            axis = ax[ax_idx]
        PlotLatentSpace.plot_result(
            axis,
            (ls_encoding["xx"], ls_encoding["yy"], ls_encoding["zz"]),
            x_dict,
            x_dict_kwargs=x_dict_kwargs,
            title="Seed: {} | Fold: {}\nAccuracy: {}".format(
                param["seed"], param["fold"], param["accuracy"]
            ),
        )

    plt.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    filename = get_name(
        dataset, labeled_sites, unlabeled_sites, model_name, ssl, harmonize
    )
    f.savefig(
        os.path.join(output_dir, "{}_latent_space_plot.png".format(filename))
    )


def main(args):
    input_dir = os.path.abspath(args.input)

    data_dict = dict()
    for dataset in Dataset:
        for harmonize in [True, False]:
            data_dict[dataset.value, harmonize] = ModelBaseDataloader(
                dataset, harmonize
            )

    n_jobs = args.worker
    Parallel(n_jobs)(
        delayed(plot_latent_space)(output_dir, df, data_dict)
        for output_dir, df in read_csv(input_dir)
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] - %(filename)s: %(levelname)s: "
        "%(funcName)s(): %(lineno)d:\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="../../.archive",
        help="path to directory containing training results",
    )
    parser.add_argument(
        "--worker",
        type=int,
        default=10,
        help="number of workers to run in parallel",
    )
    args = parser.parse_args()
    main(args)
