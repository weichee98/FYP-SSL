import os
import ast
import sys
import traceback
import argparse
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from torch_geometric.data import Data
from typing import Any, Dict, Optional, Sequence, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import ModelBase
from factory import SingleStageFrameworkFactory
from utils import seed_torch
from data import Dataset, ModelBaseDataloader
from biomarker.visualize import (
    ABIDEBiomarkersVisualizer,
    ADHDBiomarkersVisualizer,
    PowerCrossleyVisualizer,
)


score_matrix_filename = "model_mean_score.npy"
score_matrix_meta = "model_mean_score_meta.csv"


def get_score_path(param: Dict[str, Any]) -> np.ndarray:
    model_path = os.path.abspath(param["model_path"])
    biomarkers_dir = os.path.join(
        os.path.dirname(os.path.dirname(model_path)), "score_matrices"
    )
    try:
        os.makedirs(biomarkers_dir)
    except FileExistsError:
        pass
    basename = os.path.basename(model_path).replace(".pt", "")
    score_path = os.path.join(biomarkers_dir, "{}.npy".format(basename))
    return score_path


def load_model(param: Dict[str, Any]) -> ModelBase:
    model_cls: ModelBase = SingleStageFrameworkFactory.get_model_class(
        param["model_name"]
    )
    model = model_cls.load_from_state_dict(
        param["model_path"], ast.literal_eval(param["model_params"])
    )
    return model


def parse_sites(sites):
    if isinstance(sites, float) and np.isnan(sites):
        return []
    try:
        site_list = ast.literal_eval(sites)
        if isinstance(site_list, (list, tuple)):
            return list(site_list)
    except:
        pass
    return [sites]


def model_score(
    param: Dict[str, Any], data_dict: Dict[str, Data], force: bool
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    seed_torch()
    try:
        score_path = get_score_path(param)
        if os.path.exists(score_path) and not force:
            scores = np.load(score_path, allow_pickle=True)
        else:
            dataloader: ModelBaseDataloader = data_dict[
                param["dataset"], param["harmonize"]
            ]
            model: ModelBase = load_model(param)
            try:
                data = dataloader.load_all_data(num_process=10)["data"]
                scores = model.saliency_score(data)
            except:
                labeled_sites = parse_sites(param["labeled_sites"])
                unlabeled_sites = parse_sites(param["unlabeled_sites"])
                data = dataloader.load_all_data(
                    sites=labeled_sites + unlabeled_sites, num_process=10
                )["data"]
                scores = model.saliency_score(data)
            scores = scores.mean(axis=0)
            np.save(score_path, scores)
        param["max_score"] = scores.max()
        param["min_score"] = scores.min()
        param["mean_score"] = scores.mean()
        param["std_score"] = scores.std()
        return scores, param
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        return None, dict()


def read_csv(input_dir):
    main_folder = os.path.abspath(input_dir)
    if not os.path.exists(main_folder):
        return pd.DataFrame()

    all_df = list()
    for sub_folder in os.listdir(main_folder):
        if os.path.isdir(os.path.join(main_folder, sub_folder)):
            csv_path = os.path.join(
                main_folder, sub_folder, "{}.csv".format(sub_folder)
            )
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            all_df.append(df)

    if not all_df:
        return pd.DataFrame()
    if len(all_df) == 1:
        return all_df[0]
    return pd.concat(all_df, axis=0, ignore_index=True).reset_index(drop=True)


def generate_score_matrices(
    input_dir: str, output_dir: str, n_jobs: int, force: bool
) -> Tuple[pd.DataFrame, np.ndarray]:
    model_path_csv = read_csv(input_dir)
    model_path_csv = model_path_csv.dropna(subset=["model_path"])

    data_dict = dict()
    for dataset in Dataset:
        for harmonize in [True, False]:
            dataloader = ModelBaseDataloader(dataset, harmonize)
            data_dict[dataset.value, harmonize] = dataloader

    results = Parallel(n_jobs=n_jobs)(
        delayed(model_score)(param, data_dict, force)
        for param in tqdm(model_path_csv.to_dict("records"), ncols=60)
    )

    all_scores, all_params = zip(
        *list(filter(lambda x: x[0] is not None, results))
    )
    all_scores = np.array(all_scores)
    output_csv_path = os.path.join(output_dir, score_matrix_meta)
    params_df = pd.DataFrame(all_params)
    params_df.to_csv(output_csv_path, index=False)
    return params_df, all_scores


def get_name(param: Dict[str, Any], group_columns: Sequence[str]) -> str:
    name_components = []
    for column in group_columns:
        if column not in param:
            continue
        if column == "ssl":
            name_components.append("SSL" if param[column] else "SL")
        elif column == "harmonize":
            if param[column]:
                name_components.append("ComBat")
        elif column in ["labeled_sites", "unlabeled_sites"]:
            try:
                value = ast.literal_eval(param[column])
            except:
                value = param[column]
            if isinstance(value, (list, tuple)):
                name_components.append("-".join(value))
            else:
                name_components.append(str(value))
        else:
            name_components.append(str(param[column]))
    return "_".join(name_components)


def plot_biomarkers(
    param: Dict[str, Any],
    score_matrices: np.ndarray,
    viz_dict: Dict[str, PowerCrossleyVisualizer],
    gcol: Sequence[str],
    output_dir: str,
):
    idx = param["index"]
    matrix = np.mean(score_matrices[idx], axis=0)
    prefix = get_name(param, gcol)
    output_dir = os.path.join(output_dir, prefix)

    viz = viz_dict[param["dataset"]]
    viz.plot_connectome(matrix, os.path.join(output_dir, "connectome.png"))
    viz.plot_stat_map(matrix, output_dir, threshold=0.1, vmax=10)
    viz.plot_module_importance_boxplot(
        matrix, os.path.join(output_dir, "boxplot.png")
    )
    viz.plot_complete_score_matrix(
        matrix, os.path.join(output_dir, "conn_mat.png")
    )
    _, msm, module_labels = viz.plot_module_sensitivity_map(
        matrix, os.path.join(output_dir, "msm.png"), vmax=10
    )
    param["biomarkers"] = msm.tolist()
    param["module_labels"] = module_labels.tolist()
    return param


def visualize_biomarkers(
    df: pd.DataFrame, score_matrices: np.ndarray, output_dir: str, n_jobs: int
):
    assert len(df) == len(score_matrices)

    gcol = [
        "dataset",
        "labeled_sites",
        "unlabeled_sites",
        "model_name",
        "ssl",
        "harmonize",
    ]

    df["index"] = range(len(df))
    df = df.groupby(gcol, dropna=False)["index"].apply(list).reset_index()

    viz_dict = dict()
    for dataset in Dataset:
        if dataset == Dataset.ABIDE:
            viz = ABIDEBiomarkersVisualizer()
        elif dataset == Dataset.ADHD:
            viz = ADHDBiomarkersVisualizer()
        else:
            raise NotImplementedError
        viz_dict[dataset.value] = viz

    result = Parallel(n_jobs=n_jobs)(
        delayed(plot_biomarkers)(
            param, score_matrices, viz_dict, gcol, output_dir
        )
        for param in tqdm(df.to_dict("records"), ncols=60)
    )
    result_df = pd.DataFrame(result)
    result_df.to_parquet(os.path.join(output_dir, "biomarkers.parquet"))


def main(args):
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.join(input_dir, "biomarkers")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_jobs = args.worker
    force = args.force
    df, score_matrices = generate_score_matrices(
        input_dir, output_dir, 1, force
    )
    visualize_biomarkers(df, score_matrices, output_dir, n_jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="path to directory containing training results",
    )
    parser.add_argument(
        "--worker",
        type=int,
        default=10,
        help="number of workers to run in parallel",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args)
