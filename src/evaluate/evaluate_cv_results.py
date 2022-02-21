import os
import json
from typing import Dict, List

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def read_csv(main_folder: str) -> Dict[str, pd.DataFrame]:
    main_folder = os.path.abspath(main_folder)
    if not os.path.exists(main_folder):
        return dict()

    all_df = dict()
    for sub_folder in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, sub_folder)
        if not os.path.isdir(folder_path):
            continue
        csv_path = os.path.join(
            main_folder, sub_folder, "{}_cv_results.csv".format(sub_folder)
        )
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        all_df[folder_path] = df
    return all_df


def evaluate(folder_path: str, cv_results: pd.DataFrame):
    columns: List[str] = cv_results.columns.tolist()
    hyperparameter_cols = [
        col
        for col in columns
        if col.startswith("model_params__")
        or col.startswith("optim_params__")
        or col.startswith("hyperparameters__")
    ]

    config_path = os.path.join(
        folder_path, "{}.config.json".format(os.path.basename(folder_path))
    )
    with open(config_path, "r") as f:
        config = json.load(f)
    metric_name: str = config["strategy"]["metric"]
    metric_col = "mean_" + metric_name

    for col in hyperparameter_cols:
        if cv_results[col].nunique() == 1:
            continue
        f, ax = plt.subplots(1, 1, figsize=(8, 8))
        sb.boxplot(data=cv_results, x=col, y=metric_col, showmeans=True, ax=ax)

        ax.set_title(
            "{} DISTRIBUTION FOR {}".format(metric_name.upper(), col.upper())
        )
        ax.set_ylabel(metric_name)
        ax.set_xlabel(col)

        num_samples = {
            str(k): v
            for k, v in cv_results[col].value_counts().to_dict().items()
        }
        ax.set_xticklabels(
            [
                str(x.get_text()) + "\nn = {}".format(num_samples[x.get_text()])
                for x in ax.get_xticklabels()
            ]
        )

        plt.tight_layout()
        f.savefig(
            os.path.join(
                folder_path, "{}_{}_boxplot.png".format(col, metric_name)
            )
        )


def main(directory):
    all_folders_results = read_csv(directory)
    Parallel(n_jobs=10)(
        delayed(evaluate)(folder_path, cv_results)
        for folder_path, cv_results in all_folders_results.items()
    )


if __name__ == "__main__":
    main("/data/yeww0006/FYP-SSL/.archive/ABIDE_TUNING")
