import os
import ast
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from data_processing import DataProcessing
from grouping import GroupSeedFold
from plot_utils import PlotCharts


def parse_sites(site):
    if isinstance(site, float) and np.isnan(site):
        return None
    try:
        site = ast.literal_eval(site)
        if isinstance(site, list):
            return tuple(site)
        return site
    except:
        return site


def color_dict():
    return {
        "VAE-FFN (SSL)": "deepskyblue",
        "VAE-FFN (SSL ComBat)": "steelblue",
        "VAECH-II (SSL)": "yellowgreen",
        "VAESDR (SSL)": "mediumslateblue",
        "EDC-VAE": "deepskyblue",
        "SHRED": "yellowgreen",
        "SHRED-II": "steelblue",
        "VAESDR": "mediumslateblue",
    }


def rename_models():
    return {
        "VAE-FFN (SSL)": "EDC-VAE",
        "VAE-FFN (SSL ComBat)": "SHRED-II",
        "VAECH-II (SSL)": "SHRED",
        "VAESDR (SSL)": "VAESDR",
    }


def rename_plot_target_cols():
    return [
        "EDC-VAE",
        "VAESDR",
        "SHRED",
        "SHRED-II",
    ]


def read_csv(folder: str):
    all_df = list()
    for sub_folder in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, sub_folder)):
            csv_path = os.path.join(
                folder, sub_folder, "{}.csv".format(sub_folder)
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


def load_csv(main_folder: str) -> pd.DataFrame:
    main_folder = os.path.abspath(main_folder)
    if not os.path.exists(main_folder):
        return

    df = read_csv(main_folder)
    if not df.empty:
        yield os.path.dirname(main_folder), df

    for sub_folder, _, _ in os.walk(main_folder):
        df = read_csv(sub_folder)
        if not df.empty:
            yield os.path.dirname(sub_folder), df


def evaluate(output_dir, df, metric_name="accuracy"):
    grouping = GroupSeedFold(df, groupby_unlabeled=True)
    grouped_df = grouping.group_seed_fold()

    metric_table_list = DataProcessing.prepare_metric_tables_per_group(
        grouped_df=grouped_df,
        metric_name=metric_name,
        grouping_cols=("dataset", "labeled_sites"),
        index_cols=("num_unlabeled_train",),
        num_subject_cols=None,
    )
    for metric_table in metric_table_list:
        PlotCharts.plot_metrics_line(
            output_dir=output_dir,
            file_suffix="N_SSL",
            metric_table=metric_table,
            target_cols=metric_table.mean.columns.tolist(),
            x_axis_label="Number of Unlabeled Data",
            color_dict=color_dict(),
        )
        PlotCharts.plot_metrics_line(
            output_dir=output_dir,
            file_suffix="FYP_N_SSL",
            metric_table=metric_table,
            target_cols=rename_plot_target_cols(),
            x_axis_label="Number of Unlabeled Data",
            color_dict=color_dict(),
            rename_experiments=rename_models(),
        )


def main(directory):
    all_folders_results = load_csv(directory)
    Parallel(n_jobs=10)(
        delayed(evaluate)(folder_path, df, "accuracy")
        for folder_path, df in all_folders_results
    )


if __name__ == "__main__":
    main("/data/yeww0006/FYP-SSL/.archive/ABLATION_N_SSL")
