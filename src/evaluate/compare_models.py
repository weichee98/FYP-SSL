import os
import pandas as pd

from plot_utils import PlotCharts
from grouping import GroupSeedFold
from data_processing import DataProcessing
from export import ExportDocument


def read_csv(main_folder: str) -> pd.DataFrame:
    main_folder = os.path.abspath(main_folder)
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


def color_dict():
    return {
        "ASDSAENet (SL)": "goldenrod",
        "GCN-FCNN (SL)": "gold",
        "GAE-FCNN (SL)": "orange",
        "VAE-FFN (SL)": "cadetblue",
        "VAE-FFN (SSL)": "deepskyblue",
        "VAE-FFN (SL ComBat)": "steelblue",
        "VAE-FFN (SSL ComBat)": "steelblue",
        "VAECH (SSL)": "tab:green",
        "VAECH-I (SSL)": "springgreen",
        "VAECH-II (SSL)": "yellowgreen",
        "VAECH (SL)": "tab:green",
        "VAECH-I (SL)": "springgreen",
        "VAECH-II (SL)": "yellowgreen",
        "EDC-VAE (SSL)": "deepskyblue",
        "SHRED (SSL)": "springgreen",
        "SHRED-II (SSL)": "steelblue",
        "SHRED-III (SSL)": "yellowgreen",
        "VAESDR (SL)": "mediumslateblue",
        "VAESDR (SSL)": "mediumslateblue",
        "ASDSAENet": "goldenrod",
        "GAE-FCNN": "orange",
        "EDC-VAE": "deepskyblue",
        "VAESDR": "mediumslateblue",
        "SHRED": "springgreen",
        "SHRED-II": "steelblue",
    }


def plot_target_cols():
    return {
        "FFN_AE_VAE": [
            "FFN (SL)",
            "AE-FFN (SL)",
            "VAE-FFN (SL)",
            "FFN (SL ComBat)",
            "AE-FFN (SL ComBat)",
            "VAE-FFN (SL ComBat)",
            "AE-FFN (SSL)",
            "VAE-FFN (SSL)",
            "AE-FFN (SSL ComBat)",
            "VAE-FFN (SSL ComBat)",
        ],
        "VAESDR": [
            "VAE-FFN (SL)",
            "VAESDR (SL)",
            "VAESDR-D (SL)",
            "VAESDR-DS (SL)",
            "VAESDR-W (SL)",
            "VAESDR-DW (SL)",
            "VAESDR-DSW (SL)",
            "VAE-FFN (SSL)",
            "VAESDR (SSL)",
            "VAESDR-D (SSL)",
            "VAESDR-DS (SSL)",
            "VAESDR-W (SSL)",
            "VAESDR-DW (SSL)",
            "VAESDR-DSW (SSL)",
            "VAE-FFN (SL ComBat)",
            "VAE-FFN (SSL ComBat)",
        ],
        "VAECH": [
            "VAE-FFN (SL)",
            "VAECH (SL)",
            # "VAEVCH (SL)",
            "VAECH-I (SL)",
            "VAECH-II (SL)",
            "VAE-FFN (SSL)",
            "VAECH (SSL)",
            # "VAEVCH (SSL)",
            "VAECH-I (SSL)",
            "VAECH-II (SSL)",
            "VAE-FFN (SL ComBat)",
            "VAE-FFN (SSL ComBat)",
        ],
        "": [
            "ASDSAENet (SL)",
            "GCN-FCNN (SL)",
            "GAE-FCNN (SL)",
            "VAE-FFN (SL)",
            "VAE-FFN (SSL)",
            "VAECH-I (SSL)",
            "VAE-FFN (SSL ComBat)",
        ],
    }


def rename_models():
    return {
        "ASDSAENet (SL)": "ASDSAENet",
        "GAE-FCNN (SL)": "GAE-FCNN",
        "VAE-FFN (SSL)": "EDC-VAE",
        "VAE-FFN (SSL ComBat)": "SHRED-II",
        # "VAECH-I (SSL)": "SHRED (SSL)",
        # "VAECH-II (SSL)": "SHRED-III (SSL)",
        "VAECH-II (SSL)": "SHRED",
        "VAESDR (SSL)": "VAESDR",
    }


def rename_plot_target_cols():
    return {
        # "MICCAI": [
        #     "ASDSAENet (SL)",
        #     "GAE-FCNN (SL)",
        #     "EDC-VAE (SSL)",
        #     "SHRED (SSL)",
        #     "SHRED-II (SSL)",
        # ],
        # "SHRED": ["SHRED (SSL)", "SHRED-II (SSL)", "SHRED-III (SSL)",],
        # "SHRED_VAESDR": ["SHRED (SSL)", "SHRED-II (SSL)", "SHRED-III (SSL)", "VAESDR (SSL)"],
        "FYP": [
            "ASDSAENet",
            "GAE-FCNN",
            "EDC-VAE",
            "VAESDR",
            "SHRED",
            "SHRED-II",
        ],
    }


def plot_metrics_bar(grouped_df, output_dir, metric_name="accuracy"):
    metric_table_list = DataProcessing.prepare_metric_tables_per_group(
        grouped_df=grouped_df,
        metric_name=metric_name,
        grouping_cols=("dataset",),
        index_cols=("labeled_sites",),
        num_subject_cols="num_subjects",
    )
    for metric_table in metric_table_list:
        ExportDocument.export_metric_table_docx(
            output_dir=output_dir, metric_table=metric_table
        )
        for file_suffix, target_cols in plot_target_cols().items():
            PlotCharts.plot_metrics_bar(
                output_dir=output_dir,
                file_suffix=file_suffix,
                metric_table=metric_table,
                target_cols=target_cols,
                x_axis_label="Labeled Site",
                num_subjects_y_axis_label="Number of Subjects",
                color_dict=color_dict(),
            )
        for file_suffix, target_cols in rename_plot_target_cols().items():
            PlotCharts.plot_metrics_bar(
                output_dir=output_dir,
                file_suffix=file_suffix,
                metric_table=metric_table,
                target_cols=target_cols,
                x_axis_label="Labeled Site",
                num_subjects_y_axis_label="Number of Subjects",
                color_dict=color_dict(),
                rename_experiments=rename_models(),
            )


def main(directory):
    df = read_csv(directory)
    if df.empty:
        return

    output_dir = directory

    grouping = GroupSeedFold(df)
    grouped_df = grouping.group_seed_fold()

    plot_metrics_bar(grouped_df, output_dir, metric_name="accuracy")

    for dataset in grouped_df["dataset"].unique():
        dataset_df = grouped_df[grouped_df["dataset"] == dataset]
        overall_df = grouping.group_overall_dataset(dataset_df)

        ExportDocument.export_csv(
            output_dir=output_dir,
            dataset_name=dataset,
            grouped_df=dataset_df,
            overall_df=overall_df,
        )

        accuracy_ttest = DataProcessing.prepare_ttest_table_per_group(
            grouped_df=dataset_df, metric_name="accuracy",
        )
        overall_ttest = DataProcessing.prepare_ttest_table(
            grouped_df=overall_df,
            metric_name="accuracy",
            group={"dataset": dataset},
        )

        ExportDocument.export_excel_sheet(
            output_dir=output_dir,
            dataset_name=dataset,
            grouped_df=dataset_df,
            overall_df=overall_df,
            grouped_ttest_table=accuracy_ttest,
            overall_ttest_table=[overall_ttest],
        )

    pass


if __name__ == "__main__":
    main("/data/yeww0006/FYP-SSL/.archive/ABIDE_INDIVIDUAL")
    main("/data/yeww0006/FYP-SSL/.archive/ADHD_INDIVIDUAL")
    main("/data/yeww0006/FYP-SSL/.archive/ABIDE_WHOLE")
    main("/data/yeww0006/FYP-SSL/.archive/ADHD_WHOLE")
