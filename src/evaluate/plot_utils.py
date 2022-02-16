import os
import math
import numpy as np
from contextlib import contextmanager
from typing import Any, Sequence, Dict, Tuple
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from data_processing import MetricTable


@contextmanager
def catch_exception():
    try:
        yield
    except Exception as e:
        print(e)


class PlotCharts:
    @classmethod
    def plot_metrics_bar(
        cls,
        output_dir: str,
        file_suffix: str,
        metric_table: MetricTable,
        target_cols: Sequence[str],
        x_axis_label: str = "Labeled Site",
        num_subjects_y_axis_label: str = "Number of Subjects",
        color_dict: Dict[str, str] = dict(),
    ):
        metric_name = metric_table.metric_name
        group_name = " ".join(metric_table.group.values())
        num_subjects = metric_table.num_subjects
        if num_subjects is not None:
            num_subjects = num_subjects.sort_values(by=num_subjects.columns[0])
            sorted_index = num_subjects.index
            metric_mean = metric_table.mean.loc[sorted_index, :]
            if metric_table.std is not None:
                metric_std = metric_table.std.loc[sorted_index, :]
            else:
                metric_std = None
        else:
            metric_mean = metric_table.mean
            metric_std = metric_table.std

        error_kw = dict(ecolor="gray")
        width = max(10, metric_mean.shape[0] * 1.25)
        height = 5
        ylim0 = 0.5
        ylim1 = 1.0

        with catch_exception() as c:
            target_cols = [
                col for col in target_cols if col in metric_mean.columns
            ]
            if not color_dict:
                colors = None
            else:
                colors = [color_dict[x] for x in target_cols]

            ax1 = metric_mean[target_cols].plot(
                kind="bar",
                rot=45,
                yerr=metric_std,
                error_kw=error_kw,
                ylim=(ylim0, ylim1),
                figsize=(width, height),
                color=colors,
            )
            ax1.set_xlabel(x_axis_label)
            ax1.set_ylabel(metric_name.title())
            legend_1 = plt.legend()

            if num_subjects is not None:
                ax2 = num_subjects.plot(
                    ax=ax1,
                    kind="line",
                    color="r",
                    marker="o",
                    secondary_y=True,
                    label=num_subjects_y_axis_label,
                    xlabel=x_axis_label,
                )
                ax2.set_xlabel(x_axis_label)
                ax2.set_ylabel(num_subjects_y_axis_label)
                legend_2 = plt.legend()

                legend_1.remove()
                legend_2.remove()

                plt.legend(
                    legend_1.get_patches() + legend_2.get_lines(),
                    [text.get_text() for text in legend_1.get_texts()]
                    + [num_subjects_y_axis_label],
                    loc="best",
                    ncol=int(math.ceil((metric_mean.shape[1] + 1) / 4.0)),
                )

            plt.title(
                "{} {} FOR EACH {}".format(
                    group_name.upper(),
                    metric_name.upper(),
                    x_axis_label.upper(),
                )
            )
            plt.tight_layout()
            output_dir = os.path.abspath(output_dir)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            if file_suffix:
                filename = "{}_{}_{}_bar_plots".format(
                    group_name, file_suffix, metric_name
                )
            else:
                filename = "{}_{}_bar_plots".format(group_name, metric_name)
            save_path = os.path.join(output_dir, "{}.png".format(filename),)
            plt.savefig(save_path)
            plt.close()

    @classmethod
    def plot_metrics_line(
        cls,
        output_dir: str,
        file_suffix: str,
        metric_table: MetricTable,
        target_cols: Sequence[str],
        x_axis_label: str = "Number of Unlabeled Data",
        color_dict: Dict[str, str] = dict(),
    ):
        metric_name = metric_table.metric_name
        group_name = " ".join(metric_table.group.values())
        metric_mean = metric_table.mean.sort_index()

        width = 10
        height = 5

        with catch_exception() as c:
            target_cols = [
                col for col in target_cols if col in metric_mean.columns
            ]
            metric_mean = metric_mean[target_cols]
            metric_mean.index = map(int, metric_mean.index)

            if not color_dict:
                colors = None
            else:
                colors = [color_dict[x] for x in target_cols]

            ax1 = metric_mean.plot(
                kind="line", marker="o", color=colors, figsize=(width, height),
            )
            ax1.set_xlabel(x_axis_label)
            ax1.set_ylabel(metric_name.title())
            plt.legend(loc="best")
            plt.title(
                "{} {} AGAINST {}".format(
                    group_name.upper(),
                    metric_name.upper(),
                    x_axis_label.upper(),
                )
            )
            plt.tight_layout()
            output_dir = os.path.abspath(output_dir)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            if file_suffix:
                filename = "{}_{}_{}_line_plots".format(
                    group_name, file_suffix, metric_name
                )
            else:
                filename = "{}_{}_line_plots".format(group_name, metric_name)
            save_path = os.path.join(output_dir, "{}.png".format(filename),)
            plt.savefig(save_path)
            plt.close()


class PlotLatentSpace:
    @classmethod
    def plot_result(
        cls,
        ax: Axes,
        surface: Tuple[np.ndarray, np.ndarray, np.ndarray],
        x_dict: Dict[str, np.ndarray],
        surface_kwargs: Dict[str, Any] = dict(cmap="Paired", alpha=0.3),
        x_dict_kwargs: Dict[str, Dict[str, Any]] = dict(),
        title: str = "",
        x_label: str = "PCA 1",
        y_label: str = "PCA 2",
    ):
        xx, yy, zz = surface
        zz = np.round(zz).astype(int)

        ax.contourf(xx, yy, zz, **surface_kwargs)
        for label, xt in x_dict.items():
            ax.scatter(
                xt[:, 0],
                xt[:, 1],
                label=label,
                **x_dict_kwargs.get(label, dict())
            )

        if title:
            ax.set_title(title)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()

