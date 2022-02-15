import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Sequence
from dataclasses import dataclass, field
from scipy.stats import ttest_ind


@dataclass(frozen=True)
class MetricTable:
    metric_name: str
    mean: pd.DataFrame
    std: Optional[pd.DataFrame] = field(default=None)
    num_subjects: Optional[pd.DataFrame] = field(default=None)
    group: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TTestTable:
    metric_name: str
    result_table: pd.DataFrame
    ttest_table: pd.DataFrame
    group: Dict[str, Any] = field(default_factory=dict)


class DataProcessing:
    @classmethod
    def __legend_name(cls, row):
        if "model_name" not in row:
            raise KeyError("model_name not found")

        name = row["model_name"]
        desc = ""

        if "ssl" in row:
            desc = "" if not desc else desc + " "
            desc += "SSL" if row["ssl"] else "SL"

        if "harmonize" in row and row["harmonize"]:
            desc = "" if not desc else desc + " "
            desc += "ComBat"

        if desc:
            name = "{} ({})".format(name, desc)
        return name

    @classmethod
    def create_metric_table(
        cls,
        dataset_df: pd.DataFrame,
        metric_name: str = "accuracy",
        index_cols: Sequence[str] = ("labeled_sites",),
        num_subject_cols: str = "num_subjects",
        group: Dict[str, Any] = dict(),
    ) -> MetricTable:
        mean_metric_col = "{} | mean".format(metric_name)
        if mean_metric_col not in dataset_df.columns:
            mean_metric_col = metric_name
            if mean_metric_col not in dataset_df.columns:
                raise Exception('metric "{}" not found'.format(metric_name))

        dataset_df = dataset_df.copy()
        dataset_df["experiment"] = dataset_df.apply(cls.__legend_name, axis=1,)
        experiment_grouped = dataset_df.groupby(
            list(index_cols) + ["experiment"]
        )

        metric_mean_df = experiment_grouped[mean_metric_col].mean().unstack()

        std_metric_col = "{} | std".format(metric_name)
        if std_metric_col not in dataset_df.columns:
            metric_std_df = None
        else:
            metric_std_df = experiment_grouped[std_metric_col].mean().unstack()

        if num_subject_cols not in dataset_df.columns:
            num_subjects = None
        else:
            num_subjects = (
                experiment_grouped[num_subject_cols]
                .mean()
                .reset_index()
                .drop("experiment", axis=1)
                .groupby(list(index_cols))
                .mean()
                .astype(int)
            )

        return MetricTable(
            metric_name, metric_mean_df, metric_std_df, num_subjects, group,
        )

    @classmethod
    def prepare_metric_tables_per_group(
        cls,
        grouped_df: pd.DataFrame,
        metric_name: str = "accuracy",
        grouping_cols: Sequence[str] = ("dataset",),
        index_cols: Sequence[str] = ("labeled_sites",),
        num_subject_cols: str = "num_subjects",
    ) -> Sequence[MetricTable]:
        results = list()
        for group, grouped_df in grouped_df.groupby(list(grouping_cols)):
            group_metric_table = cls.create_metric_table(
                grouped_df,
                metric_name,
                index_cols,
                num_subject_cols,
                dict(zip(grouping_cols, group))
                if len(grouping_cols) > 1
                else {grouping_cols[0]: group},
            )
            results.append(group_metric_table)
        return results

    @classmethod
    def __ttest_compare_cols(self, df: pd.DataFrame):
        return [
            col
            for col in (
                "model_name",
                "model_params",
                "optim_params",
                "hyperparameters",
                "ssl",
                "harmonize",
            )
            if col in df.columns
        ]

    @classmethod
    def prepare_ttest_table(
        cls,
        grouped_df: pd.DataFrame,
        metric_name: str = "accuracy",
        group: Dict[str, Any] = dict(),
    ) -> TTestTable:
        metric_list_col = "{}_list".format(metric_name)
        if metric_list_col not in grouped_df:
            raise KeyError('column "{}" not found'.format(metric_list_col))

        compare_cols = cls.__ttest_compare_cols(grouped_df)
        multi_index = pd.MultiIndex.from_frame(grouped_df[compare_cols])

        metric_list = grouped_df[metric_list_col].tolist()
        n = len(metric_list)
        ttest_matrix = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # row > column
                ttest = ttest_ind(
                    metric_list[i], metric_list[j], equal_var=False,
                )
                ttest_matrix[i, j] = ttest.pvalue
                ttest_matrix[j, i] = ttest.pvalue

        ttest_df = pd.DataFrame(
            ttest_matrix, index=multi_index, columns=multi_index
        )

        metric_mean_col = "{} | mean".format(metric_name)
        if not metric_mean_col in grouped_df.columns:
            metric_mean_col = metric_name
            if not metric_mean_col in grouped_df.columns:
                metric_mean_col = metric_list_col

        if metric_mean_col == metric_list_col:
            ttest_df["{} | mean".format(metric_name)] = (
                grouped_df[metric_list_col].apply(np.mean).values
            )
        else:
            ttest_df["{} | mean".format(metric_name)] = grouped_df[
                metric_mean_col
            ].values

        return TTestTable(metric_name, grouped_df, ttest_df, group)

    @classmethod
    def prepare_ttest_table_per_group(
        cls,
        grouped_df: pd.DataFrame,
        metric_name: str = "accuracy",
        grouping_cols: Sequence[str] = ("dataset", "labeled_sites",),
    ) -> Sequence[TTestTable]:
        for col in grouping_cols:
            if col not in grouped_df.columns:
                raise KeyError('column "{}" not found'.format(col))

        results = list()
        for group, new_grouped_df in grouped_df.groupby(
            list(grouping_cols), dropna=False
        ):
            ttest_table = cls.prepare_ttest_table(
                new_grouped_df,
                metric_name,
                dict(zip(grouping_cols, group))
                if len(grouping_cols) > 1
                else {grouping_cols[0]: group},
            )
            results.append(ttest_table)
        return results

    @classmethod
    def merge_ttest_tables(
        cls, grouped_ttest_table: Sequence[TTestTable]
    ) -> Dict[str, pd.DataFrame]:
        all_metrics = set(map(lambda x: x.metric_name, grouped_ttest_table))
        all_tables = dict()

        for metric in all_metrics:
            metric_ttest_tables = list()
            for ttest_table in filter(
                lambda x: x.metric_name == metric, grouped_ttest_table
            ):
                ttest_df = ttest_table.ttest_table
                group = ttest_table.group

                index = ttest_df.index.to_frame()
                for k, v in sorted(group.items(), reverse=True):
                    index.insert(0, k, v)
                ttest_df.index = pd.MultiIndex.from_frame(index)

                metric_ttest_tables.append(ttest_df)

            if not metric_ttest_tables:
                continue
            if len(metric_ttest_tables) == 1:
                all_tables[metric] = metric_ttest_tables[0]
            else:
                all_tables[metric] = pd.concat(metric_ttest_tables, axis=0)

        return all_tables
