import pandas as pd
import numpy as np
from typing import Optional, Sequence
from functools import reduce


class GroupSeedFold:
    def __init__(
        self,
        df: pd.DataFrame,
        std_cols: Sequence[str] = ("accuracy",),
        accumulate_metrics: Sequence[str] = ("accuracy",),
    ):
        self.df = df
        self.std_cols = std_cols
        self.accumulate_metrics = accumulate_metrics

    @property
    def group_overall_cols(self):
        return [
            col
            for col in (
                "dataset",
                "model_name",
                "model_params",
                "optim_params",
                "hyperparameters",
                "ssl",
                "harmonize",
            )
            if col in self.df.columns
        ]

    @property
    def group_fold_cols(self):
        return [
            col
            for col in (
                "dataset",
                "labeled_sites", 
                # "unlabeled_sites",
                "model_name",
                "model_params",
                "optim_params",
                "hyperparameters",
                "ssl",
                "harmonize",
            )
            if col in self.df.columns
        ]

    @property
    def group_seed_cols(self):
        return self.group_fold_cols + ["seed"]

    @property
    def metrics(self):
        return [
            col
            for col in (
                "accuracy",
                "sensitivity",
                "specificity",
                "f1",
                "precision",
            )
            if col in self.df.columns
        ]

    @property
    def losses(self):
        return [col for col in self.df.columns if "loss" in col]

    @property
    def epochs(self):
        return [
            col for col in self.df.columns if "epoch" in col and "best" in col
        ]

    @property
    def remaining_cols(self):
        return [
            *[
                col
                for col in (
                    "model_size",
                    "num_labeled_train",
                    "num_unlabeled_train",
                    "num_valid",
                    "baseline_accuracy",
                )
                if col in self.df.columns
            ],
            *self.metrics,
            *self.losses,
            *self.epochs,
            *[col for col in ("time_taken",) if col in self.df.columns],
        ]

    def _group_seed(self):
        group_seed_cols = self.group_seed_cols
        all = self.remaining_cols
        grouped = self.df.groupby(group_seed_cols, dropna=False)
        grouped_mean = grouped[all].mean()

        for metric in self.accumulate_metrics:
            if metric not in self.df.columns:
                continue
            grouped_acc_list = grouped.agg({metric: list})
            grouped_mean["{}_list".format(metric)] = grouped_acc_list

        return grouped_mean.reset_index()

    def group_seed_fold(self) -> pd.DataFrame:
        grouped_mean = self._group_seed()
        group_fold_cols = self.group_fold_cols

        agg_dict = dict((col, np.mean) for col in self.remaining_cols)
        for col in self.std_cols:
            if col not in grouped_mean.columns:
                continue
            agg_dict[col] = [np.mean, np.std]
        agg_dict["seed"] = len

        grouped_folds = grouped_mean.groupby(group_fold_cols, dropna=False)

        result = grouped_folds.agg(agg_dict)
        metric_losses_epoch = self.metrics + self.losses + self.epochs
        result.columns = [
            " | ".join(col) if col[0] in metric_losses_epoch else col[0]
            for col in result.columns
        ]
        result = result.rename(columns={"seed": "num_seed"})

        if (
            "num_labeled_train" in result.columns
            and "num_valid" in result.columns
        ):
            result["num_subjects"] = (
                result["num_labeled_train"] + result["num_valid"]
            ).astype(int)

        for metric in self.accumulate_metrics:
            col_name = "{}_list".format(metric)
            if col_name not in grouped_mean:
                continue
            grouped_acc_list = grouped_folds.agg(
                {col_name: lambda acc_ls: reduce(lambda x, y: x + y, acc_ls)}
            )
            result[col_name] = grouped_acc_list

        return result.reset_index()

    def group_overall_dataset(
        self, grouped_seed_fold: Optional[pd.DataFrame] = None
    ):
        if grouped_seed_fold is None:
            grouped_seed_fold = self.group_seed_fold()

        group_cols = self.group_overall_cols
        ignored_cols = list(set(self.group_fold_cols) - set(group_cols))
        remaining_cols = [
            col
            for col in grouped_seed_fold
            if col not in group_cols + ignored_cols
        ]

        grouped_overall = grouped_seed_fold.groupby(group_cols, dropna=False)

        result = grouped_overall[remaining_cols].mean()
        for metric in self.accumulate_metrics:
            col_name = "{}_list".format(metric)
            if col_name not in grouped_seed_fold:
                continue
            grouped_acc_list = grouped_overall.agg(
                {col_name: lambda acc_ls: reduce(lambda x, y: x + y, acc_ls)}
            )
            result[col_name] = grouped_acc_list

        return result.reset_index()

