from abc import ABC, abstractmethod
from itertools import product
from pickle import TRUE
from typing import Any, Dict, Generator, List, Sequence, Tuple

import copy
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from config import TuningStrategy as TS


class TuningBase(ABC):
    def __init__(
        self,
        params: Dict[str, Sequence[Any]],
        metric: str,
        direction: TS.Direction,
        seed_fold_list: Sequence[Tuple[int, int]],
        **kwargs
    ):
        self.params = params
        self.metric = metric
        self.direction = direction
        self.seed_fold_list = set(seed_fold_list)

        self.__valid = True
        self.__updated_model_size = False
        self.__param_list: List[Dict[str, Any]] = list()
        self.__result_list: List[Dict[Tuple[int, int], float]] = list()

    @property
    def best_param(self):
        if not self.__valid:
            raise Exception(
                "finish update current param result first "
                "before getting the best param"
            )
        cv_results = self.cv_results
        idx = np.argmin(cv_results["rank"])
        return copy.deepcopy(self.__param_list[idx])

    @property
    def cv_results(self):
        if not self.__valid:
            raise Exception(
                "finish update current param result first "
                "before getting cross validation results"
            )
        params_df = pd.DataFrame(self.__param_list)
        results_df = pd.DataFrame(self.__result_list)
        model_size = params_df["model_size"]
        mean = results_df.mean(axis=1)
        std = results_df.std(axis=1)
        rank_criterion = pd.concat(
            [
                mean if self.direction == TS.Direction.MINIMUM else -mean,
                model_size,
                std,
            ],
            axis=1,
        )
        rank = (
            rank_criterion.apply(tuple, axis=1)
            .rank(method="dense", ascending=True)
            .astype(int)
        )

        cv_results = pd.concat(
            [params_df, results_df.add_suffix("_{}".format(self.metric))],
            axis=1,
        )
        cv_results["mean_{}".format(self.metric)] = mean
        cv_results["std_{}".format(self.metric)] = std
        cv_results["rank"] = rank
        return cv_results

    def _get_last_result(self) -> Dict[str, float]:
        return copy.deepcopy(self.__result_list[-1])

    @abstractmethod
    def _generate_sample(self) -> Generator[Dict[str, Any], None, None]:
        """
        A generator object
        """
        raise NotImplementedError

    def next_sample(self) -> Generator[Dict[str, Any], None, None]:
        generator = self._generate_sample()
        while True:
            if not self.__valid:
                raise Exception(
                    "finish update current param result first "
                    "before getting the next sample"
                )
            try:
                new_sample: Dict[str, Any] = next(generator)
                self.__param_list.append(copy.deepcopy(new_sample))
                self.__result_list.append(dict())
                self.__valid = False
                self.__updated_model_size = False
                yield new_sample
            except StopIteration as e:
                break

    def update_param_model_size(self, model_size: int):
        if self.__updated_model_size or np.isnan(model_size):
            return
        self.__param_list[-1]["model_size"] = model_size
        self.__updated_model_size = True

    def update_param_result(self, seed: int, fold: int, metric_value: float):
        self.__result_list[-1][seed, fold] = metric_value
        if not (self.seed_fold_list - set(self.__result_list[-1])):
            self.__valid = True


class GridSearch(TuningBase):
    def _generate_sample(self) -> Generator[Dict[str, Any], None, None]:
        keys, values = list(zip(*self.params.items()))
        for value in product(*values):
            yield dict(zip(keys, value))


class BayesianGridSearch(GridSearch):
    def __init__(
        self,
        params: Dict[str, Any],
        metric: str,
        direction: TS.Direction,
        seed_fold_list: Sequence[Tuple[int, int]],
        num_samples_init: int = 10,
        num_samples_total: int = 50,
        **kwargs
    ):
        super().__init__(params, metric, direction, seed_fold_list)
        self.__init_all_combinations()
        self.num_samples_init = num_samples_init
        self.num_samples_total = min(
            num_samples_total, len(self.__all_param_combinations)
        )

    def __init_all_combinations(self):
        keys, values = list(zip(*self.params.items()))
        self.__all_param_combinations = [
            dict(zip(keys, value)) for value in product(*values)
        ]

        df = pd.DataFrame(self.__all_param_combinations)
        numerical = df.select_dtypes(include=np.number)
        categorical = df.select_dtypes(exclude=np.number)
        if categorical.empty:
            self.__X = numerical.values
        else:
            self.__X = pd.concat(
                [numerical, pd.get_dummies(categorical)], axis=1
            ).values

        self.__idx_is_sampled = np.zeros(
            len(self.__all_param_combinations), dtype=bool
        )
        self.__real_scores = np.full(
            len(self.__all_param_combinations),
            0.0 if self.direction == TS.Direction.MAXIMUM else np.inf,
        )
        self.__gp_model = GaussianProcessRegressor()

    def acquisition(self):
        self.__gp_model.fit(
            self.__X[self.__idx_is_sampled & np.isfinite(self.__real_scores)],
            self.__real_scores[
                self.__idx_is_sampled & np.isfinite(self.__real_scores)
            ],
        )
        sampled_scores = self.__gp_model.predict(
            self.__X[self.__idx_is_sampled]
        )
        if self.direction == TS.Direction.MAXIMUM:
            best_score = sampled_scores.max()
        else:
            best_score = sampled_scores.min()
        pred_scores, pred_std = self.__gp_model.predict(
            self.__X[~self.__idx_is_sampled], return_std=TRUE
        )
        probabilities = norm.cdf((pred_scores - best_score) / (pred_std + 1e-9))
        idx = np.argmax(probabilities)
        return np.argwhere(~self.__idx_is_sampled).flatten()[idx]

    def _generate_sample(self) -> Generator[Dict[str, Any], None, None]:
        if len(self.__all_param_combinations) <= self.num_samples_init:
            return super()._generate_sample()

        init_indices = np.random.choice(
            len(self.__all_param_combinations),
            self.num_samples_init,
            replace=False,
        )
        for idx in init_indices:
            self.__idx_is_sampled[idx] = True
            yield self.__all_param_combinations[idx]
            self.__real_scores[idx] = np.mean(
                list(self._get_last_result().values())
            )

        for _ in range(self.num_samples_init, self.num_samples_total):
            idx = self.acquisition()
            self.__idx_is_sampled[idx] = True
            yield self.__all_param_combinations[idx]
            self.__real_scores[idx] = np.nanmean(
                list(self._get_last_result().values())
            )


def load_tuning_object(
    strategy: TS,
    params: Dict[str, Sequence[Any]],
    seed_fold_list: Sequence[Tuple[int, int]],
) -> TuningBase:
    if strategy.sampling_method == TS.SamplingMethod.GRID.value:
        return GridSearch(
            params,
            strategy.metric,
            TS.Direction(strategy.direction),
            seed_fold_list,
            **strategy.sampling_kwargs
        )
    elif strategy.sampling_method == TS.SamplingMethod.BAYES_GRID.value:
        return BayesianGridSearch(
            params,
            strategy.metric,
            TS.Direction(strategy.direction),
            seed_fold_list,
            **strategy.sampling_kwargs
        )
    else:
        raise NotImplementedError

