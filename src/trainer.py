from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

import os
import json
import time
import copy
import logging
import numpy as np
import torch
from torch_geometric.data import Data

from data import DataloaderBase, Dataset
from factory import DoubleStageFrameworkFactory, SingleStageFrameworkFactory
from models import count_parameters
from utils import get_device, get_pbar, mkdir, seed_torch


@dataclass(frozen=True)
class TrainerParams:
    output_directory: str
    model_name: str
    model_params: Dict[str, Any]
    optim_params: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    dataset: Dataset
    seed: int
    fold: int
    ssl: bool
    harmonize: bool
    validation: bool
    labeled_sites: Optional[Union[str, Sequence[str]]] = field(default=None)
    unlabeled_sites: Optional[Union[str, Sequence[str]]] = field(default=None)
    num_unlabeled: Optional[int] = field(default=None)
    device: int = field(default=-1)
    verbose: bool = field(default=False)
    patience: int = field(default=np.inf)
    max_epoch: int = field(default=1000)
    save_model: bool = field(default=False)
    dataloader_num_process: int = 1
    time_id: bool = field(init=False, default_factory=lambda: int(time.time()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_params": str(self.model_params),
            "optim_params": str(self.optim_params),
            "hyperparameters": str(self.hyperparameters),
            "dataset": self.dataset.value,
            "seed": self.seed,
            "fold": self.fold,
            "ssl": self.ssl,
            "harmonize": self.harmonize,
            "validation": self.validation,
            "labeled_sites": self.labeled_sites,
            "unlabeled_sites": self.unlabeled_sites,
            "device": self.device,
            "epochs_log_path": self.epochs_log_path,
        }

    @property
    def model_path(self):
        return os.path.join(
            os.path.abspath(self.output_directory),
            "models",
            "{}_{}_{}_{}_{}.pt".format(
                self.dataset.value,
                self.model_name,
                self.seed,
                self.fold,
                self.time_id,
            ),
        )

    @property
    def epochs_log_path(self):
        return os.path.join(
            os.path.abspath(self.output_directory),
            "epochs_log",
            "{}_{}_{}_{}_{}.log".format(
                self.dataset.value,
                self.model_name,
                self.seed,
                self.fold,
                self.time_id,
            ),
        )


@dataclass(frozen=True)
class TrainerResults:
    trainer_params: TrainerParams
    num_labeled_train: int
    num_unlabeled_train: int
    num_valid: int
    baseline_accuracy: float
    best_metrics: Dict[str, float]
    best_epoch: int
    time_taken: int
    model_size: int
    model_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.trainer_params.to_dict(),
            "num_labeled_train": self.num_labeled_train,
            "num_unlabeled_train": self.num_unlabeled_train,
            "num_valid": self.num_valid,
            "baseline_accuracy": self.baseline_accuracy,
            **self.best_metrics,
            "best_epoch": self.best_epoch,
            "time_taken": self.time_taken,
            "model_size": self.model_size,
            "model_path": self.model_path,
        }


class Trainer(ABC):
    def __init__(
        self, dataloader: DataloaderBase, trainer_params: TrainerParams,
    ):
        super().__init__()
        if dataloader.dataset != trainer_params.dataset:
            raise Exception(
                "dataloader.dataset != trainer_params.dataset, {} != {}".format(
                    dataloader.dataset.value, trainer_params.dataset.value
                )
            )
        if dataloader.harmonize != trainer_params.harmonize:
            raise Exception(
                "dataloader.harmonize != trainer_params.harmonize, {} != {}".format(
                    dataloader.harmonize, trainer_params.harmonize
                )
            )
        self.dataloader = dataloader
        self.trainer_params = trainer_params
        self.__called = False

    def _set_called(self):
        if self.__called:
            raise Exception("Trainer.run() can only be called once")
        self.__called = True

    @staticmethod
    def verbose_info(train_metrics: dict, valid_metrics: dict) -> str:
        all_metrics = []
        for k, v in train_metrics.items():
            all_metrics.append("train_{}: {:.4f}".format(k, v))
        for k, v in valid_metrics.items():
            all_metrics.append("valid_{}: {:.4f}".format(k, v))
        return " ".join(all_metrics)

    @staticmethod
    def _get_baseline_accuracy(data: Union[Data, Sequence[Data]]) -> float:
        if not isinstance(data, Data):
            y = torch.cat([d.y for d in data], dim=0)
        else:
            y = data.y
        _, counts = y.unique(return_counts=True)
        return (counts.max() / y.size(0)).item()

    @abstractmethod
    def run(self):
        raise NotImplementedError


class SingleStageFrameworkTrainer(Trainer):
    def run(self) -> TrainerResults:
        self._set_called()

        seed_torch()
        device = get_device(self.trainer_params.device)
        verbose = self.trainer_params.verbose

        start = time.time()
        data_dict = self.dataloader.load_split_data(
            seed=self.trainer_params.seed,
            fold=self.trainer_params.fold,
            ssl=self.trainer_params.ssl,
            validation=self.trainer_params.validation,
            labeled_sites=self.trainer_params.labeled_sites,
            unlabeled_sites=self.trainer_params.unlabeled_sites,
            num_process=self.trainer_params.dataloader_num_process,
        )

        num_labeled_train = data_dict.get("num_labeled_train", 0)
        num_unlabeled_train = data_dict.get("num_unlabeled_train", 0)
        if self.trainer_params.validation:
            num_valid = data_dict.get("num_valid", 0)
            baseline_accuracy = self._get_baseline_accuracy(
                data_dict.get("valid")
            )
        else:
            num_valid = data_dict.get("num_test", 0)
            baseline_accuracy = self._get_baseline_accuracy(
                data_dict.get("test")
            )

        self.trainer_params.model_params["input_size"] = data_dict["input_size"]
        self.trainer_params.model_params["num_sites"] = data_dict["num_sites"]
        model = SingleStageFrameworkFactory.load_model(
            self.trainer_params.model_name, self.trainer_params.model_params
        )
        model_size = count_parameters(model)
        optimizer = model.get_optimizer(self.trainer_params.optim_params)

        patience = self.trainer_params.patience
        cur_patience = 0
        max_epoch = self.trainer_params.max_epoch
        best_epoch = 0
        best_metrics = {
            "ce_loss": np.inf,
            "accuracy": 0,
        }
        save_model = self.trainer_params.save_model
        best_model_state_dict = None

        epochs_log_path = self.trainer_params.epochs_log_path
        mkdir(os.path.dirname(epochs_log_path))
        with open(epochs_log_path, "w") as f:
            f.write("")

        pbar = get_pbar(max_epoch, verbose)
        for epoch in pbar:
            try:
                train_metrics = model.train_step(
                    device,
                    data_dict.get("labeled_train", None),
                    data_dict.get("unlabeled_train", None),
                    optimizer,
                    self.trainer_params.hyperparameters,
                )
                if self.trainer_params.validation:
                    valid_metrics = model.test_step(
                        device, data_dict.get("valid", None)
                    )
                else:
                    valid_metrics = model.test_step(
                        device, data_dict.get("test", None)
                    )
            except Exception as e:
                logging.error(e)
            with open(epochs_log_path, "a") as f:
                f.write(
                    json.dumps(
                        {"train": train_metrics, "valid": valid_metrics},
                        sort_keys=True,
                    )
                    + "\n"
                )

            """
            save priority:
            1. accuracy
            2. ce_loss
            """
            save = valid_metrics["accuracy"] > best_metrics["accuracy"] or (
                valid_metrics["accuracy"] == best_metrics["accuracy"]
                and valid_metrics["ce_loss"] < best_metrics["ce_loss"]
            )
            if save:
                best_epoch = epoch
                best_metrics = valid_metrics.copy()
                if save_model:
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                cur_patience = 0
            else:
                cur_patience += 1

            if verbose:
                pbar.set_postfix_str(
                    self.verbose_info(train_metrics, valid_metrics)
                )
            if cur_patience == patience:
                break

        if save_model and best_model_state_dict is not None:
            try:
                model_path = self.trainer_params.model_path
                mkdir(os.path.dirname(model_path))
                torch.save(best_model_state_dict, model_path)
            except Exception as e:
                logging.error(str(e))
                model_path = None
        else:
            model_path = None

        end = time.time()
        return TrainerResults(
            trainer_params=self.trainer_params,
            num_labeled_train=num_labeled_train,
            num_unlabeled_train=num_unlabeled_train,
            num_valid=num_valid,
            baseline_accuracy=baseline_accuracy,
            best_metrics=best_metrics,
            best_epoch=best_epoch,
            time_taken=end - start,
            model_size=model_size,
            model_path=model_path,
        )


@dataclass(frozen=True)
class DoubleTrainerParams(TrainerParams):
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        del result["epochs_log_path"]
        result["ae_epochs_log_path"] = self.ae_epochs_log_path
        result["fcnn_epochs_log_path"] = self.fcnn_epochs_log_path
        return result

    @property
    def ae_epochs_log_path(self):
        return os.path.join(
            os.path.abspath(self.output_directory),
            "epochs_log",
            "{}_{}_AE_{}_{}_{}.log".format(
                self.dataset.value,
                self.model_name,
                self.seed,
                self.fold,
                self.time_id,
            ),
        )

    @property
    def fcnn_epochs_log_path(self):
        return os.path.join(
            os.path.abspath(self.output_directory),
            "epochs_log",
            "{}_{}_FCNN_{}_{}_{}.log".format(
                self.dataset.value,
                self.model_name,
                self.seed,
                self.fold,
                self.time_id,
            ),
        )


@dataclass(frozen=True)
class DoubleTrainerResults:
    trainer_params: TrainerParams
    num_labeled_train: int
    num_unlabeled_train: int
    num_valid: int
    baseline_accuracy: float
    best_ae_metrics: Dict[str, float]
    best_fcnn_metrics: Dict[str, float]
    best_ae_epoch: int
    best_fcnn_epoch: int
    time_taken: int
    model_size: int
    model_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.trainer_params.to_dict(),
            "num_labeled_train": self.num_labeled_train,
            "num_unlabeled_train": self.num_unlabeled_train,
            "num_valid": self.num_valid,
            "baseline_accuracy": self.baseline_accuracy,
            **self.best_ae_metrics,
            "best_ae_epoch": self.best_ae_epoch,
            **self.best_fcnn_metrics,
            "best_fcnn_epoch": self.best_fcnn_epoch,
            "time_taken": self.time_taken,
            "model_size": self.model_size,
            "model_path": self.model_path,
        }


class DoubleStageFrameworkTrainer(Trainer):
    def run(self) -> TrainerResults:
        self._set_called()

        seed_torch()
        device = get_device(self.trainer_params.device)
        verbose = self.trainer_params.verbose

        start = time.time()
        data_dict = self.dataloader.load_split_data(
            seed=self.trainer_params.seed,
            fold=self.trainer_params.fold,
            ssl=self.trainer_params.ssl,
            validation=self.trainer_params.validation,
            labeled_sites=self.trainer_params.labeled_sites,
            unlabeled_sites=self.trainer_params.unlabeled_sites,
            num_process=self.trainer_params.dataloader_num_process,
        )

        num_labeled_train = data_dict.get("num_labeled_train", 0)
        num_unlabeled_train = data_dict.get("num_unlabeled_train", 0)
        if self.trainer_params.validation:
            num_valid = data_dict.get("num_valid", 0)
            baseline_accuracy = self._get_baseline_accuracy(
                data_dict.get("valid")
            )
        else:
            num_valid = data_dict.get("num_test", 0)
            baseline_accuracy = self._get_baseline_accuracy(
                data_dict.get("test")
            )

        self.trainer_params.model_params["ae_param"]["input_size"] = data_dict[
            "input_size"
        ]
        self.trainer_params.model_params["ae_param"]["num_sites"] = data_dict[
            "num_sites"
        ]
        self.trainer_params.model_params["fcnn_param"]["num_sites"] = data_dict[
            "num_sites"
        ]
        ae_model, fcnn_model = DoubleStageFrameworkFactory.load_model(
            self.trainer_params.model_name, self.trainer_params.model_params
        )
        ae_optim = ae_model.get_optimizer(
            self.trainer_params.optim_params.get("ae_param", dict())
        )
        fcnn_optim = fcnn_model.get_optimizer(
            self.trainer_params.optim_params.get("fcnn_param", dict())
        )

        patience = self.trainer_params.patience
        cur_patience = 0
        max_epoch = self.trainer_params.max_epoch
        best_ae_epoch = 0
        best_ae_metrics = {
            "rc_loss": np.inf,
        }
        best_ae_model_state_dict = None

        ae_epochs_log_path = self.trainer_params.ae_epochs_log_path
        mkdir(os.path.dirname(ae_epochs_log_path))
        with open(ae_epochs_log_path, "w") as f:
            f.write("")

        pbar = get_pbar(max_epoch, verbose)
        for epoch in pbar:
            try:
                train_metrics = ae_model.train_step(
                    device,
                    data_dict.get("labeled_train", None),
                    data_dict.get("unlabeled_train", None),
                    ae_optim,
                    self.trainer_params.hyperparameters.get("ae_param", dict()),
                )
                if self.trainer_params.validation:
                    valid_metrics = ae_model.test_step(
                        device, data_dict.get("valid", None)
                    )
                else:
                    valid_metrics = ae_model.test_step(
                        device, data_dict.get("test", None)
                    )
            except Exception as e:
                logging.error(e)
            with open(ae_epochs_log_path, "a") as f:
                f.write(
                    json.dumps(
                        {"train": train_metrics, "valid": valid_metrics},
                        sort_keys=True,
                    )
                    + "\n"
                )

            """
            save priority:
            1. rc_loss
            """
            save = valid_metrics["rc_loss"] < best_ae_metrics["rc_loss"]
            if save:
                best_ae_epoch = epoch
                best_ae_metrics = valid_metrics.copy()
                best_ae_model_state_dict = copy.deepcopy(ae_model.state_dict())
                cur_patience = 0
            else:
                cur_patience += 1

            if verbose:
                pbar.set_postfix_str(
                    self.verbose_info(train_metrics, valid_metrics)
                )
            if cur_patience == patience:
                break

        ae_model.load_state_dict(best_ae_model_state_dict)
        new_data_dict = dict()
        for key in ["labeled_train", "unlabeled_train", "valid", "test"]:
            if key in data_dict:
                new_data_dict[key] = ae_model.prepare_z_y(
                    device, data_dict[key]
                )

        patience = self.trainer_params.patience
        cur_patience = 0
        max_epoch = self.trainer_params.max_epoch
        best_fcnn_epoch = 0
        best_fcnn_metrics = {
            "ce_loss": np.inf,
            "accuracy": 0,
        }
        best_fcnn_model_state_dict = None

        fcnn_epochs_log_path = self.trainer_params.fcnn_epochs_log_path
        mkdir(os.path.dirname(fcnn_epochs_log_path))
        with open(fcnn_epochs_log_path, "w") as f:
            f.write("")

        pbar = get_pbar(max_epoch, verbose)
        for epoch in pbar:
            try:
                train_metrics = fcnn_model.train_step(
                    device,
                    new_data_dict.get("labeled_train", None),
                    new_data_dict.get("unlabeled_train", None),
                    fcnn_optim,
                    self.trainer_params.hyperparameters.get(
                        "fcnn_param", dict()
                    ),
                )
                if self.trainer_params.validation:
                    valid_metrics = fcnn_model.test_step(
                        device, new_data_dict.get("valid", None)
                    )
                else:
                    valid_metrics = fcnn_model.test_step(
                        device, new_data_dict.get("test", None)
                    )
            except:
                continue
            with open(fcnn_epochs_log_path, "a") as f:
                f.write(
                    json.dumps(
                        {"train": train_metrics, "valid": valid_metrics},
                        sort_keys=True,
                    )
                    + "\n"
                )

            """
            save priority:
            1. rc_loss
            """
            save = valid_metrics["accuracy"] > best_fcnn_metrics[
                "accuracy"
            ] or (
                valid_metrics["accuracy"] == best_fcnn_metrics["accuracy"]
                and valid_metrics["ce_loss"] < best_fcnn_metrics["ce_loss"]
            )
            if save:
                best_fcnn_epoch = epoch
                best_fcnn_metrics = valid_metrics.copy()
                best_fcnn_model_state_dict = copy.deepcopy(
                    fcnn_model.state_dict()
                )
                cur_patience = 0
            else:
                cur_patience += 1

            if verbose:
                pbar.set_postfix_str(
                    self.verbose_info(train_metrics, valid_metrics)
                )
            if cur_patience == patience:
                break

        fcnn_model.load_state_dict(best_fcnn_model_state_dict)

        best_model = DoubleStageFrameworkFactory.compile_model(
            self.trainer_params.model_name, ae_model, fcnn_model
        )
        model_size = count_parameters(best_model)

        if self.trainer_params.save_model:
            try:
                best_model_state_dict = best_model.state_dict()
                model_path = self.trainer_params.model_path
                mkdir(os.path.dirname(model_path))
                torch.save(best_model_state_dict, model_path)
            except Exception as e:
                logging.error(str(e))
                model_path = None
        else:
            model_path = None

        end = time.time()
        return DoubleTrainerResults(
            trainer_params=self.trainer_params,
            num_labeled_train=num_labeled_train,
            num_unlabeled_train=num_unlabeled_train,
            num_valid=num_valid,
            baseline_accuracy=baseline_accuracy,
            best_ae_metrics=best_ae_metrics,
            best_fcnn_metrics=best_fcnn_metrics,
            best_ae_epoch=best_ae_epoch,
            best_fcnn_epoch=best_fcnn_epoch,
            time_taken=end - start,
            model_size=model_size,
            model_path=model_path,
        )
