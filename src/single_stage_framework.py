from __future__ import annotations

import os
import json
import time
import yaml
import logging
import argparse
import pandas as pd
from itertools import product
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

from data import Dataset
from config import EXPERIMENT_DIR
from utils import mkdir, on_error
from factory import SingleStageFrameworkFactory
from trainer import SingleStageFrameworkTrainer, TrainerParams


@dataclass(frozen=True)
class RangeGenerator:
    min: int
    max: int

    def generate(self):
        return list(range(self.min, self.max))

    @staticmethod
    def parse(range_cfg: Dict[str, Any]) -> RangeGenerator:
        return RangeGenerator(**range_cfg)


@dataclass(frozen=True)
class ModelConfig:
    all_models: Sequence[single_model]

    @dataclass(frozen=True)
    class single_model:
        model_name: str
        model_params: Dict[str, Any]
        optim_params: Dict[str, Any] = field(default_factory=dict)
        hyperparameters: Dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> Dict[str, Any]:
            return dict(
                model_name=self.model_name,
                model_params=self.model_params,
                optim_params=self.optim_params,
                hyperparameters=self.hyperparameters,
            )

    def generate(self):
        return [cfg.to_dict() for cfg in self.all_models]

    @staticmethod
    def parse(model_configs: Sequence[Dict[str, Any]]) -> ModelConfig:
        all_models = [ModelConfig.single_model(**cfg) for cfg in model_configs]
        return ModelConfig(all_models)


@dataclass(frozen=True)
class DataConfig:
    all_data: Sequence[single_data]

    @dataclass(frozen=True)
    class single_data:
        dataset: str
        labeled_sites: Sequence[Optional[Union[str, Sequence[str]]]]
        output_directory: Optional[str] = field(default=None)

    def generate(self):
        return [
            dict(
                dataset=cfg.dataset,
                labeled_sites=sites,
                output_directory=cfg.output_directory,
            )
            for cfg in self.all_data
            for sites in cfg.labeled_sites
        ]

    @staticmethod
    def parse(data_configs: Sequence[Dict[str, Any]]) -> DataConfig:
        return DataConfig(
            [DataConfig.single_data(**cfg) for cfg in data_configs]
        )


@dataclass(frozen=True)
class ExperimentSettings:
    all_settings: Sequence[single_setting]

    @dataclass(frozen=True)
    class single_setting:
        ssl: bool = field(default=False)
        harmonize: bool = field(default=False)
        validation: bool = field(default=False)

        def to_dict(self) -> Dict[str, bool]:
            return dict(
                ssl=self.ssl,
                harmonize=self.harmonize,
                validation=self.validation,
            )

    def generate(self):
        return [cfg.to_dict() for cfg in self.all_settings]

    @staticmethod
    def parse(exp_settings: Sequence[Dict[str, bool]]) -> ExperimentSettings:
        return ExperimentSettings(
            [ExperimentSettings.single_setting(**cfg) for cfg in exp_settings]
        )


@dataclass(frozen=True)
class ProcessConfig:
    device: int = field(default=-1)
    verbose: bool = field(default=0)
    save_model_condition: Sequence[Dict[str, Any]] = field(default_factory=list)

    def match_save_model_condition(self, config: Dict[str, Any]):
        if not self.save_model_condition:
            return True
        for condition in self.save_model_condition:
            matched = True
            for key, value in condition.items():
                if key not in config:
                    matched = False
                elif value != config[key]:
                    matched = False
                if not matched:
                    break
            if matched:
                return True
        return False

    def update(self, config: Dict[str, Any]):
        config["device"] = self.device
        config["verbose"] = self.verbose
        config["save_model"] = self.match_save_model_condition(config)
        return config


@dataclass(frozen=True)
class ConfigParser:
    seed: RangeGenerator
    fold: RangeGenerator
    model: ModelConfig
    data: DataConfig
    experiment_settings: ExperimentSettings
    process: ProcessConfig

    def generate(self):
        for model, data, exp_setting in product(
            self.model.generate(),
            self.data.generate(),
            self.experiment_settings.generate(),
        ):
            config = {
                "seed": self.seed.generate(),
                "fold": self.fold.generate(),
                **model,
                **data,
                **exp_setting,
            }
            config = self.process.update(config)
            yield config

    @staticmethod
    def parse(
        seed: Dict[str, int],
        fold: Dict[str, Any],
        model: Sequence[Dict[str, Any]],
        data: Sequence[Dict[str, Any]],
        experiment_settings: Sequence[Dict[str, bool]],
        process: Dict[str, Any],
    ) -> ConfigParser:
        return ConfigParser(
            seed=RangeGenerator.parse(seed),
            fold=RangeGenerator.parse(fold),
            model=ModelConfig.parse(model),
            data=DataConfig.parse(data),
            experiment_settings=ExperimentSettings.parse(experiment_settings),
            process=ProcessConfig(**process),
        )


@on_error(dict(), True)
def experiment(trainer: SingleStageFrameworkTrainer):
    trainer_results = trainer.run()
    return trainer_results.to_dict()


def process(config: Dict[str, Any]):
    logging.info("CONFIG:\n{}".format(json.dumps(config, indent=4)))

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    experiment_name = "{}_{}_{}".format(
        script_name, int(time.time()), os.getpid()
    )

    output_dir = (
        config.get("output_directory", EXPERIMENT_DIR) or EXPERIMENT_DIR
    )
    output_dir = os.path.abspath(os.path.join(output_dir, experiment_name))

    config_path = os.path.join(
        output_dir, "{}.config.json".format(experiment_name),
    )
    mkdir(output_dir)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

    dataloader = SingleStageFrameworkFactory.load_dataloader(
        model_name=config["model_name"],
        dataloader_param={
            "dataset": Dataset(config["dataset"]),
            "harmonize": config["harmonize"],
        },
    )
    all_results = list()

    for seed, fold in product(config["seed"], config["fold"]):
        trainer = SingleStageFrameworkTrainer(
            dataloader=dataloader,
            trainer_params=TrainerParams(
                output_dir,
                config.get("model_name"),
                config.get("model_params", dict()),
                config.get("optim_params", dict()),
                config.get("hyperparameters", dict()),
                Dataset(config["dataset"]),
                seed,
                fold,
                config.get("ssl", False),
                config.get("harmonize", False),
                config.get("validation", False),
                config.get("labeled_sites", None),
                config.get("unlabeled_sites", None),
                config.get("num_unlabeled", None),
                config.get("device", -1),
                config.get("verbose", False),
                config.get("patience", float("inf")),
                config.get("max_epoch", 1000),
                config.get("save_model", False),
                config.get("dataloader_num_process", 10),
            ),
        )
        result = experiment(trainer)
        all_results.append(result)

        logging.info("RESULT:\n{}".format(json.dumps(result, indent=4)))

        df = pd.DataFrame(all_results).dropna(how="all")
        if df.empty:
            continue

        mkdir(output_dir)
        results_path = os.path.join(
            output_dir, "{}.csv".format(experiment_name),
        )
        df.to_csv(results_path, index=False)


def main(args):
    with open(os.path.abspath(args.config), "r") as f:
        configs: Dict[str, Any] = yaml.full_load(f)

    parser: ConfigParser = ConfigParser.parse(**configs)
    for config in parser.generate():
        process(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config_templates/single_stage_framework/config.yml",
        help="the path to the config file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] - %(filename)s: %(levelname)s: "
        "%(funcName)s(): %(lineno)d:\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main(args)
