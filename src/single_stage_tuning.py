import os
import json
import time
import numpy as np
import yaml
import logging
import argparse
import pandas as pd
from itertools import product
from typing import Any, Dict

from data import Dataset
from config import (
    EXPERIMENT_DIR,
    CandidateParameters,
    TuningConfigParser,
    TuningStrategy,
)
from tuning import TuningBase, load_tuning_object
from utils import mkdir, on_error, seed_torch
from factory import SingleStageFrameworkFactory
from trainer import SingleStageFrameworkTrainer, TrainerParams


@on_error(dict(), True)
def experiment(trainer: SingleStageFrameworkTrainer):
    trainer_results = trainer.run()
    return trainer_results.to_dict()


def process(config: Dict[str, Any]):
    seed_torch()
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
    results_path = os.path.join(output_dir, "{}.csv".format(experiment_name),)
    cv_results_path = os.path.join(
        output_dir, "{}_cv_results.csv".format(experiment_name)
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

    seed_fold_list = list(product(config["seed"], config["fold"]))
    tuning: TuningBase = load_tuning_object(
        strategy=TuningStrategy(**config["strategy"]),
        params=config["tuning_params"],
        seed_fold_list=seed_fold_list,
    )

    all_results = list()
    for param in tuning.next_sample():
        param = CandidateParameters.decompose_sampled_params(param)
        for seed, fold in seed_fold_list:
            trainer = SingleStageFrameworkTrainer(
                dataloader=dataloader,
                trainer_params=TrainerParams(
                    output_dir,
                    config.get("model_name"),
                    param.model_params,
                    param.optim_params,
                    param.hyperparameters,
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
            tuning.update_param_model_size(result.get("model_size", np.nan))
            tuning.update_param_result(
                seed, fold, result.get(tuning.metric, np.nan)
            )
            all_results.append(result)

            logging.info("RESULT:\n{}".format(json.dumps(result, indent=4)))

            df = pd.DataFrame(all_results).dropna(how="all")
            if not df.empty:
                df.to_csv(results_path, index=False)

        cv_result = tuning.cv_results
        if not cv_result.empty:
            cv_result.to_csv(cv_results_path, index=False)


def main(args):
    with open(os.path.abspath(args.config), "r") as f:
        configs: Dict[str, Any] = yaml.full_load(f)

    parser: TuningConfigParser = TuningConfigParser.parse(**configs)
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
