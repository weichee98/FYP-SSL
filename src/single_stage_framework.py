import os
import time
import yaml
import logging
import argparse
import pandas as pd
from itertools import product
from typing import Any, Dict, List
from joblib import Parallel, delayed

from data import Dataset
from config import EXPERIMENT_DIR
from utils import mkdir, on_error
from factory import SingleStageFrameworkFactory
from trainer import SingleStageFrameworkTrainer, TrainerParams


@on_error(dict(), True)
def experiment(trainer: SingleStageFrameworkTrainer):
    trainer_results = trainer.run()
    return trainer_results.to_dict()


def process(config: Dict[str, Any]):
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    experiment_name = "{}_{}".format(script_name, int(time.time()))
    output_dir = os.path.abspath(
        os.path.join(
            config.get("output_directory", EXPERIMENT_DIR), experiment_name
        )
    )

    dataloader = SingleStageFrameworkFactory.load_dataloader(
        model_name=config["model_name"],
        dataloader_param={
            "dataset": Dataset(config["dataset"]),
            "harmonize": config["harmonize"],
        },
    )
    all_results = list()

    for seed, fold in product(range(10), range(5)):
        trainer = SingleStageFrameworkTrainer(
            dataloader=dataloader,
            trainer_params=TrainerParams(
                output_dir,
                config.get("model_name", dict()),
                config.get("model_params", dict()),
                config.get("optim_params", dict()),
                config.get("hyperparameters", dict()),
                Dataset(config["dataset"]),
                seed,
                fold,
                config["ssl"],
                config["harmonize"],
                config["validation"],
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
        configs: List[Dict[str, Any]] = yaml.full_load(f)

    Parallel(n_jobs=args.num_worker)(
        delayed(process)(config) for config in configs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config_templates/config.yml",
        help="the path to the config file",
    )
    parser.add_argument(
        "--num_worker",
        type=int,
        default=1,
        help="the number of workers for parallel trainer",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] - %(filename)s: %(levelname)s: "
        "%(funcName)s(): %(lineno)d:\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main(args)
