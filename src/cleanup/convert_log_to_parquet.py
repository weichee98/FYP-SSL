import os
import json
import psutil
import logging
import subprocess
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


def get_max_pid():
    return int(subprocess.check_output(["cat", "/proc/sys/kernel/pid_max"]))


def list_log(main_dir):
    target_subdir = "epochs_log"
    output_subdir = "epochs_log_parquet"
    max_pid = get_max_pid()

    main_dir = os.path.abspath(main_dir)
    for root, _, files in os.walk(main_dir):
        if not os.path.basename(root) == target_subdir:
            continue
        parent: str = os.path.dirname(root)
        try:
            parent_pid = int(parent.split("_")[-1])
        except:
            continue
        if parent_pid < 0 or parent_pid > max_pid:
            continue
        if psutil.pid_exists(parent_pid):
            continue
        output_dir = os.path.join(parent, output_subdir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        for file in files:
            if file.endswith(".log"):
                yield (
                    os.path.join(root, file),
                    os.path.join(output_dir, file + ".parquet"),
                )


def read_old_log(file_path):
    with open(file_path, "r") as f:
        df = pd.DataFrame([json.loads(line.strip()) for line in f.readlines()])
    return df


def save_new_log(df: pd.DataFrame, output_path):
    try:
        df.to_parquet(output_path)
        return True
    except Exception as e:
        logging.error(e)
        return False


def convert_log(log_path, output_path):
    df = read_old_log(log_path)
    if save_new_log(df, output_path):
        try:
            os.remove(log_path)
        except Exception as e:
            logging.error(e)


def remove_empty_log_dirs(main_dir):
    target_subdir = "epochs_log"

    main_dir = os.path.abspath(main_dir)
    to_be_removed = list()
    for root, _, files in os.walk(main_dir):
        if not os.path.basename(root) == target_subdir:
            continue
        if not os.listdir(root):
            to_be_removed.append(root)

    for directory in to_be_removed:
        try:
            os.rmdir(directory)
        except Exception as e:
            logging.error(e)


def main(main_dir):
    all_logs = list(list_log(main_dir))
    Parallel(os.cpu_count())(
        delayed(convert_log)(log_path, output_path)
        for log_path, output_path in tqdm(all_logs, ncols=60)
    )
    remove_empty_log_dirs(main_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] - %(filename)s: %(levelname)s: "
        "%(funcName)s(): %(lineno)d:\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main("/data/yeww0006/FYP-SSL/.archive")
