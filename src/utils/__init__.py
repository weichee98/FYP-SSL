import os
import time
import random
import logging
import traceback
import subprocess
import numpy as np
import torch
from tqdm import tqdm


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def on_error(value, print_error_stack=True):
    """
    returns a wrapper which catches error within a function 
    and returns a default value on error
    value: the default value to be returned when error occured
    """

    def decorator(function):
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                if print_error_stack:
                    logging.error(e)
                    traceback.print_exc()
                return value

        return wrapper

    return decorator


def log_time(function):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = function(*args, **kwargs)
        end = time.time()
        print("[{}] takes {:.5f} s".format(function.__name__, end - start))
        return res

    return wrapper


def get_device(id):
    if id >= 0 and torch.cuda.is_available():
        print("Using device: cuda:{}".format(id))
        device = torch.device("cuda:{}".format(id))
    else:
        print("Using device: cpu")
        device = torch.device("cpu")
    return device


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def epoch_gen(max_epoch=1000):
    i = 1
    while i <= max_epoch:
        yield i
        i += 1


def get_pbar(max_epoch, verbose):
    if verbose:
        return tqdm(epoch_gen(max_epoch))
    else:
        return epoch_gen(max_epoch)


def get_gpu(num_polling: int = 20):
    polled_memory_free = list()
    for _ in range(num_polling):
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = (
            subprocess.check_output(command.split())
            .decode("ascii")
            .split("\n")[:-1][1:]
        )
        memory_free_values = [
            int(x.split()[0]) for i, x in enumerate(memory_free_info)
        ]
        polled_memory_free.append(memory_free_values)
    gpu = np.argmax(np.mean(memory_free_values, axis=0))
    return get_device(gpu)
