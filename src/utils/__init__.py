import os
import time
import random
import traceback
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
                    traceback.print_exc()
                return value
        return wrapper
    return decorator


def log_time(function):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = function(*args, **kwargs)
        end = time.time()
        print(
            "[{}] takes {:.5f} s"
            .format(function.__name__, end - start)
        )
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
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def verbose_info(epoch, train_loss, test_loss, train_acc, test_acc):
    return "Epoch: {:03d}, Train Acc: {:.4f}, Test Acc: {:.4f}, " \
        "Train Loss: {:.4f}, Test Loss: {:.4f}".format(
            epoch, train_acc, test_acc, train_loss, test_loss
        )


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