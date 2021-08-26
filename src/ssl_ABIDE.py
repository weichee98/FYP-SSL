import os
import copy
import time
import random
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

from ABIDE import load_data_fmri, get_ages_and_genders
from config import EXPERIMENT_DIR
from metrics import EMA
from utils import (
    mkdir, on_error, corr_mx_flatten, get_pop_A, 
    make_graph,
)
from models import ChebGCN, train_GCN, test_GCN
from models import FFN, train_FFN, test_FFN


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
    torch.cuda.manual_seed_all(seed)    # if use multi-GPU (set just in case)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_experiment_param(
        model="GCN", seed=0, fold=0, 
        ssl=True, test=True, save_model=True, 
        site="NYU", harmonized=False,
        ema=None, lr=0.0001,
        **kwargs
    ):
    """
    kwargs:
    1. GCN model (hidden, emb1, emb2, K)
    2. FFN model (L1, L2, L3, gamma_lap)
    """
    param = dict()
    param["site"] = site
    param["seed"] = seed
    param["fold"] = fold
    param["model"] = model
    param["lr"] = lr
    param["ema"] = ema
    param["ssl"] = ssl
    param["test"] = test
    param["harmonized"] = harmonized
    param["save_model"] = save_model
    param["model_path"] = None      # set when run
    param["acc"] = None             # set when run
    param["loss"] = None            # set when run
    param["time"] = None            # set when run
    param["device"] = None          # set when run
    for k, v in kwargs.items():
        param[k] = v
    return param

def set_experiment_param(param, model_path, acc, loss, time, device):
    param["model_path"] = model_path
    param["acc"] = acc
    param["loss"] = loss 
    param["time"] = time
    param["device"] = device

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

@on_error({}, True)
def experiment(device, verbose, param, model_dir):
    start = time.time()
    X, Y, splits = load_data_fmri(site_id=param["site"], harmonized=param["harmonized"])
    X_flattened = corr_mx_flatten(X)
    ages, genders = get_ages_and_genders()

    seed = param["seed"]
    fold = param["fold"]
    if param["test"]:
        test_indices = splits[seed][0]
        train_indices, val_indices = splits[seed][1][fold]
        nontest_indices = np.concatenate([train_indices, val_indices], axis=0)
    else:
        nontest_indices, test_indices = splits[seed][1][fold]

    if param["ssl"]:
        # if SSL is used, all subjects from all sites are used to create graph
        A = get_pop_A(X_flattened, ages, genders)
        data = make_graph(X_flattened, A, Y.argmax(axis=1))
        laplacian_indices = np.setdiff1d(np.arange(len(X_flattened)), test_indices)
    else:
        # if SSL is not used, only subjects from largest site is used to create graph
        # adjust the indices accordingly to match the subject used in the graph
        all_indices = np.concatenate([nontest_indices, test_indices], axis=0)
        A = get_pop_A(
            X_flattened[all_indices], ages[all_indices], genders[all_indices]
        )
        data = make_graph(
            X_flattened[all_indices], A, Y[all_indices].argmax(axis=1)
        )
        n_nontest = len(nontest_indices)
        n_test = len(test_indices)
        nontest_indices = np.array(range(n_nontest))
        test_indices = np.array(range(n_nontest, n_nontest + n_test))
        laplacian_indices = None

    if param["model"] == "FFN":
        model = FFN(
            input_size=X_flattened.shape[1], 
            l1=param["L1"], 
            l2=param["L2"],
            l3=param["L3"]
        )
    elif param["model"] == "GCN":
        model = ChebGCN(
            input_size=X_flattened.shape[1], 
            hidden=param["hidden"],
            emb1=param["emb1"], 
            emb2=param["emb2"], 
            K=param["K"]
        )
    else:
        raise TypeError("Invalid model of type {}".format(param["model"]))

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=param["lr"]
    )

    best_acc = 0
    best_loss = np.inf
    best_model = None
    ema = EMA(k=param["ema"])
    patience = 100
    cur_patience = 0
    pbar = get_pbar(1000, args.verbose)

    for epoch in pbar:
        if param["model"] == "GCN":
            train_loss, train_acc = train_GCN(
                device, model, data, optimizer, nontest_indices
            )
            test_loss, test_acc = test_GCN(
                device, model, data, test_indices
            )
        elif param["model"] == "FFN":
            train_loss, train_acc = train_FFN(
                device, model, data, optimizer, nontest_indices,
                laplacian_indices, param["gamma_lap"]
            )
            test_loss, test_acc = test_FFN(
                device, model, data, test_indices
            )

        train_loss, train_acc, test_loss, test_acc = ema.update(
            train_loss, train_acc, test_loss, test_acc
        )

        # early stopping
        if test_loss >= best_loss:
            cur_patience += 1
        else:
            cur_patience = 0

        if test_loss < best_loss:
            best_loss = test_loss
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())
            model_time = int(time.time())

        if verbose:
            pbar.set_postfix_str(verbose_info(
                epoch, train_loss, test_loss, train_acc, test_acc
            ))

        # early stopping
        if cur_patience == patience:
            break

    if param["save_model"] and best_model is not None:
        mkdir(model_dir)
        model_name = "{}.pt".format(model_time)
        model_path = os.path.join(model_dir, model_name)
        torch.save(best_model, model_path)
    else:
        model_path = None

    end = time.time()
    set_experiment_param(param, model_path, best_acc, best_loss, end - start, args.gpu)
    if verbose:
        print(param)
    return param

def main(args):
    device = get_device(args.gpu)
    seed_torch()
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    for ssl in (True, False):
        experiment_name = "{}_{}".format(script_name, int(time.time()))
        exp_dir = os.path.join(args.exp_dir, experiment_name)
        model_dir = os.path.join(exp_dir, "models")    
        print("Experiment result: {}".format(exp_dir))

        res = []
        for seed in range(100):
            # param = get_experiment_param(
            #     model="GCN", hidden=150, emb1=50, emb2=30, K=3, 
            #     seed=seed, ssl=ssl, save_model=False, harmonized=False,
            #     site="NYU", ema=0.2, lr=0.00005
            # )
            param = get_experiment_param(
                model="FFN", L1=150, L2=50, L3=30, gamma_lap=0.2, 
                seed=seed, ssl=ssl, save_model=False, harmonized=False,
                site="NYU", ema=0.2, lr=0.00005
            )
            exp_res = experiment(device, args.verbose, param, model_dir)
            res.append(exp_res)
        
        mkdir(exp_dir)
        df = pd.DataFrame(res).dropna(how="all")
        res_path = os.path.join(exp_dir, "{}.csv".format(experiment_name))
        df.to_csv(res_path, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1,
                        help='gpu id (0, 1, 2, 3) or cpu (-1)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--exp_dir', type=str, default=EXPERIMENT_DIR,
                        help="directory to save experiment results")
    args = parser.parse_args()    
    main(args)
    