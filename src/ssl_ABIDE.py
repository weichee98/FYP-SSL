import os
import copy
import time
import random
import argparse
import torch
import numpy as np
import pandas as pd

from ABIDE import load_data_fmri, get_ages_and_genders
from config import EXPERIMENT_DIR
from utils import mkdir, on_error, corr_mx_flatten, get_pop_A, make_graph
from models import ChebGCN, train_GCN, test_GCN
from models import FFN, train_FFN, test_FFN


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
        model="GCN", emb_size=150, K=3, L1=50, L2=30, 
        ssl=True, test=True, save_model=True, 
        seed=0, fold=0,
    ):
    param = dict()
    param["seed"] = seed
    param["fold"] = fold
    param["model"] = model
    param["emb_size"] = emb_size    # GCN parameter
    param["K"] = K                  # GCN parameter
    param["L1"] = L1                # FFN parameter
    param["L2"] = L2                # FFN parameter
    param["ssl"] = ssl
    param["test"] = test
    param["save_model"] = save_model
    param["model_path"] = None      # set when run
    param["acc"] = None             # set when run
    param["loss"] = None            # set when run
    return param

def set_experiment_param(param, model_path, acc, loss):
    param["model_path"] = model_path
    param["acc"] = acc
    param["loss"] = loss 

def verbose_info(epoch, train_loss, test_loss, train_acc, test_acc):
    return "Epoch: {:03d}, Train Acc: {:.4f}, Test Acc: {:.4f}, " \
        "Train Loss: {:.4f}, Test Loss: {:.4f}".format(
            epoch, train_acc, test_acc, train_loss, test_loss
        )

@on_error({}, True)
def experiment(device, verbose, param, model_dir):
    X, Y, _, splits = load_data_fmri()
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
        A = get_pop_A(X_flattened, ages, genders)
        data = make_graph(X_flattened, A, Y.argmax(axis=1))
    else:
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

    if param["model"] == "FFN":
        model = FFN(
            input_size=X_flattened.shape[1], 
            l1=param["L1"], 
            l2=param["L2"]
        )
    elif param["model"] == "GCN":
        model = ChebGCN(
            input_size=X_flattened.shape[1], 
            graph_embedding_size=param["emb_size"], 
            K=param["K"]
        )
    else:
        raise TypeError("Invalid model of type {}".format(param["model"]))

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.0001
    )

    best_acc = 0
    best_loss = np.inf
    best_model = None
    for epoch in range(1, 251):
        if param["model"] == "GCN":
            train_loss, train_acc = train_GCN(
                device, model, data, optimizer, nontest_indices
            )
            test_loss, test_acc = test_GCN(
                device, model, data, test_indices
            )
        elif param["model"] == "FFN":
            train_loss, train_acc = train_FFN(
                device, model, data, optimizer, nontest_indices
            )
            test_loss, test_acc = test_FFN(
                device, model, data, test_indices
            )

        if test_loss < best_loss:
            best_loss = test_loss

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())
            model_time = int(time.time())

        if verbose and epoch % 10 == 0:
            print(verbose_info(
                epoch, train_loss, test_loss, train_acc, test_acc
            ))

    if param["save_model"] and best_model is not None:
        model_name = "{}.pt".format(model_time)
        model_path = os.path.join(model_dir, model_name)
        torch.save(best_model, model_path)
    else:
        model_path = None

    set_experiment_param(param, model_path, best_acc, best_loss)
    if verbose:
        print(param)
    return param

def main(args):
    if args.gpu >= 0 and torch.cuda.is_available():
        print("Using device: cuda:{}".format(args.gpu))
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        print("Using device: cpu")
        device = torch.device("cpu")

    seed_torch()
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    experiment_name = "{}_{}".format(script_name, int(time.time()))

    exp_dir = os.path.join(args.exp_dir, experiment_name)
    model_dir = os.path.join(exp_dir, "models")
    mkdir(exp_dir)
    mkdir(model_dir)
    print("Experiment result: {}".format(exp_dir))

    res = []
    for seed in range(10):
        param = get_experiment_param(
            model="GCN", emb_size=150, K=3, seed=seed, ssl=False
        )
        exp_res = experiment(device, args.verbose, param, model_dir)
        res.append(exp_res)
    
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