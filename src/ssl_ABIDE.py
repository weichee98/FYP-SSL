import os
import copy
import time
import random
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from ABIDE import *
from config import EXPERIMENT_DIR
from metrics import EMA
from utils import mkdir, on_error
from models import *
from data import *


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

def get_experiment_param(
        model="GCN", seed=0, fold=0, 
        ssl=True, test=True, save_model=True, 
        site="NYU", harmonized=False,
        ema=None, lr=0.0001, l2_reg=0.001,
        **kwargs
    ):
    """
    kwargs:
    1. GCN model (hidden, emb1, emb2, K)
    2. FFN model (L1, L2, L3, gamma_lap)
    3. AE model (L1, L2, L3, emb, gamma)
    4. VAE model (L1, L2, L3, emb, gamma1, gamma2)
    5. VGAE model (hidden, emb1, emb2, L1, gamma1, gamma2, num_process, batch_size)
    6. GNN model (hidden, emb1, emb2, L1, gamma, num_process, batch_size)
    """
    param = dict()
    param["site"] = site
    param["seed"] = seed
    param["fold"] = fold
    param["model"] = model
    param["lr"] = lr
    param["l2_reg"] = l2_reg
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

def load_data(param, verbose):
    X, Y = load_data_fmri(harmonized=param["harmonized"])
    splits = get_splits(site_id=param["site"], test=param["test"])
    ages, genders = get_ages_and_genders()

    seed = param["seed"]
    fold = param["fold"]
    if param["test"]:
        test_indices = splits[seed][0]
        train_indices, val_indices = splits[seed][1][fold]
        labeled_train_indices = np.concatenate(
            [train_indices, val_indices], axis=0
        )
    else:
        labeled_train_indices, test_indices = splits[seed][1][fold]

    if param["model"] == "GCN":
        (data, labeled_train_indices, 
        all_train_indices, test_indices) = load_GCN_data(
            X, Y, ages, genders, param["ssl"], 
            labeled_train_indices, test_indices
        )
    elif param["model"] == "FFN":
        (data, labeled_train_indices, 
        all_train_indices, test_indices) = load_FFN_data(
            X, Y, ages, genders, param["ssl"], 
            labeled_train_indices, test_indices
        )
    elif param["model"] == "AE" or param["model"] == "VAE":
        (data, labeled_train_indices, 
        all_train_indices, test_indices) = load_AE_data(
            X, Y, param["ssl"], labeled_train_indices, 
            test_indices
        )
    elif param["model"] == "GNN":
        (data, labeled_train_indices, 
        all_train_indices, test_indices) = load_GNN_data(
            X, Y, labeled_train_indices, test_indices,
            param.get("num_process", 1), param["batch_size"], 
            verbose=False
        )
    elif param["model"] == "VGAE":
        (data, labeled_train_indices, 
        all_train_indices, test_indices) = load_GAE_data(
            X, Y, param["ssl"], labeled_train_indices, 
            test_indices, param.get("num_process", 1),
            param["batch_size"], verbose=False
        )
    else:
        raise NotImplementedError(
            "No dataloader function implemented for model {}"
            .format(param["model"])
        )

    num_labeled = labeled_train_indices.shape[0]
    num_all = num_labeled \
        if all_train_indices is None \
        else all_train_indices.shape[0]
    num_test = test_indices.shape[0]

    print("NUM_LABELED_TRAIN: {}".format(num_labeled))
    print("NUM_UNLABELED_TRAIN: {}".format(num_all - num_labeled))
    print("NUM_TRAIN: {}".format(num_all))
    print("NUM_TEST: {}".format(num_test))

    return data, labeled_train_indices, all_train_indices, test_indices

def load_model(param, data):
    if param["model"] == "FFN":
        model = FFN(
            input_size=data.x.size(1), 
            l1=param["L1"], 
            l2=param["L2"],
            l3=param["L3"]
        )
    elif param["model"] == "GCN":
        model = ChebGCN(
            input_size=data.x.size(1), 
            hidden=param["hidden"],
            emb1=param["emb1"], 
            emb2=param["emb2"], 
            K=param["K"]
        )
    elif param["model"] == "AE":
        model = AE(
            input_size=data.x.size(1), 
            l1=param["L1"], 
            l2=param["L2"],
            l3=param["L3"],
            emb_size=param["emb"],
        )
    elif param["model"] == "VAE":
        model = VAE(
            input_size=data.x.size(1), 
            l1=param["L1"], 
            l2=param["L2"],
            l3=param["L3"], 
            emb_size=param["emb"], 
        )
    elif param["model"] == "GNN":
        batch = next(iter(data[0]))
        model = GNN(
            input_size=batch.x.size(1), 
            hidden=param["hidden"],
            emb1=param["emb1"], 
            emb2=param["emb2"],
            l1=param["L1"], 
        )
    elif param["model"] == "VGAE":
        batch = next(iter(data[0]))
        model = VGAE(
            input_size=batch.x.size(1), 
            hidden=param["hidden"],
            emb1=param["emb1"], 
            emb2=param["emb2"],
            l1=param["L1"], 
        )
    else:
        raise TypeError(
            "Invalid model of type {}".format(param["model"])
        )
    return model

def train_test_step(
        param, device, model, data, optimizer, 
        labeled_train_indices, all_train_indices,
        test_indices
    ):
    if param["model"] == "GCN":
        train_loss, train_acc = train_GCN(
            device, model, data, optimizer, labeled_train_indices
        )
        test_loss, test_acc = test_GCN(
            device, model, data, test_indices
        )
    elif param["model"] == "FFN":
        train_loss, train_acc = train_FFN(
            device, model, data, optimizer, labeled_train_indices,
            all_train_indices, param["gamma_lap"]
        )
        test_loss, test_acc = test_FFN(
            device, model, data, test_indices
        )
    elif param["model"] == "AE":
        train_loss, train_acc = train_AE(
            device, model, data, optimizer, labeled_train_indices,
            all_train_indices, param["gamma"]
        )
        test_loss, test_acc = test_AE(
            device, model, data, test_indices
        )
    elif param["model"] == "VAE":
        train_loss, train_acc = train_VAE(
            device, model, data, optimizer, labeled_train_indices,
            all_train_indices, param["gamma1"], param["gamma2"]
        )
        test_loss, test_acc = test_VAE(
            device, model, data, test_indices
        )
    elif param["model"] == "GNN":
        train_dl, _, test_dl = data
        train_loss, train_acc = train_GNN(
            device, model, train_dl, optimizer, param["gamma"]
        )
        test_loss, test_acc = test_GNN(
            device, model, test_dl
        )
    elif param["model"] == "VGAE":
        labeled_dl, unlabeled_dl, test_dl = data
        train_loss, train_acc = train_VGAE(
            device, model, labeled_dl, unlabeled_dl, optimizer,
            param["gamma1"], param["gamma2"]
        )
        test_loss, test_acc = test_VGAE(
            device, model, test_dl
        )
    else:
        raise TypeError(
            "Invalid model of type {}".format(param["model"])
        )
    return train_loss, train_acc, test_loss, test_acc

@on_error({}, True)
def experiment(args, param, model_dir):
    device = get_device(args.gpu)
    verbose = args.verbose

    start = time.time()
    (data, labeled_train_indices, 
    all_train_indices, test_indices) = load_data(param, verbose)
    model = load_model(param, data)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=param["lr"], weight_decay=param["l2_reg"]
    )

    best_acc = 0
    best_loss = np.inf
    best_model = None
    ema = EMA(k=param["ema"])
    patience = 1000
    cur_patience = 0
    pbar = get_pbar(1000, verbose)

    for epoch in pbar:
        train_loss, train_acc, test_loss, test_acc = train_test_step(
            param, device, model, data, optimizer, 
            labeled_train_indices, all_train_indices,
            test_indices
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
    set_experiment_param(
        param, model_path, best_acc, best_loss, end - start, args.gpu
    )
    print(param)
    return param

def main(args):
    seed_torch()
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    sites = np.array([
        "CALTECH", "LEUVEN_1", "LEUVEN_2", "MAX_MUN", "NYU", "OHSU", 
        "OLIN", "PITT", "STANFORD", "TRINITY", "UCLA_1", "UCLA_2", 
        "UM_1", "UM_2", "USM", "YALE"
    ])
    print(sites)

    for seed in range(10):
        ssl = False
        harmonized = False
            
        print("===================")
        print("EXPERIMENT SETTINGS")
        print("===================")
        print("SSL: {}".format(ssl))
        print("HARMONIZED: {}".format(harmonized))

        experiment_name = "{}_{}".format(script_name, int(time.time()))
        exp_dir = os.path.join(args.exp_dir, experiment_name)
        model_dir = os.path.join(exp_dir, "models")    
        print("Experiment result: {}".format(exp_dir))

        res = []
        for site in sites:
            print("SITE: {}".format(site))
            for fold in range(5):
                # param = get_experiment_param(
                #     model="GCN", hidden=150, emb1=50, emb2=30, K=3, 
                #     seed=seed, fold=fold, ssl=ssl, save_model=False, 
                #     site=site, ema=0.2, lr=0.00005, l2_reg=0.001,
                #     test=False, harmonized=harmonized,
                # )
                # param = get_experiment_param(
                #     model="FFN", L1=150, L2=50, L3=30, gamma_lap=0, 
                #     seed=seed, fold=fold, ssl=ssl, save_model=False, 
                #     site=site, ema=0.2, lr=0.00005, l2_reg=0.001,
                #     test=False, harmonized=harmonized,
                # )
                # param = get_experiment_param(
                #     model="AE", L1=300, L2=50, emb=150, L3=30, gamma=2e-3, 
                #     seed=seed, fold=fold, ssl=ssl, save_model=False, 
                #     site=site, ema=0.2, lr=0.0001, l2_reg=0.001,
                #     test=False, harmonized=harmonized,
                # )
                # param = get_experiment_param(
                #     model="VAE", L1=300, L2=50, emb=150, L3=30, gamma1=1e-5, gamma2=1e-3, 
                #     seed=seed, fold=fold, ssl=ssl, save_model=False,
                #     site=site, ema=0.2, lr=0.0001, l2_reg=0.001, 
                #     test=False, harmonized=harmonized,
                # )
                # param = get_experiment_param(
                #     model="VGAE", hidden=300, emb1=150, emb2=50, L1=30,
                #     gamma1=3e-6, gamma2=1e-6, num_process=10, batch_size=10,
                #     seed=seed, fold=fold, ssl=ssl, save_model=False,
                #     site=site, ema=0.2, lr=0.0001, l2_reg=0.001, 
                #     test=False, harmonized=harmonized,
                # )
                param = get_experiment_param(
                    model="GNN", hidden=300, emb1=150, emb2=50, L1=30,
                    gamma=0, num_process=10, batch_size=10,
                    seed=seed, fold=fold, ssl=ssl, save_model=False,
                    site=site, ema=0.2, lr=0.0001, l2_reg=0.001, 
                    test=False, harmonized=harmonized,
                )

                # SSL
                # param = get_experiment_param(
                #     model="VAE", L1=300, L2=50, emb=150, L3=30, gamma1=3e-5, gamma2=1e-3, 
                #     seed=seed, fold=fold, ssl=ssl, save_model=False,
                #     site=site, ema=0.2, lr=0.0001, l2_reg=0.001, 
                #     test=False, harmonized=harmonized,
                # )
                # param = get_experiment_param(
                #     model="AE", L1=300, L2=50, emb=150, L3=30, gamma=1e-3, 
                #     seed=seed, fold=fold, ssl=ssl, save_model=False, 
                #     site=site, ema=0.2, lr=0.0001, l2_reg=0.001,
                #     test=False, harmonized=harmonized,
                # )
                exp_res = experiment(args, param, model_dir)
                res.append(exp_res)
        
        mkdir(exp_dir)
        df = pd.DataFrame(res).dropna(how="all")
        if not df.empty:
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