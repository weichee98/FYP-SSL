import os
import copy
import time
import argparse
import torch
import numpy as np
import pandas as pd

from ABIDE import *
from config import EXPERIMENT_DIR
from utils import *
from models import *
from data import *


def get_experiment_param(
        model="GCN", seed=0, fold=0, epochs=1000,
        ssl=True, test=True, save_model=True, 
        site="NYU", harmonized=False,
        lr=0.0001, l2_reg=0.001,
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
    7. DIVA model (emb, hidden1, hidden2, beta_klzd, beta_klzx, 
                   beta_klzy, beta_d, beta_y, beta_recon)
    """
    param = dict()
    param["site"] = site
    param["seed"] = seed
    param["fold"] = fold
    param["epochs"] = epochs
    param["model"] = model
    param["lr"] = lr
    param["l2_reg"] = l2_reg
    param["ssl"] = ssl
    param["test"] = test
    param["harmonized"] = harmonized
    param["save_model"] = save_model
    for k, v in kwargs.items():
        param[k] = v
    return param

def set_experiment_param(
        param, model_path, time, device, 
        best_epoch, acc, loss, **kwargs
    ):
    param["accuracy"] = acc
    param["loss"] = loss
    for k, v in kwargs.items():
        param[k] = v
    param["best_epoch"] = best_epoch
    param["model_path"] = model_path
    param["time_taken"] = time
    param["device"] = device

def load_data(param):
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
    elif param["model"] == "DIVA":
        (data, labeled_train_indices, 
        all_train_indices, test_indices) = load_DIVA_data(
            X, Y, get_sites(), param["ssl"], 
            labeled_train_indices, test_indices
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
    elif param["model"] == "DIVA":
        model = DIVA(
            input_size=data.x.size(1), 
            z_dim=param["emb"], 
            d_dim=data.d.unique().size(0), 
            hidden1=param["hidden1"], 
            hidden2=param["hidden2"]
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
        test_loss, test_acc, test_metrics = test_GCN(
            device, model, data, test_indices
        )
    elif param["model"] == "FFN":
        train_loss, train_acc = train_FFN(
            device, model, data, optimizer, labeled_train_indices,
            all_train_indices, param["gamma_lap"]
        )
        test_loss, test_acc, test_metrics = test_FFN(
            device, model, data, test_indices
        )
    elif param["model"] == "AE":
        train_loss, train_acc = train_AE(
            device, model, data, optimizer, labeled_train_indices,
            all_train_indices, param["gamma"]
        )
        test_loss, test_acc, test_metrics = test_AE(
            device, model, data, test_indices
        )
    elif param["model"] == "VAE":
        train_loss, train_acc = train_VAE(
            device, model, data, optimizer, labeled_train_indices,
            all_train_indices, param["gamma1"], param["gamma2"]
        )
        test_loss, test_acc, test_metrics = test_VAE(
            device, model, data, test_indices
        )
    elif param["model"] == "GNN":
        train_dl, _, test_dl = data
        train_loss, train_acc = train_GNN(
            device, model, train_dl, optimizer, param["gamma"]
        )
        test_loss, test_acc, test_metrics = test_GNN(
            device, model, test_dl
        )
    elif param["model"] == "VGAE":
        labeled_dl, unlabeled_dl, test_dl = data
        train_loss, train_acc = train_VGAE(
            device, model, labeled_dl, unlabeled_dl, optimizer,
            param["gamma1"], param["gamma2"]
        )
        test_loss, test_acc, test_metrics = test_VGAE(
            device, model, test_dl
        )
    elif param["model"] == "DIVA":
        train_loss, train_acc = train_DIVA(
            device, model, data, optimizer, labeled_train_indices,
            all_train_indices, param["beta_klzd"], param["beta_klzx"],
            param["beta_klzy"], param["beta_d"], param["beta_y"],
            param["beta_recon"]
        )
        test_loss, test_acc, test_metrics = test_DIVA(
            device, model, data, test_indices
        )
    else:
        raise TypeError(
            "Invalid model of type {}".format(param["model"])
        )
    return train_loss, train_acc, test_loss, test_acc, test_metrics

def verbose_info(epoch, train_loss, test_loss, train_acc, test_acc):
    return "Epoch: {:03d}, Train Acc: {:.4f}, Test Acc: {:.4f}, " \
        "Train Loss: {:.4f}, Test Loss: {:.4f}".format(
            epoch, train_acc, test_acc, train_loss, test_loss
        )

@on_error({}, True)
def experiment(args, param, model_dir):
    seed_torch()
    device = get_device(args.gpu)
    verbose = args.verbose

    start = time.time()
    (data, labeled_train_indices, 
    all_train_indices, test_indices) = load_data(param)
    model = load_model(param, data)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=param["lr"], weight_decay=param["l2_reg"]
    )

    best_epoch = 0
    best_acc = 0
    best_loss = np.inf
    acc_loss = np.inf
    best_metrics = {}
    best_model = None
    patience = 1000
    cur_patience = 0
    pbar = get_pbar(param["epochs"], verbose)

    for epoch in pbar:
        train_loss, train_acc, test_loss, test_acc, test_metrics = train_test_step(
            param, device, model, data, optimizer, 
            labeled_train_indices, all_train_indices,
            test_indices
        )

        # early stopping
        if test_loss >= best_loss:
            cur_patience += 1
        else:
            cur_patience = 0

        """
        save priority
        1. accuracy
        2. f1
        3. loss
        """
        save = (test_acc > best_acc) or (
            test_acc == best_acc and (
                test_metrics["f1"] > best_metrics["f1"] or (
                    test_metrics["f1"] == best_metrics["f1"] and
                    test_loss < acc_loss
                )
            )
        )
        if test_loss < best_loss:
            best_loss = test_loss
        if save:
            best_epoch = epoch
            best_acc = test_acc
            acc_loss = test_loss
            best_metrics = test_metrics
            if param["save_model"]:
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
        param, time=end - start, device=args.gpu, model_path=model_path, 
        best_epoch=best_epoch, acc=best_acc, loss=best_loss,
        acc_loss=acc_loss, **best_metrics
    )
    print(param)
    return param

def main(args):
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    sites = np.array([
        "CALTECH", "LEUVEN_1", "LEUVEN_2", "MAX_MUN", "NYU", "OHSU", 
        "OLIN", "PITT", "STANFORD", "TRINITY", "UCLA_1", "UCLA_2", 
        "UM_1", "UM_2", "USM", "YALE"
    ])
    print(sites)

    for seed in range(10):
        ssl = True
        harmonized = True
            
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
                #     site=site, lr=0.00005, l2_reg=0.001,
                #     test=False, harmonized=harmonized, epochs=1000
                # )
                # param = get_experiment_param(
                #     model="FFN", L1=150, L2=50, L3=30, gamma_lap=0, 
                #     seed=seed, fold=fold, ssl=ssl, save_model=False, 
                #     site=site, lr=0.00005, l2_reg=0.001,
                #     test=False, harmonized=harmonized, epochs=1000
                # )
                # param = get_experiment_param(
                #     model="AE", L1=300, L2=50, emb=150, L3=30, gamma=1e-3, 
                #     seed=seed, fold=fold, ssl=ssl, save_model=False, 
                #     site=site, lr=0.0001, l2_reg=0.001,
                #     test=False, harmonized=harmonized, epochs=1000
                # )
                # param = get_experiment_param(
                #     model="VAE", L1=300, L2=50, emb=150, L3=30, gamma1=1e-5, gamma2=1e-3, 
                #     seed=seed, fold=fold, ssl=ssl, save_model=False,
                #     site=site, lr=0.0001, l2_reg=0.001, 
                #     test=False, harmonized=harmonized, epochs=1000
                # )
                # param = get_experiment_param(
                #     model="VGAE", hidden=300, emb1=150, emb2=50, L1=30,
                #     gamma1=3e-6, gamma2=1e-6, num_process=10, batch_size=10,
                #     seed=seed, fold=fold, ssl=ssl, save_model=False,
                #     site=site, lr=0.0001, l2_reg=0.001, 
                #     test=False, harmonized=harmonized, epochs=500
                # )
                # param = get_experiment_param(
                #     model="GNN", hidden=300, emb1=150, emb2=50, L1=30,
                #     gamma=0, num_process=10, batch_size=10,
                #     seed=seed, fold=fold, ssl=ssl, save_model=False,
                #     site=site, lr=0.0001, l2_reg=0.001, 
                #     test=False, harmonized=harmonized, epochs=500
                # )
                # param = get_experiment_param(
                #     model="DIVA", hidden1=150, emb=50, hidden2=30, 
                #     beta_klzd=1, beta_klzx=1, beta_klzy=1, 
                #     beta_d=1, beta_y=1, beta_recon=3e-6,
                #     seed=seed, fold=fold, ssl=ssl, save_model=False,
                #     site=site, lr=0.0001, l2_reg=0.001, 
                #     test=False, harmonized=harmonized, epochs=500
                # )

                # SSL
                param = get_experiment_param(
                    model="VAE", L1=300, L2=50, emb=150, L3=30, gamma1=3e-5, gamma2=1e-3, 
                    seed=seed, fold=fold, ssl=ssl, save_model=False,
                    site=site, lr=0.0001, l2_reg=0.001, 
                    test=False, harmonized=harmonized, epochs=1000
                )
                # param = get_experiment_param(
                #     model="AE", L1=300, L2=50, emb=150, L3=30, gamma=1e-3, 
                #     seed=seed, fold=fold, ssl=ssl, save_model=False, 
                #     site=site, lr=0.0001, l2_reg=0.001,
                #     test=False, harmonized=harmonized, epochs=1000
                # )
                exp_res = experiment(args, param, model_dir)
                res.append(exp_res)
        
        mkdir(exp_dir)
        df = pd.DataFrame(res).dropna(how="all")
        if not df.empty:
            res_path = os.path.join(exp_dir, "{}.csv".format(experiment_name))
            df.to_csv(res_path, index=False)


    # res = []
    # experiment_name = "{}_{}".format(script_name, int(time.time()))
    # exp_dir = os.path.join(args.exp_dir, experiment_name)
    # model_dir = os.path.join(exp_dir, "models")    
    # print("Experiment result: {}".format(exp_dir))

    # for seed in range(10):
    #     for harmonized in [False, True]:                
    #         print("===================")
    #         print("EXPERIMENT SETTINGS")
    #         print("===================")
    #         print("HARMONIZED: {}".format(harmonized))
    #         for fold in range(5):
    #             param = get_experiment_param(
    #                 model="FFN", L1=150, L2=50, L3=30, gamma_lap=0, 
    #                 seed=seed, fold=fold, ssl=False, save_model=False, 
    #                 site=None, lr=0.00005, l2_reg=0.001,
    #                 test=False, harmonized=harmonized, epochs=1000
    #             )
    #             exp_res = experiment(args, param, model_dir)
    #             res.append(exp_res)
            
    # mkdir(exp_dir)
    # df = pd.DataFrame(res).dropna(how="all")
    # if not df.empty:
    #     res_path = os.path.join(exp_dir, "{}.csv".format(experiment_name))
    #     df.to_csv(res_path, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1,
                        help='gpu id (0, 1, 2, 3) or cpu (-1)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--exp_dir', type=str, default=EXPERIMENT_DIR,
                        help="directory to save experiment results")
    args = parser.parse_args()
    main(args)