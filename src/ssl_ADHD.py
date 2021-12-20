import os
import copy
import time
import json
import argparse
import torch
import numpy as np
import pandas as pd

from ADHD import *
from config import EXPERIMENT_DIR
from utils import *
from models import *
from data import *


def get_experiment_param(
    model="GCN",
    seed=0,
    fold=0,
    epochs=1000,
    patience=1000,
    ssl=True,
    test=True,
    save_model=True,
    site="NYU",
    harmonized=False,
    lr=0.0001,
    l2_reg=0.001,
    **kwargs
):
    """
    kwargs:
    1. GCN model (hidden, emb1, emb2, K)
    2. FFN model (L1, L2, L3, gamma_lap)
    3. AE model (L1, L2, L3, emb, gamma)
    4. VAE model (L1, L2, L3, emb, gamma1, gamma2)
    5. VGAE model (emb1, emb2, L1, gamma1, gamma2, num_process)
    6. GNN model (emb1, emb2, L1, gamma, num_process)
    7. DIVA model (emb, hidden1, hidden2, beta_klzd, beta_klzx, 
                   beta_klzy, beta_d, beta_y, beta_recon)
    8. VGAETS model (embts, emb1, emb2, L1, gamma1, gamma2, 
                    num_process)
    9. VAESDR model (L1, emb, gamma1, gamma2, gamma3, gamma4, gamma5)
    10. VAECH model (L1, L2, L3, emb, gamma1, gamma2)
    """
    param = dict()
    param["site"] = site
    param["seed"] = seed
    param["fold"] = fold
    param["epochs"] = epochs
    param["patience"] = patience
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


def set_experiment_param(param, model_path, time, device, best_epoch, loss, **kwargs):
    param["loss"] = loss
    for k, v in kwargs.items():
        param[k] = v
    param["best_epoch"] = best_epoch
    param["model_path"] = model_path
    param["time_taken"] = time
    param["device"] = device


def get_split(site, param):
    seed = param["seed"]
    fold = param["fold"]
    splits = get_splits(site_id=site, test=param["test"])
    if param["test"]:
        test_indices = splits[seed][0]
        train_indices, val_indices = splits[seed][1][fold]
        labeled_train_indices = np.concatenate([train_indices, val_indices], axis=0)
    else:
        labeled_train_indices, test_indices = splits[seed][1][fold]
    return labeled_train_indices, test_indices


def load_data(param):
    X, Y = load_data_fmri(harmonized=param["harmonized"])

    ages, genders = get_ages_and_genders()
    if param["site"] is None:
        sites = ["KKI", "NI", "NYU", "OHSU", "PKU"]
    else:
        sites = [param["site"]]
    indices = Parallel(len(sites))(delayed(get_split)(site, param) for site in sites)
    labeled_train_indices, test_indices = zip(*indices)
    labeled_train_indices = np.concatenate(labeled_train_indices, axis=0)
    test_indices = np.concatenate(test_indices, axis=0)
    param["baseline_accuracy"] = np.mean(Y[test_indices], axis=0).max()

    if param["model"] == "GCN":
        (data, labeled_train_indices, all_train_indices, test_indices) = load_GCN_data(
            X, Y, ages, genders, param["ssl"], labeled_train_indices, test_indices
        )
    elif param["model"] == "FFN":
        (data, labeled_train_indices, all_train_indices, test_indices) = load_FFN_data(
            X, Y, ages, genders, param["ssl"], labeled_train_indices, test_indices
        )
    elif param["model"] == "AE" or param["model"] == "VAE":
        (data, labeled_train_indices, all_train_indices, test_indices) = load_AE_data(
            X, Y, param["ssl"], labeled_train_indices, test_indices
        )
    elif param["model"] == "GNN":
        (data, labeled_train_indices, all_train_indices, test_indices) = load_GNN_data(
            X,
            Y,
            labeled_train_indices,
            test_indices,
            num_process=param.get("num_process", 1),
            verbose=False,
        )
    elif param["model"] == "VGAE":
        (data, labeled_train_indices, all_train_indices, test_indices) = load_GAE_data(
            X,
            Y,
            param["ssl"],
            labeled_train_indices,
            test_indices,
            num_process=param.get("num_process", 1),
            verbose=False,
        )
    elif param["model"] == "DIVA" or "VAESDR" in param["model"]:
        (data, labeled_train_indices, all_train_indices, test_indices) = load_DIVA_data(
            X, Y, get_sites(), param["ssl"], labeled_train_indices, test_indices
        )
    elif param["model"] in ["VAECH"]:
        (
            data,
            labeled_train_indices,
            all_train_indices,
            test_indices,
        ) = load_VAECH_data(
            X,
            Y,
            get_sites(),
            ages,
            genders,
            param["ssl"],
            labeled_train_indices,
            test_indices,
        )
    # elif param["model"] == "VGAETS":
    #     (data, labeled_train_indices,
    #     all_train_indices, test_indices) = load_GAE_data(
    #         X, Y, param["ssl"], labeled_train_indices,
    #         test_indices, X_ts,
    #         num_process=param.get("num_process", 1),
    #         verbose=False
    #     )
    else:
        raise NotImplementedError(
            "No dataloader function implemented for model {}".format(param["model"])
        )

    num_labeled = labeled_train_indices.shape[0]
    num_all = num_labeled if all_train_indices is None else all_train_indices.shape[0]
    num_test = test_indices.shape[0]

    print("NUM_LABELED_TRAIN: {}".format(num_labeled))
    print("NUM_UNLABELED_TRAIN: {}".format(num_all - num_labeled))
    print("NUM_TRAIN: {}".format(num_all))
    print("NUM_TEST: {}".format(num_test))

    param["num_labeled_train"] = num_labeled
    param["num_unlabeled_train"] = num_all - num_labeled
    param["num_test"] = num_test
    return data, labeled_train_indices, all_train_indices, test_indices


def load_model(param, data):
    if param["model"] == "FFN":
        model = FFN(
            input_size=data.x.size(1), l1=param["L1"], l2=param["L2"], l3=param["L3"]
        )
    elif param["model"] == "GCN":
        model = ChebGCN(
            input_size=data.x.size(1),
            hidden=param["hidden"],
            emb1=param["emb1"],
            emb2=param["emb2"],
            K=param["K"],
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
    elif "VAESDR" in param["model"]:
        model = VAESDR(
            input_size=data.x.size(1),
            l1=param["L1"],
            emb_size=param["emb"],
            num_site=data.d.unique().size(0),
        )
    elif param["model"] == "VAECH":
        model = VAECH(
            input_size=data.x.size(1),
            l1=param["L1"],
            l2=param["L2"],
            l3=param["L3"],
            emb_size=param["emb"],
            num_sites=data.d.size(1),
        )
    elif param["model"] == "GNN":
        batch = next(iter(data[0]))
        model = GNN(
            input_size=batch.x.size(1),
            emb1=param["emb1"],
            emb2=param["emb2"],
            l1=param["L1"],
        )
    elif param["model"] == "VGAE":
        batch = next(iter(data[0]))
        model = VGAE(
            input_size=batch.x.size(1),
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
            hidden2=param["hidden2"],
        )
    # elif param["model"] == "VGAETS":
    #     model = VGAETS(
    #         tsemb=param["tsemb"],
    #         emb1=param["emb1"],
    #         emb2=param["emb2"],
    #         l1=param["L1"],
    #         bidirectional=param["bidirectional"]
    #     )
    else:
        raise TypeError("Invalid model of type {}".format(param["model"]))
    param["model_size"] = count_parameters(model)
    print("MODEL_SIZE: {}".format(param["model_size"]))

    if "VAESDR" in param["model"]:
        model_optim = torch.optim.Adam(
            map(
                lambda p: p[1],
                filter(
                    lambda p: p[1].requires_grad
                    and "dis" not in p[0]
                    and "cls" not in p[0],
                    model.named_parameters(),
                ),
            ),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        disease_dis_optim = torch.optim.Adam(
            map(
                lambda p: p[1],
                filter(
                    lambda p: p[1].requires_grad and "disease_dis" in p[0],
                    model.named_parameters(),
                ),
            ),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        site_dis_optim = torch.optim.Adam(
            map(
                lambda p: p[1],
                filter(
                    lambda p: p[1].requires_grad and "site_dis" in p[0],
                    model.named_parameters(),
                ),
            ),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        disease_cls_optim = torch.optim.Adam(
            map(
                lambda p: p[1],
                filter(
                    lambda p: p[1].requires_grad and "disease_cls" in p[0],
                    model.named_parameters(),
                ),
            ),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        site_cls_optim = torch.optim.Adam(
            map(
                lambda p: p[1],
                filter(
                    lambda p: p[1].requires_grad and "site_cls" in p[0],
                    model.named_parameters(),
                ),
            ),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
        optimizer = (
            model_optim,
            disease_cls_optim,
            site_cls_optim,
            disease_dis_optim,
            site_dis_optim,
        )
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=param["lr"],
            weight_decay=param["l2_reg"],
        )
    return model, optimizer


def train_test_step(
    param,
    device,
    model,
    data,
    optimizer,
    labeled_train_indices,
    all_train_indices,
    test_indices,
):
    if param["model"] == "GCN":
        train_loss, train_acc, train_metrics = train_GCN(
            device, model, data, optimizer, labeled_train_indices, weight=False
        )
        test_loss, test_acc, test_metrics = test_GCN(device, model, data, test_indices)
    elif param["model"] == "FFN":
        train_loss, train_acc, train_metrics = train_FFN(
            device,
            model,
            data,
            optimizer,
            labeled_train_indices,
            all_train_indices,
            param["gamma_lap"],
            weight=False,
        )
        test_loss, test_acc, test_metrics = test_FFN(device, model, data, test_indices)
    elif param["model"] == "AE":
        train_loss, train_acc, train_metrics = train_AE(
            device,
            model,
            data,
            optimizer,
            labeled_train_indices,
            all_train_indices,
            param["gamma"],
            weight=False,
        )
        test_loss, test_acc, test_metrics = test_AE(device, model, data, test_indices)
    elif param["model"] == "VAE":
        train_loss, train_acc, train_metrics = train_VAE(
            device,
            model,
            data,
            optimizer,
            labeled_train_indices,
            all_train_indices,
            param["gamma1"],
            param["gamma2"],
            weight=False,
        )
        test_loss, test_acc, test_metrics = test_VAE(device, model, data, test_indices)
    elif "VAESDR" in param["model"]:
        train_loss, train_acc, train_metrics = train_VAESDR(
            device,
            model,
            data,
            optimizer,
            labeled_train_indices,
            all_train_indices,
            param["gamma1"],
            param["gamma2"],
            param["gamma3"],
            param["gamma4"],
            param["gamma5"],
        )
        test_loss, test_acc, test_metrics = test_VAESDR(
            device, model, data, test_indices
        )
    elif param["model"] == "VAECH":
        train_loss, train_acc, train_metrics = train_VAECH(
            device,
            model,
            data,
            optimizer,
            labeled_train_indices,
            all_train_indices,
            param["gamma1"],
            param["gamma2"],
        )
        test_loss, test_acc, test_metrics = test_VAECH(
            device, model, data, test_indices
        )
    elif param["model"] == "GNN":
        train_dl, _, test_dl = data
        train_loss, train_acc, train_metrics = train_GNN(
            device, model, train_dl, optimizer, weight=False
        )
        test_loss, test_acc, test_metrics = test_GNN(device, model, test_dl)
    elif param["model"] == "VGAE":
        labeled_dl, unlabeled_dl, test_dl = data
        train_loss, train_acc, train_metrics = train_VGAE(
            device,
            model,
            labeled_dl,
            unlabeled_dl,
            optimizer,
            param["gamma1"],
            param["gamma2"],
            weight=False,
        )
        test_loss, test_acc, test_metrics = test_VGAE(device, model, test_dl)
    elif param["model"] == "DIVA":
        train_loss, train_acc, train_metrics = train_DIVA(
            device,
            model,
            data,
            optimizer,
            labeled_train_indices,
            all_train_indices,
            param["beta_klzd"],
            param["beta_klzx"],
            param["beta_klzy"],
            param["beta_d"],
            param["beta_y"],
            param["beta_recon"],
            weight=False,
        )
        test_loss, test_acc, test_metrics = test_DIVA(device, model, data, test_indices)
    # elif param["model"] == "VGAETS":
    #     labeled_dl, unlabeled_dl, test_dl = data
    #     train_loss, train_acc, train_metrics = train_VGAETS(
    #         device, model, labeled_dl, unlabeled_dl, optimizer,
    #         param["gamma1"], param["gamma2"], weight=False
    #     )
    #     test_loss, test_acc, test_metrics = test_VGAETS(
    #         device, model, test_dl
    #     )
    else:
        raise TypeError("Invalid model of type {}".format(param["model"]))
    train_metrics["accuracy"] = train_acc
    test_metrics["accuracy"] = test_acc
    return train_loss, train_metrics, test_loss, test_metrics


def verbose_info(epoch, train_loss, test_loss, train_acc, test_acc, best_test_acc):
    return (
        "Train Acc: {:.4f}, Test Acc: {:.4f}, Best Test Acc: {:.4f}, "
        "Train Loss: {:.4f}, Test Loss: {:.4f}".format(
            train_acc, test_acc, best_test_acc, train_loss, test_loss
        )
    )


@on_error(({}, {}), True)
def experiment(args, param, model_dir):
    seed_torch()
    device = get_device(args.gpu)
    verbose = args.verbose

    start = time.time()
    (data, labeled_train_indices, all_train_indices, test_indices) = load_data(param)
    model, optimizer = load_model(param, data)

    best_epoch = 0
    best_loss = np.inf
    acc_loss = np.inf
    best_metrics = {}
    best_model = None
    patience = param["patience"]
    cur_patience = 0
    pbar = get_pbar(param["epochs"], verbose)

    train_losses = []
    test_losses = []
    train_f1_scores = []
    test_f1_scores = []
    train_accuracies = []
    test_accuracies = []

    for epoch in pbar:
        train_loss, train_metrics, test_loss, test_metrics = train_test_step(
            param,
            device,
            model,
            data,
            optimizer,
            labeled_train_indices,
            all_train_indices,
            test_indices,
        )

        """
        save priority
        1. accuracy
        2. f1
        3. loss
        """
        save = (
            (len(best_metrics) == 0)
            or (test_metrics["accuracy"] > best_metrics["accuracy"])
            or (
                test_metrics["accuracy"] == best_metrics["accuracy"]
                and (
                    test_metrics["f1"] > best_metrics["f1"]
                    or (
                        test_metrics["f1"] == best_metrics["f1"]
                        and test_loss < acc_loss
                    )
                )
            )
        )
        if test_loss < best_loss:
            best_loss = test_loss
        if save:
            best_epoch = epoch
            acc_loss = test_loss
            best_metrics = test_metrics.copy()
            if param["save_model"]:
                best_model = copy.deepcopy(model.state_dict())
                model_time = int(time.time())
            cur_patience = 0
        else:
            cur_patience += 1

        if verbose:
            pbar.set_postfix_str(
                verbose_info(
                    epoch,
                    train_loss,
                    test_loss,
                    train_metrics["accuracy"],
                    test_metrics["accuracy"],
                    best_metrics["accuracy"],
                )
            )
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_f1_scores.append(train_metrics["f1"])
        test_f1_scores.append(test_metrics["f1"])
        train_accuracies.append(train_metrics["accuracy"])
        test_accuracies.append(test_metrics["accuracy"])

        # early stopping
        if cur_patience == patience:
            break

    if param["save_model"] and best_model is not None:
        mkdir(model_dir)
        model_name = "{}.pt".format(model_time)
        model_path = os.path.abspath(os.path.join(model_dir, model_name))
        torch.save(best_model, model_path)
    else:
        model_path = None

    end = time.time()
    set_experiment_param(
        param,
        time=end - start,
        device=args.gpu,
        model_path=model_path,
        best_epoch=best_epoch,
        loss=best_loss,
        acc_loss=acc_loss,
        **best_metrics
    )
    print(param)

    training_curve = {
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "train_f1_scores": train_f1_scores,
        "test_f1_scores": test_f1_scores,
        "train_losses": train_losses,
        "test_losses": test_losses,
    }
    training_curve.update(param)
    return param, training_curve


def main(args):
    from itertools import product

    script_name = os.path.splitext(os.path.basename(__file__))[0]

    if "all" in args.site:
        sites = ["NI", "NYU", "OHSU", "PKU"]
    else:
        sites = [(x.upper() if x != "None" else None) for x in args.site]
    print("SITES:", sites)

    models = args.model
    print("MODELS:", models)

    ssl = args.ssl
    harmonized = args.harmonize
    SEED = 10

    for site, model in product(sites, models):
        experiment_name = "{}_{}".format(script_name, int(time.time()))
        exp_dir = os.path.join(args.exp_dir, experiment_name)
        model_dir = os.path.join(exp_dir, "models")
        print("Experiment result: {}".format(exp_dir))
        res = []
        curves = []
        save_model = True if site in ["NYU", "PKU", None] else False

        for seed, fold in product(range(SEED), range(5)):

            print("===================")
            print("EXPERIMENT SETTINGS")
            print("===================")
            print("SEED: {}".format(seed))
            print("FOLD: {}".format(fold))
            print("SSL: {}".format(ssl))
            print("HARMONIZED: {}".format(harmonized))
            print("SITE: {}".format(site))

            if model == "GCN":
                param = get_experiment_param(
                    model="GCN",
                    hidden=150,
                    emb1=50,
                    emb2=30,
                    K=3,
                    seed=seed,
                    fold=fold,
                    ssl=ssl,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.001,
                    test=False,
                    harmonized=harmonized,
                    epochs=1000,
                )
            elif model == "FFN":
                param = get_experiment_param(
                    model="FFN",
                    L1=150,
                    L2=50,
                    L3=30,
                    gamma_lap=0,
                    seed=seed,
                    fold=fold,
                    ssl=ssl,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.001,
                    test=False,
                    harmonized=harmonized,
                    epochs=1000,
                )
            elif model == "AE":
                param = get_experiment_param(
                    model="AE",
                    L1=300,
                    L2=50,
                    emb=150,
                    L3=30,
                    gamma=1e-3,
                    seed=seed,
                    fold=fold,
                    ssl=ssl,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.001,
                    test=False,
                    harmonized=harmonized,
                    epochs=1000,
                )
            elif model == "VAE":
                param = get_experiment_param(
                    model="VAE",
                    L1=300,
                    L2=50,
                    emb=150,
                    L3=30,
                    gamma1=1e-5,
                    gamma2=1e-3,
                    seed=seed,
                    fold=fold,
                    ssl=ssl,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.001,
                    test=False,
                    harmonized=harmonized,
                    epochs=1000,
                )
            elif model == "VAECH":
                param = get_experiment_param(
                    model="VAECH",
                    L1=300,
                    L2=50,
                    emb=150,
                    L3=30,
                    gamma1=1e-5,
                    gamma2=1e-3,
                    seed=seed,
                    fold=fold,
                    ssl=ssl,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.001,
                    test=False,
                    harmonized=harmonized,
                    epochs=1000,
                )
            elif model == "VAESDR":
                param = get_experiment_param(
                    model="VAESDR",
                    L1=300,
                    emb=150,
                    gamma1=1e-5,
                    gamma2=1e-3,
                    gamma3=1e-3,
                    gamma4=0.3,
                    gamma5=0,
                    seed=seed,
                    fold=fold,
                    ssl=ssl,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.001,
                    test=False,
                    harmonized=harmonized,
                    epochs=1000,
                )
            elif model == "VAESDR0":
                param = get_experiment_param(
                    model="VAESDR0",
                    L1=300,
                    emb=150,
                    gamma1=1e-5,
                    gamma2=1e-3,
                    gamma3=1e-3,
                    gamma4=0,
                    gamma5=0,
                    seed=seed,
                    fold=fold,
                    ssl=ssl,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.001,
                    test=False,
                    harmonized=harmonized,
                    epochs=1000,
                )
            elif model == "VAESDR1":
                param = get_experiment_param(
                    model="VAESDR1",
                    L1=300,
                    emb=150,
                    gamma1=1e-5,
                    gamma2=1e-3,
                    gamma3=1e-3,
                    gamma4=0.3,
                    gamma5=1,
                    seed=seed,
                    fold=fold,
                    ssl=ssl,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.001,
                    test=False,
                    harmonized=harmonized,
                    epochs=1000,
                )
            elif model == "VGAE":
                param = get_experiment_param(
                    model="VGAE",
                    emb1=300,
                    emb2=100,
                    L1=50,
                    gamma1=1e-5,
                    gamma2=5e-6,
                    num_process=10,
                    seed=seed,
                    fold=fold,
                    ssl=ssl,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.001,
                    test=False,
                    harmonized=harmonized,
                    epochs=1000,
                    patience=300,
                )
            # elif model == "VGAETS":
            #     param = get_experiment_param(
            #         model="VGAETS",
            #         tsemb=500,
            #         emb1=300,
            #         emb2=100,
            #         L1=50,
            #         bidirectional=True,
            #         gamma1=1e-5,
            #         gamma2=5e-6,
            #         num_process=10,
            #         seed=seed,
            #         fold=fold,
            #         ssl=ssl,
            #         save_model=save_model,
            #         site=site,
            #         lr=0.0001,
            #         l2_reg=0.001,
            #         test=False,
            #         harmonized=harmonized,
            #         epochs=1000,
            #         patience=300,
            #     )
            elif model == "GNN":
                param = get_experiment_param(
                    model="GNN",
                    emb1=300,
                    emb2=100,
                    L1=50,
                    num_process=10,
                    seed=seed,
                    fold=fold,
                    ssl=ssl,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.001,
                    test=False,
                    harmonized=harmonized,
                    epochs=1000,
                )
            elif model == "DIVA":
                param = get_experiment_param(
                    model="DIVA",
                    hidden1=150,
                    emb=50,
                    hidden2=30,
                    beta_klzd=1,
                    beta_klzx=1,
                    beta_klzy=1,
                    beta_d=1,
                    beta_y=1,
                    beta_recon=3e-6,
                    seed=seed,
                    fold=fold,
                    ssl=ssl,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.001,
                    test=False,
                    harmonized=harmonized,
                    epochs=500,
                )
            else:
                raise NotImplementedError("{} not implemented".format(args.model))

            exp_res, training_curve = experiment(args, param, model_dir)
            res.append(exp_res)
            curves.append(training_curve)

            df = pd.DataFrame(res).dropna(how="all")
            if df.empty:
                continue
            mkdir(exp_dir)
            res_path = os.path.join(exp_dir, "{}.csv".format(experiment_name))
            df.to_csv(res_path, index=False)

            curves_path = os.path.join(exp_dir, "{}.json".format(experiment_name))
            with open(curves_path, "w") as f:
                json.dump(curves, f, indent=4, sort_keys=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=-1, help="gpu id (0, 1, 2, 3) or cpu (-1)"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ssl", action="store_true")
    parser.add_argument("--harmonize", action="store_true")
    parser.add_argument("--model", default=["FFN"], nargs="+")
    parser.add_argument("--site", default=["None"], nargs="+")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=EXPERIMENT_DIR,
        help="directory to save experiment results",
    )
    args = parser.parse_args()
    main(args)
