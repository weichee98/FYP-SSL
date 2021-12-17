import os
import copy
import time
import json
import argparse
import torch
import numpy as np
import pandas as pd

from ABIDE import *
from config import EXPERIMENT_DIR
from utils import *
from models import *
from data import *
from utils.distribution import SiteDistribution


def get_experiment_param(
    model="GCN",
    seed=0,
    fold=0,
    epochs=1000,
    patience=1000,
    ssl_group=None,
    test=True,
    save_model=True,
    site="NYU",
    harmonized=False,
    lr=0.0001,
    l2_reg=0.001,
    n_ssl=None,
    **kwargs
):
    """
    kwargs:
    1. GCN model (hidden, emb1, emb2, K)
    2. FFN model (L1, L2, L3, gamma_lap)
    3. AE model (L1, L2, L3, emb, gamma)
    4. VAE model (L1, L2, L3, emb, gamma1, gamma2)
    5. VGAE model (emb1, emb2, L1, gamma1, gamma2, num_process)
    6. DIVA model (emb, hidden1, hidden2, beta_klzd, beta_klzx, 
                   beta_klzy, beta_d, beta_y, beta_recon)
    7. VGAETS model (embts, emb1, emb2, L1, gamma1, gamma2, 
                    num_process)
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
    param["ssl_group"] = ssl_group
    param["n_ssl"] = n_ssl
    param["test"] = test
    param["harmonized"] = harmonized
    param["save_model"] = save_model
    for k, v in kwargs.items():
        param[k] = v
    return param


def set_experiment_param(
    param, model_path, time, device, best_epoch, acc, loss, **kwargs
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
    X, Y, X_ts = load_data_fmri(harmonized=param["harmonized"], time_series=True)
    splits = get_splits(site_id=param["site"], test=param["test"])
    ages, genders = get_ages_and_genders()
    sites = get_sites()

    seed = param["seed"]
    fold = param["fold"]
    if param["test"]:
        test_indices = splits[seed][0]
        train_indices, val_indices = splits[seed][1][fold]
        labeled_train_indices = np.concatenate([train_indices, val_indices], axis=0)
    else:
        labeled_train_indices, test_indices = splits[seed][1][fold]
    param["baseline_accuracy"] = np.mean(Y[test_indices], axis=0).max()

    if param["model"] == "GCN":
        (data, labeled_train_indices, all_train_indices, test_indices) = load_GCN_data(
            X,
            Y,
            ages,
            genders,
            param["ssl_group"],
            labeled_train_indices,
            test_indices,
            sites,
            param.get("n_ssl", None),
        )
    elif param["model"] == "FFN":
        (data, labeled_train_indices, all_train_indices, test_indices) = load_FFN_data(
            X,
            Y,
            ages,
            genders,
            param["ssl_group"],
            labeled_train_indices,
            test_indices,
            sites,
            param.get("n_ssl", None),
        )
    elif param["model"] == "AE" or param["model"] == "VAE":
        (data, labeled_train_indices, all_train_indices, test_indices) = load_AE_data(
            X,
            Y,
            param["ssl_group"],
            labeled_train_indices,
            test_indices,
            sites,
            param.get("n_ssl", None),
        )
    elif param["model"] == "VGAE":
        (data, labeled_train_indices, all_train_indices, test_indices) = load_GAE_data(
            X,
            Y,
            param["ssl_group"],
            labeled_train_indices,
            test_indices,
            num_process=param.get("num_process", 1),
            verbose=False,
            sites=sites,
            n_ssl=param.get("n_ssl", None),
        )
    elif param["model"] == "DIVA":
        (data, labeled_train_indices, all_train_indices, test_indices) = load_DIVA_data(
            X,
            Y,
            sites,
            param["ssl_group"],
            labeled_train_indices,
            test_indices,
            param.get("n_ssl", None),
        )
    elif param["model"] == "VGAETS":
        (data, labeled_train_indices, all_train_indices, test_indices) = load_GAE_data(
            X,
            Y,
            param["ssl_group"],
            labeled_train_indices,
            test_indices,
            X_ts,
            num_process=param.get("num_process", 1),
            verbose=False,
            sites=sites,
            n_ssl=param.get("n_ssl", None),
        )
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
    elif param["model"] == "VGAETS":
        model = VGAETS(
            tsemb=param["tsemb"],
            emb1=param["emb1"],
            emb2=param["emb2"],
            l1=param["L1"],
            bidirectional=param["bidirectional"],
        )
    else:
        raise TypeError("Invalid model of type {}".format(param["model"]))
    param["model_size"] = count_parameters(model)
    print("MODEL_SIZE: {}".format(param["model_size"]))
    return model


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
        train_loss, train_acc, _ = train_GCN(
            device, model, data, optimizer, labeled_train_indices
        )
        test_loss, test_acc, test_metrics = test_GCN(device, model, data, test_indices)
    elif param["model"] == "FFN":
        train_loss, train_acc, _ = train_FFN(
            device,
            model,
            data,
            optimizer,
            labeled_train_indices,
            all_train_indices,
            param["gamma_lap"],
        )
        test_loss, test_acc, test_metrics = test_FFN(device, model, data, test_indices)
    elif param["model"] == "AE":
        train_loss, train_acc, _ = train_AE(
            device,
            model,
            data,
            optimizer,
            labeled_train_indices,
            all_train_indices,
            param["gamma"],
        )
        test_loss, test_acc, test_metrics = test_AE(device, model, data, test_indices)
    elif param["model"] == "VAE":
        train_loss, train_acc, _ = train_VAE(
            device,
            model,
            data,
            optimizer,
            labeled_train_indices,
            all_train_indices,
            param["gamma1"],
            param["gamma2"],
        )
        test_loss, test_acc, test_metrics = test_VAE(device, model, data, test_indices)
    elif param["model"] == "VGAE":
        labeled_dl, unlabeled_dl, test_dl = data
        train_loss, train_acc, _ = train_VGAE(
            device,
            model,
            labeled_dl,
            unlabeled_dl,
            optimizer,
            param["gamma1"],
            param["gamma2"],
        )
        test_loss, test_acc, test_metrics = test_VGAE(device, model, test_dl)
    elif param["model"] == "DIVA":
        train_loss, train_acc, _ = train_DIVA(
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
        )
        test_loss, test_acc, test_metrics = test_DIVA(device, model, data, test_indices)
    elif param["model"] == "VGAETS":
        labeled_dl, unlabeled_dl, test_dl = data
        train_loss, train_acc, _ = train_VGAETS(
            device,
            model,
            labeled_dl,
            unlabeled_dl,
            optimizer,
            param["gamma1"],
            param["gamma2"],
        )
        test_loss, test_acc, test_metrics = test_VGAETS(device, model, test_dl)
    else:
        raise TypeError("Invalid model of type {}".format(param["model"]))
    return train_loss, train_acc, test_loss, test_acc, test_metrics


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
    model = load_model(param, data)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=param["lr"],
        weight_decay=param["l2_reg"],
    )

    best_epoch = 0
    best_acc = 0
    best_loss = np.inf
    acc_loss = np.inf
    best_metrics = {}
    best_model = None
    patience = param["patience"]
    cur_patience = 0
    pbar = get_pbar(param["epochs"], verbose)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in pbar:
        train_loss, train_acc, test_loss, test_acc, test_metrics = train_test_step(
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
        save = (test_acc > best_acc) or (
            test_acc == best_acc
            and (
                test_metrics["f1"] > best_metrics["f1"]
                or (test_metrics["f1"] == best_metrics["f1"] and test_loss < acc_loss)
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
            cur_patience = 0
        else:
            cur_patience += 1

        if verbose:
            pbar.set_postfix_str(
                verbose_info(
                    epoch, train_loss, test_loss, train_acc, test_acc, best_acc
                )
            )
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

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
        acc=best_acc,
        loss=best_loss,
        acc_loss=acc_loss,
        **best_metrics
    )
    print(param)

    training_curve = {
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "train_losses": train_losses,
        "test_losses": test_losses,
    }
    training_curve.update(param)
    return param, training_curve


def get_ssl_groups(harmonized, site, pre_end=None, reverse=False):
    X, _ = load_data_fmri(harmonized=harmonized)
    sites = get_sites()
    sd = SiteDistribution()
    hm = sd.distribution_heatmap(X, sites, sd.METRIC.HELLINGER, sd.METHOD.KDE)
    ssl_sites = hm[site]
    ssl_sites.pop(site)
    ssl_sites_order = sorted(ssl_sites, key=lambda x: ssl_sites[x], reverse=reverse)

    print(ssl_sites_order)

    if pre_end is None:
        yield None
        for i in range(1, len(ssl_sites_order) + 1):
            yield ssl_sites_order[:i]
    else:
        idx = ssl_sites_order.index(pre_end)
        for i in range(idx + 2, len(ssl_sites_order) + 1):
            yield ssl_sites_order[:i]


def get_n_ssl(site, pre_end=None):
    sites = get_sites()
    total = int(np.sum(sites != site))
    start_n = 0 if pre_end is None else pre_end + 50
    all_n = list(range(start_n, total, 50)) + [total]
    print(all_n)
    for i in all_n:
        yield i


def main(args):
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    site = "NYU"
    harmonized = args.harmonize
    SEED = 10

    # ssl_group = True
    # for n_ssl in get_n_ssl(site):
    n_ssl = None
    for ssl_group in get_ssl_groups(harmonized, site, reverse=True):

        experiment_name = "{}_{}".format(script_name, int(time.time()))
        exp_dir = os.path.join(args.exp_dir, experiment_name)
        model_dir = os.path.join(exp_dir, "models")
        print("Experiment result: {}".format(exp_dir))
        res = []
        curves = []

        for seed in range(SEED):

            for fold in range(5):

                print("===================")
                print("EXPERIMENT SETTINGS")
                print("===================")
                print("SEED: {}".format(seed))
                print("FOLD: {}".format(fold))
                print("SSL GROUP: {}".format(ssl_group))
                print("HARMONIZED: {}".format(harmonized))
                print("SITE: {}".format(site))

                # param = get_experiment_param(
                #     model="GCN", hidden=150, emb1=50, emb2=30, K=3,
                #     seed=seed, fold=fold, ssl_group=ssl_group, save_model=False,
                #     site=site, lr=0.00005, l2_reg=0.001, n_ssl=n_ssl,
                #     test=False, harmonized=harmonized, epochs=1000
                # )
                # param = get_experiment_param(
                #     model="FFN", L1=150, L2=50, L3=30, gamma_lap=0,
                #     seed=seed, fold=fold, ssl_group=ssl_group, save_model=False,
                #     site=site, lr=0.00005, l2_reg=0.001, n_ssl=n_ssl,
                #     test=False, harmonized=harmonized, epochs=1000
                # )
                if args.model == "AE":
                    param = get_experiment_param(
                        model="AE",
                        L1=300,
                        L2=50,
                        emb=150,
                        L3=30,
                        gamma=1e-3,
                        seed=seed,
                        fold=fold,
                        ssl_group=ssl_group,
                        save_model=False,
                        site=site,
                        lr=0.0001,
                        l2_reg=0.001,
                        n_ssl=n_ssl,
                        test=False,
                        harmonized=harmonized,
                        epochs=1000,
                    )
                elif args.model == "VAE":
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
                        ssl_group=ssl_group,
                        save_model=False,
                        site=site,
                        lr=0.0001,
                        l2_reg=0.001,
                        n_ssl=n_ssl,
                        test=False,
                        harmonized=harmonized,
                        epochs=1000,
                    )
                # param = get_experiment_param(
                #     model="VGAE", emb1=300, emb2=100, L1=50,
                #     gamma1=1e-5, gamma2=5e-6, num_process=10,
                #     seed=seed, fold=fold, ssl_group=ssl_group, save_model=False,
                #     site=site, lr=0.0001, l2_reg=0.001, n_ssl=n_ssl,
                #     test=False, harmonized=harmonized,
                #     epochs=1000, patience=300
                # )
                # param = get_experiment_param(
                #     model="VGAETS", tsemb=500, emb1=300, emb2=100,
                #     L1=50, bidirectional=True,
                #     gamma1=1e-5, gamma2=5e-6, num_process=10,
                #     seed=seed, fold=fold, ssl_group=ssl_group, save_model=False,
                #     site=site, lr=0.0001, l2_reg=0.001, n_ssl=n_ssl,
                #     test=False, harmonized=harmonized,
                #     epochs=1000, patience=300
                # )
                # param = get_experiment_param(
                #     model="DIVA", hidden1=150, emb=50, hidden2=30,
                #     beta_klzd=1, beta_klzx=1, beta_klzy=1,
                #     beta_d=1, beta_y=1, beta_recon=3e-6,
                #     seed=seed, fold=fold, ssl_group=ssl_group, save_model=False,
                #     site=site, lr=0.0001, l2_reg=0.001, n_ssl=n_ssl,
                #     test=False, harmonized=harmonized, epochs=500
                # )
                exp_res, training_curve = experiment(args, param, model_dir)
                res.append(exp_res)
                curves.append(training_curve)

                df = pd.DataFrame(res).dropna(how="all")
                if not df.empty:
                    mkdir(exp_dir)
                    res_path = os.path.join(exp_dir, "{}.csv".format(experiment_name))
                    df.to_csv(res_path, index=False)

                    curves_path = os.path.join(
                        exp_dir, "{}.json".format(experiment_name)
                    )
                    with open(curves_path, "w") as f:
                        json.dump(curves, f, indent=4, sort_keys=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=-1, help="gpu id (0, 1, 2, 3) or cpu (-1)"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--harmonize", action="store_true")
    parser.add_argument("--model", type=str, default="AE")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=EXPERIMENT_DIR,
        help="directory to save experiment results",
    )
    args = parser.parse_args()
    main(args)
