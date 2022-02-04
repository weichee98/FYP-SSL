import os
import copy
import time
import json
import argparse
import torch
import numpy as np
import pandas as pd

from config import EXPERIMENT_DIR
from utils import *
from models import ASDSAENet, GAEFCNN, count_parameters
from data import *


def get_experiment_param(
    dataset,
    model="ASDSAENet",
    seed=0,
    fold=0,
    epochs=1000,
    patience=1000,
    test=True,
    save_model=True,
    site="NYU",
    lr=0.0001,
    l2_reg=0.001,
    **kwargs
):
    """
    kwargs:
    1. ASDSAENet model (emb, L1, L2, beta, p)
    2. ASDSAENet1 model (emb, L1, L2, beta, p, mask_ratio)
    3. GAE-FCNN model (emb1, emb2, tau, l1, l2, l3)
    3. VGAE-FCNN model (emb1, emb2, tau, l1, l2, l3)
    3. GCN-FCNN model (emb1, emb2, tau, l1, l2, l3)
    """
    param = dict()
    param["dataset"] = dataset
    param["site"] = site
    param["seed"] = seed
    param["fold"] = fold
    param["epochs"] = epochs
    param["patience"] = patience
    param["model"] = model
    param["lr"] = lr
    param["l2_reg"] = l2_reg
    param["ssl"] = False
    param["test"] = test
    param["harmonized"] = False
    param["save_model"] = save_model
    for k, v in kwargs.items():
        param[k] = v
    return param


def set_experiment_param(
    param,
    model_path,
    time,
    device,
    best_clf_epoch,
    best_ae_epoch,
    acc,
    clf_loss,
    ae_loss,
    **kwargs
):
    param["accuracy"] = acc
    param["loss"] = clf_loss
    param["ae_loss"] = ae_loss
    for k, v in kwargs.items():
        param[k] = v
    param["best_ae_epoch"] = best_ae_epoch
    param["best_clf_epoch"] = best_clf_epoch
    param["model_path"] = model_path
    param["time_taken"] = time
    param["device"] = device


def load_data(param):
    X, Y = load_data_fmri(harmonized=param["harmonized"])
    splits = get_splits(site_id=param["site"], test=param["test"])

    seed = param["seed"]
    fold = param["fold"]
    if param["test"]:
        test_indices = splits[seed][0]
        train_indices, val_indices = splits[seed][1][fold]
        labeled_train_indices = np.concatenate([train_indices, val_indices], axis=0)
    else:
        labeled_train_indices, test_indices = splits[seed][1][fold]
    param["baseline_accuracy"] = np.mean(Y[test_indices], axis=0).max()

    if param["model"] in ["ASDSAENet", "ASDSAENet1"]:
        (data, labeled_train_indices, all_train_indices, test_indices) = load_AE_data(
            X, Y, param["ssl"], labeled_train_indices, test_indices
        )
    elif param["model"] in ["GAE-FCNN", "VGAE-FCNN", "GCN-FCNN"]:
        (data, labeled_train_indices, all_train_indices, test_indices) = load_GAE_data(
            X,
            Y,
            param["ssl"],
            labeled_train_indices,
            test_indices,
            num_process=param.get("num_process", 1),
            verbose=False,
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
    if param["model"] == "ASDSAENet":
        ae = ASDSAENet.SAE(input_size=data.x.size(1), emb=param.get("emb", 4975))
        clf = ASDSAENet.FCNN(
            input_size=param.get("emb", 4975),
            l1=param.get("L1", 2487),
            l2=param.get("L2", 500),
        )
    elif param["model"] == "ASDSAENet1":
        ae = ASDSAENet.MaskedSAE(
            input_size=data.x.size(1),
            emb=param.get("emb", 4975),
            mask_ratio=param.get("masked_ratio", 0.5),
        )
        param["num_input_features"] = ae.num_features
        print("NUM_INPUT_FEATURES: {}".format(ae.num_features))
        clf = ASDSAENet.FCNN(
            input_size=param.get("emb", 4975),
            l1=param.get("L1", 2487),
            l2=param.get("L2", 500),
        )
    elif param["model"] == "GAE-FCNN":
        batch = next(iter(data[0]))
        ae = GAEFCNN.GCNAE(
            input_size=batch.x.size(1),
            emb1=param.get("emb1", 64),
            emb2=param.get("emb2", 16),
            tau=param.get("tau", 0.25),
        )
        input_size = param.get("emb2") or param.get("emb1") or 16
        clf = GAEFCNN.GFCNN(
            input_size=input_size,
            num_nodes=batch.x.size(0),
            l1=param.get("L1", 256),
            l2=param.get("L2", 256),
            l3=param.get("L3", 128),
        )
    elif param["model"] == "VGAE-FCNN":
        batch = next(iter(data[0]))
        ae = GAEFCNN.VGCNAE(
            input_size=batch.x.size(1),
            emb1=param.get("emb1", 64),
            emb2=param.get("emb2", 16),
            tau=param.get("tau", 0.25),
        )
        input_size = param.get("emb2") or param.get("emb1") or 16
        clf = GAEFCNN.GFCNN(
            input_size=input_size,
            num_nodes=batch.x.size(0),
            l1=param.get("L1", 256),
            l2=param.get("L2", 256),
            l3=param.get("L3", 128),
        )
    elif param["model"] == "GCN-FCNN":
        batch = next(iter(data[0]))
        ae = GAEFCNN.GCN(
            input_size=batch.x.size(1),
            emb1=param.get("emb1", 64),
            emb2=param.get("emb2", 16),
            tau=param.get("tau", 0.25),
        )
        input_size = param.get("emb2") or param.get("emb1") or 16
        clf = GAEFCNN.GFCNN(
            input_size=input_size,
            num_nodes=batch.x.size(0),
            l1=param.get("L1", 256),
            l2=param.get("L2", 256),
            l3=param.get("L3", 128),
        )
    else:
        raise TypeError("Invalid model of type {}".format(param["model"]))
    param["model_size"] = count_parameters(ae) + count_parameters(clf)
    print("MODEL_SIZE: {}".format(param["model_size"]))

    model = (ae, clf)
    optimizer = (
        ae.get_optimizer(lr=param.get("lr", 0.0001), lmbda=param.get("l2_reg", 0.0001)),
        clf.get_optimizer(
            lr=param.get("lr", 0.0001), lmbda=param.get("l2_reg", 0.0001)
        ),
    )
    return model, optimizer


def ae_train_test_step(
    param, device, ae, data, ae_optimizer, labeled_train_indices, test_indices,
):
    if param["model"] == "ASDSAENet":
        train_loss = ASDSAENet.train_SAE(
            device,
            ae,
            data,
            ae_optimizer,
            labeled_train_indices,
            beta=param.get("beta", 2),
            p=param.get("p", 0.05),
        )
        test_loss = ASDSAENet.test_SAE(device, ae, data, test_indices)
    elif param["model"] == "ASDSAENet1":
        train_loss = ASDSAENet.train_MaskedSAE(
            device,
            ae,
            data,
            ae_optimizer,
            labeled_train_indices,
            beta=param.get("beta", 2),
            p=param.get("p", 0.05),
        )
        test_loss = ASDSAENet.test_MaskedSAE(device, ae, data, test_indices)
    elif param["model"] in ["GAE-FCNN", "VGAE-FCNN"]:
        train_dl, _, test_dl = data
        train_loss = GAEFCNN.train_GCNAE(device, ae, train_dl, ae_optimizer)
        test_loss = GAEFCNN.test_GCNAE(device, ae, test_dl)
    else:
        raise TypeError("Invalid model of type {}".format(param["model"]))
    return train_loss, test_loss


def clf_train_test_step(
    param, device, clf, ae, data, clf_optimizer, labeled_train_indices, test_indices,
):
    if param["model"] in ["ASDSAENet", "ASDSAENet1"]:
        train_loss, train_acc, _ = ASDSAENet.train_FCNN(
            device, clf, ae, data, clf_optimizer, labeled_train_indices
        )
        test_loss, test_acc, test_metrics = ASDSAENet.test_FCNN(
            device, clf, ae, data, test_indices
        )
    elif param["model"] in ["GAE-FCNN", "VGAE-FCNN"]:
        train_dl, _, test_dl = data
        train_loss, train_acc, _ = GAEFCNN.train_GFCNN(
            device, clf, ae, train_dl, clf_optimizer
        )
        test_loss, test_acc, test_metrics = GAEFCNN.test_GFCNN(device, clf, ae, test_dl)
    elif param["model"] == "GCN-FCNN":
        train_dl, _, test_dl = data
        gcn_optimizer, clf_optimizer = clf_optimizer
        train_loss, train_acc, _ = GAEFCNN.train_GCNFCNN(
            device, ae, clf, train_dl, gcn_optimizer, clf_optimizer
        )
        test_loss, test_acc, test_metrics = GAEFCNN.test_GCNFCNN(
            device, ae, clf, test_dl
        )
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


def clf_training_loop(
    param,
    verbose,
    device,
    model,
    data,
    clf_optimizer,
    labeled_train_indices,
    test_indices,
):
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
    ae, clf = model

    for epoch in pbar:
        train_loss, train_acc, test_loss, test_acc, test_metrics = clf_train_test_step(
            param,
            device,
            clf,
            ae,
            data,
            clf_optimizer,
            labeled_train_indices,
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
            if param["model"] == "GCN-FCNN":
                best_model = (
                    copy.deepcopy(ae.state_dict()),
                    copy.deepcopy(clf.state_dict()),
                )
            else:
                best_model = copy.deepcopy(clf.state_dict())
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

    if param["model"] == "GCN-FCNN":
        ae.load_state_dict(best_model[0])
        clf.load_state_dict(best_model[1])
        best_model = (ae, clf)
    else:
        clf.load_state_dict(best_model)
        best_model = clf
    return dict(
        best_clf_epoch=best_epoch,
        best_clf=best_model,
        best_acc=best_acc,
        best_clf_loss=best_loss,
        acc_loss=acc_loss,
        best_metrics=best_metrics,
        train_clf_losses=train_losses,
        test_clf_losses=test_losses,
        train_accuracies=train_accuracies,
        test_accuracies=test_accuracies,
    )


def ae_training_loop(
    param, verbose, device, ae, data, ae_optimizer, labeled_train_indices, test_indices
):
    best_epoch = 0
    best_loss = np.inf
    best_model = None
    patience = param["patience"]
    cur_patience = 0
    pbar = get_pbar(param["epochs"], verbose)

    train_losses = []
    test_losses = []

    for epoch in pbar:
        train_loss, test_loss = ae_train_test_step(
            param, device, ae, data, ae_optimizer, labeled_train_indices, test_indices,
        )

        """
        save priority
        1. accuracy
        2. f1
        3. loss
        """
        save = test_loss <= best_loss
        if save:
            best_loss = test_loss
            best_epoch = epoch
            best_model = copy.deepcopy(ae.state_dict())
            cur_patience = 0
        else:
            cur_patience += 1

        if verbose:
            pbar.set_postfix_str(verbose_info(epoch, train_loss, test_loss, 0, 0, 0))
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # early stopping
        if cur_patience == patience:
            break

    ae.load_state_dict(best_model)
    return dict(
        best_ae_epoch=best_epoch,
        best_ae=ae,
        best_ae_loss=best_loss,
        train_ae_losses=train_losses,
        test_ae_losses=test_losses,
    )


@on_error(({}, {}), True)
def experiment(args, param, model_dir):
    seed_torch()
    device = get_device(args.gpu)
    verbose = args.verbose

    start = time.time()
    (data, labeled_train_indices, _, test_indices) = load_data(param)
    (ae, clf), (ae_optimizer, clf_optimizer) = load_model(param, data)

    if param["model"] != "GCN-FCNN":
        ae_res = ae_training_loop(
            param,
            verbose,
            device,
            ae,
            data,
            ae_optimizer,
            labeled_train_indices,
            test_indices,
        )
        ae = ae_res["best_ae"]
        clf_res = clf_training_loop(
            param,
            verbose,
            device,
            (ae, clf),
            data,
            clf_optimizer,
            labeled_train_indices,
            test_indices,
        )
        clf = clf_res["best_clf"]
    else:
        ae_res = dict()
        clf_res = clf_training_loop(
            param,
            verbose,
            device,
            (ae, clf),
            data,
            (ae_optimizer, clf_optimizer),
            labeled_train_indices,
            test_indices,
        )
        ae, clf = clf_res["best_clf"]

    if param["model"] in ["ASDSAENet", "ASDSAENet1"]:
        best_model = ASDSAENet.ASDSAENet(ae, clf)
        best_model = best_model.state_dict()
    elif param["model"] in ["GAE-FCNN", "VGAE-FCNN", "GCN-FCNN"]:
        best_model = GAEFCNN.GAEFCNN(ae, clf)
        best_model = best_model.state_dict()
    else:
        raise TypeError("Invalid model of type {}".format(param["model"]))

    if param["save_model"] and best_model is not None:
        mkdir(model_dir)
        model_name = "{}.pt".format(int(time.time()))
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
        best_ae_epoch=ae_res.get("best_ae_epoch"),
        best_clf_epoch=clf_res["best_clf_epoch"],
        acc=clf_res["best_acc"],
        clf_loss=clf_res["best_clf_loss"],
        ae_loss=ae_res.get("best_ae_loss"),
        acc_loss=clf_res["acc_loss"],
        **clf_res["best_metrics"]
    )
    print(param)

    training_curve = {
        "train_accuracies": clf_res["train_accuracies"],
        "test_accuracies": clf_res["test_accuracies"],
        "train_clf_losses": clf_res["train_clf_losses"],
        "test_clf_losses": clf_res["test_clf_losses"],
        "train_ae_losses": ae_res.get("train_ae_losses"),
        "test_ae_losses": ae_res.get("test_ae_losses"),
    }
    training_curve.update(param)
    return param, training_curve


def main(args):
    from itertools import product

    if args.dataset == "ABIDE":
        script_name = "ssl_ABIDE"
    elif args.dataset == "ADHD":
        script_name = "ssl_ADHD"
    else:
        raise NotImplementedError("Dataset {} does not exists".format(args.dataset))
    # script_name = os.path.splitext(os.path.basename(__file__))[0]

    if "all" in args.site:
        if args.dataset == "ABIDE":
            sites = [
                "CALTECH",
                "LEUVEN_1",
                "LEUVEN_2",
                "MAX_MUN",
                "NYU",
                "OHSU",
                "OLIN",
                "PITT",
                "STANFORD",
                "TRINITY",
                "UCLA_1",
                "UCLA_2",
                "UM_1",
                "UM_2",
                "USM",
                "YALE",
            ]
        elif args.dataset == "ADHD":
            sites = ["NI", "NYU", "OHSU", "PKU"]
        else:
            raise NotImplementedError("Dataset {} does not exists".format(args.dataset))
    else:
        sites = [(x.upper() if x != "None" else None) for x in args.site]
    print("SITES:", sites)

    models = args.model
    print("MODELS:", models)
    SEED = 10

    for site, model in product(sites, models):
        experiment_name = "{}_{}".format(script_name, int(time.time()))
        exp_dir = os.path.join(args.exp_dir, experiment_name)
        model_dir = os.path.join(exp_dir, "models")
        print("Experiment result: {}".format(exp_dir))
        res = []
        curves = []

        for seed, fold in product(range(SEED), range(5)):

            print("===================")
            print("EXPERIMENT SETTINGS")
            print("===================")
            print("SEED: {}".format(seed))
            print("FOLD: {}".format(fold))
            print("SITE: {}".format(site))
            save_model = False

            if model == "ASDSAENet":
                param = get_experiment_param(
                    args.dataset,
                    model="ASDSAENet",
                    emb=4975,
                    l1=2486,
                    l2=500,
                    p=0.05,
                    beta=2,
                    seed=seed,
                    fold=fold,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.0001,
                    test=False,
                    epochs=5000,
                )
            elif model == "ASDSAENet1":
                param = get_experiment_param(
                    args.dataset,
                    model="ASDSAENet1",
                    emb=4975,
                    l1=2486,
                    l2=500,
                    p=0.05,
                    beta=2,
                    masked_ratio=0.5,
                    seed=seed,
                    fold=fold,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.0001,
                    test=False,
                    epochs=5000,
                )
            elif model == "GCN-FCNN":
                param = get_experiment_param(
                    args.dataset,
                    model="GCN-FCNN",
                    emb1=94,
                    emb2=0,
                    l1=128,
                    l2=264,
                    l3=0,
                    tau=0.25,
                    seed=seed,
                    fold=fold,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.0001,
                    test=False,
                    epochs=400,
                    num_process=10,
                )
            elif model == "GAE-FCNN":
                param = get_experiment_param(
                    args.dataset,
                    model="GAE-FCNN",
                    emb1=64,
                    emb2=16,
                    l1=256,
                    l2=256,
                    l3=128,
                    tau=0.25,
                    seed=seed,
                    fold=fold,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.0001,
                    test=False,
                    epochs=400,
                    num_process=10,
                )
            elif model == "VGAE-FCNN":
                param = get_experiment_param(
                    args.dataset,
                    model="VGAE-FCNN",
                    emb1=64,
                    emb2=16,
                    l1=256,
                    l2=256,
                    l3=128,
                    tau=0.25,
                    seed=seed,
                    fold=fold,
                    save_model=save_model,
                    site=site,
                    lr=0.0001,
                    l2_reg=0.0001,
                    test=False,
                    epochs=400,
                    num_process=10,
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
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=EXPERIMENT_DIR,
        help="directory to save experiment results",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default=["ASDSAENet"], nargs="+")
    parser.add_argument("--site", default=["None"], nargs="+")
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == "ABIDE":
        from ABIDE import *
    elif dataset == "ADHD":
        from ADHD import *
    else:
        raise NotImplementedError("Dataset {} does not exists".format(dataset))

    main(args)
