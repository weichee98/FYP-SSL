import os
import json
import time
import argparse
import torch
import numpy as np

from ABIDE import *
from config import EXPERIMENT_DIR
from utils import *
from models import *
from utils.data import *
from utils.distribution import SiteDistribution


def get_experiment_param(
        model="FFN", seed=0, fold=0, epochs=1000,
        test=True, harmonized=False,
        lr=0.0001, l2_reg=0.001,
        **kwargs
    ):
    """
    kwargs:
    1. FFN model (L1, L2, L3, gamma_lap)
    2. AE model (L1, L2, L3, emb, gamma)
    3. VAE model (L1, L2, L3, emb, gamma1, gamma2)
    4. VGAE model (hidden, emb1, emb2, L1, gamma1, gamma2, num_process, batch_size)
    5. GNN model (hidden, emb1, emb2, L1, gamma, num_process, batch_size)
    """
    param = dict()
    param["seed"] = seed
    param["fold"] = fold
    param["epochs"] = epochs
    param["model"] = model
    param["lr"] = lr
    param["l2_reg"] = l2_reg
    param["test"] = test
    param["harmonized"] = harmonized
    for k, v in kwargs.items():
        param[k] = v
    return param

def set_experiment_param(
        param, time, device, training_group, test_results,
        train_indices, test_indices
    ):
    param["time_taken"] = time
    param["device"] = device
    param["training_group"] = training_group
    param["num_train"] = len(train_indices[training_group])
    param["test_results"] = list()
    for group, res in test_results.items():
        res["testing_group"] = group
        res["num_test"] = len(test_indices[group])
        param["test_results"].append(res)

def load_input(param, groups):
    seed = param["seed"]
    fold = param["fold"]
    X, Y = load_data_fmri(harmonized=param["harmonized"])
    splits = dict()
    for site in groups:
        try:
            split = get_splits(site_id=site, test=param["test"])
        except FileNotFoundError:
            continue
        if param["test"]:
            test_indices = split[seed][0]
            train_indices, val_indices = split[seed][1][fold]
            train_indices = np.concatenate(
                [train_indices, val_indices], axis=0
            )
        else:
            train_indices, test_indices = split[seed][1][fold]
        splits[site] = {
            "train_indices": train_indices,
            "test_indices": test_indices
        }
    return X, Y, splits

def get_train_test_indices(groups, splits, target_group):
    """
    groups: dict, key = site, value = group_id
    """
    group_names = dict((group, list()) for group in groups.values())
    train_indices = {target_group: list()}
    test_indices = dict((group, list()) for group in groups.values())
    
    for site, group in groups.items():
        train_idx = splits[site]["train_indices"]
        test_idx = splits[site]["test_indices"]
        if group == target_group:
            train_indices[group].append(train_idx)
            test_indices[group].append(test_idx)
        else:
            train_test = np.concatenate(
                [train_idx, test_idx], 
                axis=0
            )
            test_indices[group].append(train_test)
        group_names[group].append(site)

    group_names = dict(
        (group, tuple(sorted(name))) 
        for group, name in group_names.items()
    )
    train_indices = dict(
        (group_names[group], np.concatenate(train_idx, axis=0)) 
        for group, train_idx in train_indices.items()
    )
    test_indices = dict(
        (group_names[group], np.concatenate(test_idx, axis=0))
        for group, test_idx in test_indices.items()
    )

    print("TRAIN:")
    for i, (group, idx) in enumerate(train_indices.items(), start=1):
        print("{}. GROUP {}: {}".format(i, group, len(idx)))
    print("TEST")
    for i, (group, idx) in enumerate(test_indices.items(), start=1):
        print("{}. GROUP {}: {}".format(i, group, len(idx)))
    return train_indices, test_indices

def load_data(param, X, Y, train_indices, test_indices):
    assert len(train_indices) == 1
    train_indices = list(train_indices.items())[0][1]

    if param["model"] in ["FFN", "AE", "VAE"]:
        X_flattened = corr_mx_flatten(X)
        data = make_dataset(X_flattened, Y.argmax(axis=1))
        
    elif param["model"] in ["GNN", "VGAE"]:
        batch_size = param["batch_size"]
        data = make_graph_dataset(
            X, Y.argmax(axis=1), param.get("num_process", 1), False
        )
        train_dl = torch_geometric.data.DataLoader(
            dataset=[data[i] for i in train_indices], 
            batch_size=train_indices.shape[0] if batch_size is None else batch_size
        )
        test_dl = dict()
        for test_group, test_idx in test_indices.items():
            test_dl[test_group] = torch_geometric.data.DataLoader(
                dataset=[data[i] for i in test_idx], 
                batch_size=test_idx.shape[0] if batch_size is None else batch_size
            )
        data = (train_dl, test_dl)

    else:
        raise NotImplementedError(
            "No dataloader function implemented for model {}"
            .format(param["model"])
        )
    return data

def load_model(param, data):
    if param["model"] == "FFN":
        model = FFN(
            input_size=data.x.size(1), 
            l1=param["L1"], 
            l2=param["L2"],
            l3=param["L3"]
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
        train_indices, test_indices
    ):
    assert len(train_indices) == 1
    train_results = dict()
    test_results = dict()

    if param["model"] in ["GNN", "VGAE"]:
        train_dl, test_dl = data

    # training
    for group, train_idx in train_indices.items():
        if param["model"] == "FFN":
            train_loss, train_acc = train_FFN(
                device, model, data, optimizer, train_idx,
                None, param["gamma_lap"]
            )
        elif param["model"] == "AE":
            train_loss, train_acc = train_AE(
                device, model, data, optimizer, train_idx,
                None, param["gamma"]
            )
        elif param["model"] == "VAE":
            train_loss, train_acc = train_VAE(
                device, model, data, optimizer, train_idx,
                None, param["gamma1"], param["gamma2"]
            )
        elif param["model"] == "GNN":
            train_loss, train_acc = train_GNN(
                device, model, train_dl, optimizer, param["gamma"]
            )
        elif param["model"] == "VGAE":
            train_loss, train_acc = train_VGAE(
                device, model, train_dl, None, optimizer,
                param["gamma1"], param["gamma2"]
            )
        else:
            raise TypeError(
                "Invalid model of type {}".format(param["model"])
            )
        train_results[group] = dict(
            loss=train_loss,
            accuracy=train_acc
        )

    # testing
    for group, test_idx in test_indices.items():
        if param["model"] == "FFN":
            test_loss, test_acc, test_metrics = test_FFN(
                device, model, data, test_idx
            )
        elif param["model"] == "AE":
            test_loss, test_acc, test_metrics = test_AE(
                device, model, data, test_idx
            )
        elif param["model"] == "VAE":
            test_loss, test_acc, test_metrics = test_VAE(
                device, model, data, test_idx
            )
        elif param["model"] == "GNN":
            test_loss, test_acc, test_metrics = test_GNN(
                device, model, test_dl[group]
            )
        elif param["model"] == "VGAE":
            test_loss, test_acc, test_metrics = test_VGAE(
                device, model, test_dl[group]
            )
        else:
            raise TypeError(
                "Invalid model of type {}".format(param["model"])
            )
        test_results[group] = dict(
            loss=test_loss,
            accuracy=test_acc,
            metrics=test_metrics
        )

    return train_results, test_results

def verbose_info(epoch, train_loss, test_loss, train_acc, test_acc):
    return "Epoch: {:03d}, Train Acc: {:.4f}, Test Acc: {:.4f}, " \
        "Train Loss: {:.4f}, Test Loss: {:.4f}".format(
            epoch, train_acc, test_acc, train_loss, test_loss
        )

@on_error({}, True)
def experiment(args, param, groups, target_group):
    seed_torch()
    device = get_device(args.gpu)
    verbose = args.verbose

    start = time.time()
    X, Y, splits = load_input(param, groups)
    train_indices, test_indices = get_train_test_indices(groups, splits, target_group)
    assert len(train_indices) == 1
    training_group = list(train_indices)[0]
    if len(train_indices[training_group]) < 100:
        return {}

    data = load_data(param, X, Y, train_indices, test_indices)
    model = load_model(param, data)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=param["lr"], weight_decay=param["l2_reg"]
    )

    best_epoch = dict()
    best_acc = dict()
    best_loss = dict()
    acc_loss = dict()
    best_metrics = dict()
    pbar = get_pbar(param["epochs"], verbose)

    for epoch in pbar:
        train_results, test_results = train_test_step(
            param, device, model, data, optimizer, 
            train_indices, test_indices
        )

        assert len(train_results) == 1
        train_results = train_results[training_group]
        train_loss = train_results["loss"]
        train_acc = train_results["accuracy"]

        """
        save priority
        1. accuracy
        2. f1
        3. loss
        """
        all_test_acc = list()
        all_test_loss = list()
        for group, test_res in test_results.items():
            test_acc = test_res["accuracy"]
            test_metrics = test_res["metrics"]
            test_loss = test_res["loss"]
            all_test_acc.append(test_acc)
            all_test_loss.append(test_loss)

            save = (epoch == 1) or (test_acc > best_acc[group]) or (
                test_acc == best_acc[group] and (
                    test_metrics["f1"] > best_metrics[group]["f1"] or (
                        test_metrics["f1"] == best_metrics[group]["f1"] and
                        test_loss < acc_loss[group]
                    )
                )
            )

            if epoch == 1 or test_loss < best_loss[group]:
                best_loss[group] = test_loss
            if save:
                best_epoch[group] = epoch
                best_acc[group] = test_acc
                acc_loss[group] = test_loss
                best_metrics[group] = test_metrics

        if verbose:
            pbar.set_postfix_str(verbose_info(
                epoch, train_loss, np.mean(all_test_loss), train_acc, np.mean(all_test_acc)
            ))

    test_results = dict()
    for group in best_metrics:
        test_results[group] = dict(
            best_epoch=best_epoch[group],
            accuracy=best_acc[group],
            loss=best_loss[group],
            acc_loss=acc_loss[group],
            **best_metrics[group]
        )

    end = time.time()
    set_experiment_param(
        param, time=end - start, device=args.gpu,
        training_group=training_group,
        test_results=test_results,
        train_indices=train_indices,
        test_indices=test_indices
    )
    print(param)
    return param

def get_groups():
    groups = get_labelling_standards()
    groups.pop("CMU")

    group_ids = list(groups.values())
    max_group = max(set(group_ids), key=group_ids.count)
    valid_sites = [site for site in groups if groups[site] == max_group]

    X, _ = load_data_fmri()
    sites = get_sites()
    SD = SiteDistribution()
    valid_idx = np.isin(sites, valid_sites)
    groups = SD.get_site_grouping(X[valid_idx], sites[valid_idx], SD.METRIC.HELLINGER, SD.METHOD.KDE, 0.063)
    return groups

def main(args):
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    groups = get_groups()
    harmonized = True

    for seed in range(10):
        experiment_name = "{}_{}".format(script_name, int(time.time()))
        exp_dir = os.path.join(args.exp_dir, experiment_name) 
        
        print("===================")
        print("EXPERIMENT SETTINGS")
        print("===================")
        print("EXPERIMENT RESULT: {}".format(exp_dir))
        print("HARMONIZED: {}".format(harmonized))
        print("GROUPS: {}".format(groups))

        res = []
        for target_group in sorted(set(groups.values())):
            print("TARGET_GROUP: {}".format(target_group))
            for fold in range(5):
                param = get_experiment_param(
                    model="FFN", L1=150, L2=50, L3=30, gamma_lap=0, 
                    seed=seed, fold=fold, epochs=1000,
                    test=False, harmonized=harmonized,
                    lr=0.0001, l2_reg=0.001,
                )
                # param = get_experiment_param(
                #     model="AE", L1=300, L2=50, emb=150, L3=30, gamma=1e-3, 
                #     seed=seed, fold=fold, epochs=1000,
                #     test=False, harmonized=harmonized,
                #     lr=0.0001, l2_reg=0.001,
                # )
                # param = get_experiment_param(
                #     model="VAE", L1=300, L2=50, emb=150, L3=30, gamma1=3e-5, gamma2=1e-3, 
                #     seed=seed, fold=fold, epochs=1000,
                #     test=False, harmonized=harmonized,
                #     lr=0.0001, l2_reg=0.001,
                # )
                # param = get_experiment_param(
                #     model="VGAE", hidden=300, emb1=150, emb2=50, L1=30,
                #     gamma1=3e-6, gamma2=1e-6, num_process=10, batch_size=10,
                #     seed=seed, fold=fold, epochs=1000,
                #     test=False, harmonized=harmonized,
                #     lr=0.0001, l2_reg=0.001,
                # )
                # param = get_experiment_param(
                #     model="GNN", hidden=300, emb1=150, emb2=50, L1=30,
                #     gamma=0, num_process=10, batch_size=10,
                #     seed=seed, fold=fold, epochs=1000,
                #     test=False, harmonized=harmonized,
                #     lr=0.0001, l2_reg=0.001,
                # )
                exp_res = experiment(args, param, groups, target_group)
                if len(exp_res) > 0:
                    res.append(exp_res)
        
                if len(res) > 0:
                    mkdir(exp_dir)
                    res_path = os.path.join(exp_dir, "{}.json".format(experiment_name))
                    with open(res_path, "w") as f:
                        json.dump({"results": res}, f, sort_keys=True, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1,
                        help='gpu id (0, 1, 2, 3) or cpu (-1)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--exp_dir', type=str, default=EXPERIMENT_DIR,
                        help="directory to save experiment results")
    args = parser.parse_args()
    main(args)