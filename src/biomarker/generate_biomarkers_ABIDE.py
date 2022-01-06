import os
import sys
import time
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_gpu
from models import *
from ABIDE import *
from utils.data import *
from biomarker.visualize import ABIDEBiomarkersVisualizer


score_matrix_filename = "model_mean_score.npy"
score_matrix_meta = "model_mean_score_meta.csv"


def load_model(param, data):  # use this instead
    if param["model"] == "FFN":
        model = FFN(
            input_size=data.x.size(1),
            l1=int(param["L1"]),
            l2=int(param["L2"]),
            l3=int(param["L3"]),
        )
    elif param["model"] == "GCN":
        model = ChebGCN(
            input_size=data.x.size(1),
            hidden=int(param["hidden"]),
            emb1=int(param["emb1"]),
            emb2=int(param["emb2"]),
            K=int(param["K"]),
        )
    elif param["model"] == "AE":
        model = AE(
            input_size=data.x.size(1),
            l1=int(param["L1"]),
            l2=int(param["L2"]),
            l3=int(param["L3"]),
            emb_size=int(param["emb"]),
        )
    elif param["model"] == "VAE":
        model = VAE(
            input_size=data.x.size(1),
            l1=int(param["L1"]),
            l2=int(param["L2"]),
            l3=int(param["L3"]),
            emb_size=int(param["emb"]),
        )
    elif "VAESDRII" in param["model"]:
        model = VAESDRII(
            input_size=data.x.size(1),
            l1=int(param["L1"]),
            emb_size=int(param["emb"]),
            num_site=data.d.unique().size(0),
        )
    elif "VAESDR" in param["model"]:
        model = VAESDR(
            input_size=data.x.size(1),
            l1=int(param["L1"]),
            emb_size=int(param["emb"]),
            num_site=data.d.unique().size(0),
        )
    elif param["model"] == "VAECH":
        model = VAECH(
            input_size=data.x.size(1),
            l1=int(param["L1"]),
            l2=int(param["L2"]),
            l3=int(param["L3"]),
            emb_size=int(param["emb"]),
            num_sites=data.d.size(1),
        )
    elif param["model"] == "VAEVCH":
        model = VAEVCH(
            input_size=data.x.size(1),
            l1=int(param["L1"]),
            l2=int(param["L2"]),
            l3=int(param["L3"]),
            emb_size=int(param["emb"]),
            num_sites=data.d.size(1),
        )
    elif param["model"] == "GNN":
        batch = data[0]
        model = GNN(
            input_size=batch.x.size(1),
            emb1=int(param["emb1"]),
            emb2=int(param["emb2"]),
            l1=int(param["L1"]),
        )
    elif param["model"] == "VGAE":
        batch = data[0]
        model = VGAE(
            input_size=batch.x.size(1),
            emb1=int(param["emb1"]),
            emb2=int(param["emb2"]),
            l1=int(param["L1"]),
        )
    else:
        raise TypeError("Invalid model of type {}".format(param["model"]))

    param["model_size"] = count_parameters(model)
    model.load_state_dict(
        torch.load(param["model_path"], map_location=torch.device("cpu"))
    )
    return model


def load_data(param, X, X_harmonized, Y, ages, genders, sites):
    X_ = X_harmonized if param["harmonized"] else X
    if param["model"] == "GCN":
        X_flattened = corr_mx_flatten(X_)
        A = get_pop_A(X_flattened, ages, genders)
        data = make_population_graph(X_flattened, A, Y)
    elif param["model"] in ["FFN", "VAE", "AE"]:
        X_flattened = corr_mx_flatten(X_)
        data = make_dataset(X_flattened, Y)
    elif param["model"] in ["DIVA"] or "VAESDR" in param["model"]:
        X_flattened = corr_mx_flatten(X_)
        data = make_dataset(X_flattened, Y, sites)
    elif param["model"] in ["VAECH", "VAEVCH"]:
        X_flattened = corr_mx_flatten(X_)
        age = np.expand_dims(ages, axis=1)
        gender = np.eye(2)[genders]
        data = make_dataset(
            X_flattened,
            Y,
            sites,
            age=torch.tensor(age).type(torch.get_default_dtype()),
            gender=torch.tensor(gender).type(torch.get_default_dtype()),
        )
        data.d = torch.eye(data.d.unique().size(0))[data.d].type(
            torch.get_default_dtype()
        )
    elif param["model"] in ["GNN", "VGAE"]:
        data = make_graph_dataset(X_, Y, X_ts=None, num_process=10, verbose=False)
    else:
        raise NotImplementedError("invalid model {}".format(param["model"]))
    return data


def get_score_path(param):
    model_path = os.path.abspath(param["model_path"])
    biomarkers_dir = os.path.join(
        os.path.dirname(os.path.dirname(model_path)), "score_matrices"
    )
    try:
        os.makedirs(biomarkers_dir)
    except FileExistsError:
        pass
    basename = os.path.basename(model_path).replace(".pt", "")
    score_path = os.path.join(biomarkers_dir, "{}.npy".format(basename))
    return score_path


def model_score(param, X, X_harmonized, Y, ages, genders, sites, force):
    seed_torch()
    score_path = get_score_path(param)
    if os.path.exists(score_path) and not force:
        scores = np.load(score_path, allow_pickle=True)
    else:
        data = load_data(param, X, X_harmonized, Y, ages, genders, sites)
        model = load_model(param, data)
        scores = model.saliency_score(data)
        scores = scores.mean(axis=0)
        np.save(score_path, scores)
    param["max_score"] = scores.max()
    param["min_score"] = scores.min()
    param["mean_score"] = scores.mean()
    param["std_score"] = scores.std()
    return scores, param


def generate_score_matrices(input_path, output_dir, n_jobs, force):
    model_path_csv = pd.read_csv(input_path)
    model_path_csv = model_path_csv.dropna(subset=["model_path"]).reset_index()

    X, Y = load_data_fmri(harmonized=False)
    X_harmonized, _ = load_data_fmri(harmonized=True)
    Y = Y.argmax(axis=1)
    ages, genders = get_ages_and_genders()
    sites = get_sites()

    results = Parallel(n_jobs=n_jobs)(
        delayed(model_score)(param, X, X_harmonized, Y, ages, genders, sites, force)
        for param in tqdm(model_path_csv.to_dict("records"), ncols=60)
    )

    all_scores, all_params = zip(*results)
    all_scores = np.array(all_scores)
    output_csv_path = os.path.join(output_dir, score_matrix_meta)
    params_df = pd.DataFrame(all_params)
    params_df.to_csv(output_csv_path, index=False)
    return model_path_csv, all_scores


def get_name(param, gcol):
    path = []
    for c in gcol:
        if c not in param:
            continue
        k = param[c]
        if c == "ssl":
            if k:
                path.append("ssl")
        elif c == "ssl_group":
            k = eval(k) if isinstance(k, str) else k
            if isinstance(k, (list, tuple)):
                path += list(map(str.lower, map(str, k)))
            else:
                path.append(str(k).lower())
        elif c == "harmonized":
            if k:
                path.append("combat")
        else:
            path.append(str(k).lower())
    return "_".join(path)


def plot_biomarkers(param, score_matrices, viz, gcol, output_dir):
    idx = param["index"]
    matrix = np.mean(score_matrices[idx], axis=0)
    prefix = get_name(param, gcol)
    output_dir = os.path.join(output_dir, prefix)
    viz.plot_connectome(matrix, os.path.join(output_dir, "connectome.png"))
    viz.plot_stat_map(matrix, output_dir, threshold=0.1, vmax=10)
    viz.plot_module_importance_boxplot(matrix, os.path.join(output_dir, "boxplot.png"))
    viz.plot_complete_score_matrix(matrix, os.path.join(output_dir, "conn_mat.png"))
    viz.plot_module_sensitivity_map(
        matrix, os.path.join(output_dir, "msm.png"), vmax=10
    )


def visualize_biomarkers(df, score_matrices, output_dir, n_jobs):
    assert len(df) == len(score_matrices)

    gcol = ["site", "model", "harmonized", "ssl", "ssl_group", "n_ssl"]
    gcol = [col for col in gcol if col in df.columns and len(df[col].unique()) > 1]

    df["index"] = range(len(df))
    df = df.groupby(gcol)["index"].apply(list).reset_index()

    viz = ABIDEBiomarkersVisualizer()
    Parallel(n_jobs=n_jobs)(
        delayed(plot_biomarkers)(param, score_matrices, viz, gcol, output_dir)
        for param in tqdm(df.to_dict("records"), ncols=60)
    )


def main(args):
    input_path = os.path.abspath(args.input)
    output_dir = os.path.join(os.path.dirname(input_path), "biomarkers")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_jobs = args.worker
    force = args.force
    df, score_matrices = generate_score_matrices(input_path, output_dir, n_jobs, force)
    visualize_biomarkers(df, score_matrices, output_dir, n_jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="path to csv containing model_path"
    )
    parser.add_argument(
        "--worker", type=int, default=1, help="number of workers to run in parallel"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args)
