import torch
from typing import Any, Dict
from torch_geometric.data import Data

from data import Dataset
from models.base import ModelBase
from utils.metrics import ClassificationMetrics as CM
from models.EDC_VAE import EDC_VAE
from models.SHRED import SHRED
from models.VAESDR import VAESDR


def load_model(model_name: str, model_params: Dict[str, Any], model_path: str):
    if model_name == "EDC_VAE":
        model = EDC_VAE(**model_params)
    elif model_name == "SHRED":
        model = SHRED(**model_params)
    elif model_name == "VAESDR":
        model = VAESDR(**model_params)
    else:
        raise NotImplementedError
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    return model


def load_data(data_csv_path: str, harmonize: bool = False):
    dataset = Dataset(data_csv_path, "", harmonize)
    return dataset.load_all_data()["data"]


def evaluate_model(model: ModelBase, data: Data):
    x, y = data.x, data.y
    if isinstance(model, SHRED):
        prediction = model.forward(x, data.age, data.gender, data.d)
    else:
        prediction = model.forward(x)
    print("accuracy: {:.5f}".format(CM.accuracy(y, prediction["y"]).item()))
    print("f1: {:.5f}".format(CM.f1_score(y, prediction["y"]).item()))
    print("recall: {:.5f}".format(CM.tpr(y, prediction["y"]).item()))
    print("precision: {:.5f}".format(CM.ppv(y, prediction["y"]).item()))


if __name__ == "__main__":
    data = load_data("../dataset/ABIDE/meta.csv")

    models = [
        (
            "EDC_VAE",
            "../saved_model/EDC_VAE_1645419832.pt",
            dict(
                input_size=34716,
                hidden_size=32,
                emb_size=16,
                clf_hidden_1=0,
                clf_hidden_2=0,
            ),
        ),
        (
            "SHRED",
            "../saved_model/SHRED_1647936853.pt",
            dict(
                input_size=34716,
                hidden_size=32,
                emb_size=32,
                clf_hidden_1=0,
                clf_hidden_2=0,
                num_sites=2,
            ),
        ),
        (
            "VAESDR",
            "../saved_model/VAESDR_1647937049.pt",
            dict(
                input_size=34716,
                hidden_size=32,
                emb_size=16,
                clf_hidden_1=0,
                clf_hidden_2=0,
                num_sites=2,
            ),
        ),
    ]

    for model_name, model_path, model_params in models:
        model = load_model(model_name, model_params, model_path)
        evaluate_model(model, data)
