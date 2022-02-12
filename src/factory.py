from abc import ABC, abstractclassmethod
from dataclasses import dataclass

import os
import sys
from typing import Any, Dict, Tuple, Union
from torch.nn import Module

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import (
    FFN,
    AE_FFN,
    VAE_FFN,
    VAECH,
    VAECH_I,
    VAECH_II,
    VAESDR,
)
from models.ASDSAENet import SAE, MaskedSAE, FCNN
from models.GAEFCNN import GCN_FCNN, GAE, GVAE, GFCNN
from data import DataloaderBase, ModelBaseDataloader, GraphModelBaseDataloader


class FrameworkFactory(ABC):
    @abstractclassmethod
    def load_model(
        cls, model_name: str, model_param: Dict[str, Any]
    ) -> Union[Module, Tuple[Module, Module]]:
        raise NotImplementedError

    @abstractclassmethod
    def load_dataloader(
        cls, model_name: str, dataloader_param: Dict[str, Any]
    ) -> DataloaderBase:
        raise NotImplementedError


class SingleStageFrameworkFactory(FrameworkFactory):
    @dataclass
    class Mapping:
        model_cls: Module
        dataloader_cls: DataloaderBase

    mapping = {
        "FFN": Mapping(FFN, ModelBaseDataloader),
        "AE-FFN": Mapping(AE_FFN, ModelBaseDataloader),
        "VAE-FFN": Mapping(VAE_FFN, ModelBaseDataloader),
        "VAECH": Mapping(VAECH, ModelBaseDataloader),
        "VAECH-I": Mapping(VAECH_I, ModelBaseDataloader),
        "VAECH-II": Mapping(VAECH_II, ModelBaseDataloader),
        "VAESDR": Mapping(VAESDR, ModelBaseDataloader),
        "VAESDR-D": Mapping(VAESDR, ModelBaseDataloader),
        "VAESDR-DS": Mapping(VAESDR, ModelBaseDataloader),
        "VAESDR-W": Mapping(VAESDR, ModelBaseDataloader),
        "VAESDR-DW": Mapping(VAESDR, ModelBaseDataloader),
        "VAESDR-DSW": Mapping(VAESDR, ModelBaseDataloader),
        "GCN-FCNN": Mapping(GCN_FCNN, GraphModelBaseDataloader),
    }

    @classmethod
    def load_model(cls, model_name: str, model_param: Dict[str, Any]) -> Module:
        model_mapping = cls.mapping.get(model_name, None)
        if model_mapping is None:
            raise NotImplementedError(
                "Model {} does not exist".format(model_name)
            )
        return model_mapping.model_cls(**model_param)

    @classmethod
    def load_dataloader(
        cls, model_name: str, dataloader_param: Dict[str, Any]
    ) -> DataloaderBase:
        model_mapping = cls.mapping.get(model_name, None)
        if model_mapping is None:
            raise NotImplementedError(
                "Model {} does not exist".format(model_name)
            )
        return model_mapping.dataloader_cls(**dataloader_param)


class DoubleStageFrameworkFactory(FrameworkFactory):
    @dataclass
    class Mapping:
        model_cls: Tuple[Module, Module]
        dataloader_cls: DataloaderBase

    mapping = {
        "ASDSAENet": Mapping((SAE, FCNN), ModelBaseDataloader),
        "ASDSAENet1": Mapping((MaskedSAE, FCNN), ModelBaseDataloader),
        "GAE-FCNN": Mapping((GAE, GFCNN), GraphModelBaseDataloader),
        "GVAE-FCNN": Mapping((GVAE, GFCNN), GraphModelBaseDataloader),
    }

    @classmethod
    def load_model(
        cls, model_name: str, model_param: Dict[str, Dict[str, Any]]
    ) -> Tuple[Module, Module]:
        model_mapping = cls.mapping.get(model_name, None)
        if model_mapping is None:
            raise NotImplementedError(
                "Model {} does not exist".format(model_name)
            )
        ae_cls, fcnn_cls = model_mapping.model_cls
        ae_param = model_param.get("ae_param", dict())
        fcnn_param = model_param.get("fcnn_param", dict())
        return ae_cls(**ae_param), fcnn_cls(**fcnn_param)

    @classmethod
    def load_dataloader(
        cls, model_name: str, dataloader_param: Dict[str, Any]
    ) -> DataloaderBase:
        model_mapping = cls.mapping.get(model_name, None)
        if model_mapping is None:
            raise NotImplementedError(
                "Model {} does not exist".format(model_name)
            )
        return model_mapping.dataloader_cls(**dataloader_param)
