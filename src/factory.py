from abc import ABC, abstractclassmethod
from dataclasses import dataclass

import os
import sys
from typing import Any, Dict, Tuple, Type, Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.base import ModelBase
from models import (
    FFN,
    VAE_FFN,
    VAECH,
    VAECH_I,
    VAECH_II,
    VAESDR,
)
from models.ASDDiagNet import ASDDiagNet
from models.ASDSAENet import ASDSAENet
from models.GAEFCNN import GAE_FCNN, GCN_FCNN, GAE, GVAE, GFCNN
from data import DataloaderBase, ModelBaseDataloader, GraphModelBaseDataloader


class FrameworkFactory(ABC):
    @abstractclassmethod
    def load_model(
        cls, model_name: str, model_param: Dict[str, Any]
    ) -> Union[ModelBase, Tuple[ModelBase, ModelBase]]:
        raise NotImplementedError

    @abstractclassmethod
    def load_dataloader(
        cls, model_name: str, dataloader_param: Dict[str, Any]
    ) -> DataloaderBase:
        raise NotImplementedError


class SingleStageFrameworkFactory(FrameworkFactory):
    @dataclass
    class Mapping:
        model_cls: Type[ModelBase]
        dataloader_cls: Type[DataloaderBase]

    mapping = {
        "FFN": Mapping(FFN, ModelBaseDataloader),
        "ASDDiagNet": Mapping(ASDDiagNet, ModelBaseDataloader),
        "ASDSAENet": Mapping(ASDSAENet, ModelBaseDataloader),
        "AE-FFN": Mapping(ASDDiagNet, ModelBaseDataloader),
        "SAE-FFN": Mapping(ASDSAENet, ModelBaseDataloader),
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
    def load_model(
        cls, model_name: str, model_param: Dict[str, Any]
    ) -> ModelBase:
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
        model_cls: Tuple[Type[ModelBase], Type[ModelBase]]
        compile_model_cls: Type[ModelBase]
        dataloader_cls: DataloaderBase

    mapping = {
        "GAE-FCNN": Mapping((GAE, GFCNN), GAE_FCNN, GraphModelBaseDataloader),
        "GVAE-FCNN": Mapping((GVAE, GFCNN), GAE_FCNN, GraphModelBaseDataloader),
    }

    @classmethod
    def compile_model(
        cls, model_name: str, ae_model: ModelBase, fcnn_model: ModelBase
    ) -> ModelBase:
        model_mapping = cls.mapping.get(model_name, None)
        if model_mapping is None:
            raise NotImplementedError(
                "Model {} does not exist".format(model_name)
            )
        model_cls = model_mapping.compile_model_cls
        return model_cls(ae_model, fcnn_model)

    @classmethod
    def load_model(
        cls, model_name: str, model_param: Dict[str, Dict[str, Any]]
    ) -> Tuple[ModelBase, ModelBase]:
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
