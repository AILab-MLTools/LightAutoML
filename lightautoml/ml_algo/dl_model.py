"""Neural net for tabular datasets."""


from lightautoml.dataset.base import LAMLDataset
from lightautoml.dataset.utils import vconcatenate
from lightautoml.utils.installation import __validate_extra_deps


__validate_extra_deps("nlp")


import gc
import logging
import os
import sys
import uuid

from copy import copy
from typing import Dict

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..dataset.np_pd_dataset import NumpyDataset
from ..tasks.losses.torch import TorchLossWrapper
from ..utils.installation import __validate_extra_deps
from ..validation.base import TrainValidIterator


__validate_extra_deps("nlp")

try:
    from transformers import AutoTokenizer

    from ..pipelines.features.text_pipeline import _model_name_by_lang
except:
    import warnings

    warnings.warn("'transformers' - package isn't installed")

from ..ml_algo.base import TabularDataset
from ..ml_algo.base import TabularMLAlgo
from ..pipelines.utils import get_columns_by_role
from ..text.embed import (
    BasicCatEmbeddingFlat,
    CatEmbedder,
    DenseEmbedding,
    DenseEmbeddingFlat,
    LinearEmbedding,
    LinearEmbeddingFlat,
    MLPContEmbedding,
    MLPContEmbeddingFlat,
    PLREmbedding,
    PLREmbeddingFlat,
    SoftEmbedding,
    SoftEmbeddingFlat,
    WeightedCatEmbedding,
    BasicCatEmbedding,
    WeightedCatEmbeddingFlat,
)
from ..text.embed import ContEmbedder
from ..text.embed import TextBert
from ..text.nn_model import TorchUniversalModel
from ..text.nn_model import UniversalDataset
from ..text.trainer import Trainer
from ..text.utils import collate_dict
from ..text.utils import inv_sigmoid
from ..text.utils import inv_softmax
from ..text.utils import is_shuffle
from ..text.utils import parse_devices
from ..text.utils import seed_everything
from .torch_based.nn_models import MLP, TabNet
from .torch_based.nn_models import NODE
from .torch_based.nn_models import SNN
from .torch_based.nn_models import SAINT
from .torch_based.nn_models import DenseLightModel
from .torch_based.nn_models import DenseModel
from .torch_based.nn_models import LinearLayer
from .torch_based.nn_models import ResNetModel
from .torch_based.nn_models import _LinearLayer
from .torch_based.nn_models import AutoInt
from .torch_based.nn_models import FTTransformer


logger = logging.getLogger(__name__)

models_dependent_on_training_data = ["saint"]

model_by_name = {
    "denselight": DenseLightModel,
    "dense": DenseModel,
    "resnet": ResNetModel,
    "mlp": MLP,
    "linear_layer": LinearLayer,
    "_linear_layer": _LinearLayer,
    "snn": SNN,
    "node": NODE,
    "autoint": AutoInt,
    "tabnet": TabNet,
    "fttransformer": FTTransformer,
    "saint":SAINT,
}

input_type_by_name = {
    "denselight": "flat",
    "dense": "flat",
    "resnet": "flat",
    "mlp": "flat",
    "linear_layer": "flat",
    "_linear_layer": "flat",
    "snn": "flat",
    "node": "flat",
    "autoint": "seq",
    "tabnet": "flat",
    "fttransformer": "seq",
    "saint": "seq",
}
cat_embedder_by_name_flat = {
    "cat": CatEmbedder,
    "cat_no_dropout": BasicCatEmbeddingFlat,
    "weighted": WeightedCatEmbeddingFlat,
}
cat_embedder_by_name = {
    "cat_no_dropout": BasicCatEmbedding,
    "cat_no_dropout": BasicCatEmbedding,
    "weighted": WeightedCatEmbedding,
}

cont_embedder_by_name_flat = {
    "cont": ContEmbedder,
    "linear": LinearEmbeddingFlat,
    "dense": DenseEmbeddingFlat,
    "plr": PLREmbeddingFlat,
    "soft": SoftEmbeddingFlat,
    "mlp": MLPContEmbeddingFlat
}
cont_embedder_by_name = {
    "cont": LinearEmbedding,
    "linear": LinearEmbedding,
    "dense": DenseEmbedding,
    "plr": PLREmbedding,
    "soft": SoftEmbedding,
    "mlp": MLPContEmbedding,
}


class TorchModel(TabularMLAlgo):
    """Neural net for tabular datasets.

    default_params:

        - bs: Batch size.
        - num_workers: Number of threads for multiprocessing.
        - max_length: Max sequence length.
        - opt_params: Dict with optim params.
        - scheduler_params: Dict with scheduler params.
        - is_snap: Use snapshots.
        - snap_params: Dict with SE parameters.
        - init_bias: Init last linear bias by mean target values.
        - n_epochs: Number of training epochs.
        - input_bn: Use 1d batch norm for input data.
        - emb_dropout: Dropout probability.
        - emb_ratio: Ratio for embedding size = (x + 1) // emb_ratio.
        - max_emb_size: Max embedding size.
        - device: Torch device or str.
        - use_cont: Use numeric data.
        - use_cat: Use category data.
        - use_text: Use text data.
        - lang: Text language.
        - bert_name: Name of HuggingFace transformer model.
        - pooling: Type of pooling strategy for bert model.
        - deterministic: CUDNN backend.
        - multigpu: Use Data Parallel.
        - model_with_emb: Use model with custom embeddings.
        - loss: Torch loss or str or func with (y_pred, y_true) args.
        - loss_params: Dict with loss params.
        - loss_on_logits: Calculate loss on logits or on predictions of model for classification tasks.
        - clip_grad: Clip gradient before loss backprop.
        - clip_grad_params: Dict with clip_grad params.
        - dataset: Class for data retrieval
        - tuned: Tune custom model
        - num_init_features: Scale input dimension to another one
        - use_noise: Use Noise
        - use_bn: Use BatchNorm
        - path_to_save: Path to save model checkpoints,
          ``None`` - stay in memory.
        - random_state: Random state to take subsample.
        - verbose_inside: Number of steps between
          verbose inside epoch or ``None``.
        - verbose: Verbose every N epochs.

    freeze_defaults:

        - ``True`` :  params can be rewritten depending on dataset.
        - ``False``:  params can be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """

    _name: str = "TorchNN"
    _params: Dict = None

    _default_models_params = {
        "n_out": None,
        "hid_factor": [2, 2],
        "hidden_size": [512, 512, 512],
        "block_config": [2, 2],
        "compression": 0.5,
        "growth_size": 256,
        "bn_factor": 2,
        "drop_rate": 0.1,
        "noise_std": 0.05,
        "num_init_features": None,
        "act_fun": nn.ReLU,
        "use_noise": False,
        "use_bn": True,
    }

    _default_params = {
        "num_workers": 0,
        "pin_memory": False,
        "max_length": 256,
        "is_snap": False,
        "input_bn": False,
        "max_emb_size": 256,
        "bert_name": None,
        "pooling": "cls",
        "device": torch.device("cuda:0"),
        "use_cont": True,
        "use_cat": True,
        "use_text": True,
        "lang": "en",
        "deterministic": True,
        "multigpu": False,
        "random_state": 42,
        "model": "dense",
        "model_with_emb": False,
        "path_to_save": os.path.join("./models/", "model"),
        "verbose_inside": None,
        "verbose": 1,
        "n_epochs": 20,
        "snap_params": {"k": 3, "early_stopping": True, "patience": 16, "swa": True},
        "bs": 512,
        "emb_dropout": 0.1,
        "emb_ratio": 3,
        "opt": torch.optim.Adam,
        "opt_params": {"weight_decay": 0, "lr": 3e-4},
        "sch": ReduceLROnPlateau,
        "scheduler_params": {"patience": 10, "factor": 1e-2, "min_lr": 1e-5},
        "loss": None,
        "loss_params": {},
        "loss_on_logits": True,
        "clip_grad": False,
        "clip_grad_params": {},
        "init_bias": True,
        "dataset": UniversalDataset,
        "tuned": False,
        "optimization_search_space": None,
        "verbose_bar": False,
        "freeze_defaults": False,
        **_default_models_params,
    }

    def _infer_params(self, train = None):
        if self.params["path_to_save"] is not None:
            self.path_to_save = os.path.relpath(self.params["path_to_save"])
            if not os.path.exists(self.path_to_save):
                os.makedirs(self.path_to_save)
        else:
            self.path_to_save = None

        params = copy(self.params)
        loss = params["loss"]

        if isinstance(loss, str):
            loss = getattr(nn, loss)
        if loss is not None and issubclass(loss, nn.Module):
            loss = TorchLossWrapper(loss, **params["loss_params"])
        if loss is None:
            loss = self.task.losses["torch"].loss

        params["loss"] = loss
        params["metric"] = self.task.losses["torch"].metric_func

        if params["bert_name"] is None and params["use_text"]:
            params["bert_name"] = _model_name_by_lang[params.get("lang", "en")]

        is_text = (len(params["text_features"]) > 0) and (params["use_text"]) and (params["device"].type == "cuda")
        is_cat = (len(params["cat_features"]) > 0) and (params["use_cat"])
        is_cont = (len(params["cont_features"]) > 0) and (params["use_cont"])

        torch_model = params["model"]

        if isinstance(torch_model, str):
            assert torch_model in model_by_name, "Wrong model name. Available models: " + str(model_by_name.keys())
            if torch_model == "snn":
                params["init_bias"] = False
            torch_model = model_by_name[torch_model]

        assert issubclass(torch_model, nn.Module), "Wrong model format, only support torch models"

        # str to nn modules
        for p_name, module in [
            ["act_fun", nn],
            ["dataset", sys.modules[__name__]],
            ["opt", torch.optim],
            ["sch", torch.optim.lr_scheduler],
        ]:
            if isinstance(params[p_name], str):
                params[p_name] = getattr(module, params[p_name])

        # params = self._select_params(params)
        if params['model'] in models_dependent_on_training_data:
            self.use_sampler = True
            if train is not None:
                self.train = train
        else:
            self.use_sampler = False

        self.train_params = {
            "dataset": params["dataset"],
            "bs": params["bs"],
            "num_workers": params["num_workers"],
            "pin_memory": params["pin_memory"],
            "tokenizer": AutoTokenizer.from_pretrained(params["bert_name"], use_fast=False) if is_text else None,
            "max_length": params["max_length"],
        }

        model = Trainer(
            net=TorchUniversalModel if not params["model_with_emb"] else params["model"],
            net_params={
                "task": self.task,
                "cont_embedder_": cont_embedder_by_name.get(params["cont_embedder"], LinearEmbedding)
                if input_type_by_name.get(params["model"], "flat") == "seq" and is_cont
                else cont_embedder_by_name_flat.get(params["cont_embedder"], ContEmbedder)
                if is_cont
                else None,
                "cont_params": {
                    # "num_dims": params["num_dims"],
                    # "input_bn": params["input_bn"],
                    # "device": params["device"],
                    # "embedding_size": params["embedding_size"],
                    **params
                }
                if is_cont
                else None,
                "cat_embedder_": cat_embedder_by_name.get(params["cat_embedder"], BasicCatEmbedding)
                if input_type_by_name.get(params["model"], "flat") == "seq" and is_cat
                else cat_embedder_by_name_flat.get(params["cat_embedder"], CatEmbedder)
                if is_cat
                else None,
                "cat_params": {
                    # "cat_vc": params["cat_vc"],
                    # "cat_dims": params["cat_dims"],
                    # "emb_dropout": params["emb_dropout"],
                    # "emb_ratio": params["emb_ratio"],
                    # "max_emb_size": params["max_emb_size"],
                    # "embedding_size": params["embedding_size"],
                    # "device": params["device"],
                    **params
                }
                if is_cat
                else None,
                "text_embedder": TextBert if is_text else None,
                "text_params": {
                    "model_name": params["bert_name"],
                    "pooling": params["pooling"],
                }
                if is_text
                else None,
                "torch_model": torch_model,
                **params,
            },
            
            **{"apex": False,
                **params},
        )

        return model

    @staticmethod
    def get_mean_target(target, task_name: str):
        """Get target mean / inverse sigmoid transformation \
            to init bias in last layer of network.

        Args:
            target: Target values.
            task_name: One of the available task names

        Returns:
            Array with bias values.

        """
        if isinstance(target, pd.Series) or isinstance(target, pd.DataFrame):
            target = target.values
        target = target.reshape(target.shape[0], -1)
        bias = (
            np.nanmean(target, axis=0).astype(float)
            if (task_name != "multiclass")
            else np.unique(target, return_counts=True)[1]
        )
        bias = (
            inv_sigmoid(bias)
            if (task_name == "binary") or (task_name == "multilabel")
            else inv_softmax(bias)
            if (task_name == "multiclass")
            else bias
        )

        bias[bias == np.inf] = np.nanmax(bias[bias != np.inf])
        bias[bias == -np.inf] = np.nanmin(bias[bias != -np.inf])
        bias[bias == np.NaN] = np.nanmean(bias[bias != np.NaN])

        return bias

    def _init_params_on_input(self, train_valid_iterator) -> dict:
        """Init params that common for all folds.

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Dict with data parameters.

        """
        params = self.params
        new_params = {}
        new_params["device"], new_params["device_ids"] = parse_devices(params["device"], params["multigpu"])

        task_name = train_valid_iterator.train.task.name
        target = train_valid_iterator.train.target

        if params["n_out"] is None:
            new_params["n_out"] = 1 if task_name != "multiclass" else (np.max(target) + 1).astype(int)
            new_params["n_out"] = target.shape[1] if task_name in ["multi:reg", "multilabel"] else new_params["n_out"]

        cat_dims = []
        new_params["cat_features"] = get_columns_by_role(train_valid_iterator.train, "Category")

        # Cat_features are needed to be preprocessed with LE, where 0 = not known category
        valid = train_valid_iterator.get_validation_data()
        cat_value_counts = []
        for cat_feature in new_params["cat_features"]:
            num_unique_categories = (
                max(
                    max(train_valid_iterator.train[:, cat_feature].data),
                    max(valid[:, cat_feature].data),
                )
                + 1
            )
            if params["cat_embedder"] == "weighted":
                values, counts = np.unique(
                    np.concatenate([train_valid_iterator.train[:, cat_feature].data, valid[:, cat_feature].data]),
                    return_counts=True,
                )
                cat_value_counts.append(dict(zip(values, counts)))
            cat_dims.append(num_unique_categories)
        new_params["cat_dims"] = cat_dims
        new_params["cat_vc"] = cat_value_counts
        new_params["cont_features"] = get_columns_by_role(train_valid_iterator.train, "Numeric")
        new_params["num_dims"] = len(new_params["cont_features"])

        new_params["text_features"] = get_columns_by_role(train_valid_iterator.train, "Text")
        new_params["bias"] = self.get_mean_target(target, task_name) if params["init_bias"] else None

        logger.debug(f'number of text features: {len(new_params["text_features"])} ')
        logger.debug(f'number of categorical features: {len(new_params["cat_features"])} ')
        logger.debug(f'number of continuous features: {new_params["num_dims"]} ')

        return new_params

    def init_params_on_input(self, train_valid_iterator) -> dict:
        """Get model parameters depending on dataset parameters.

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Parameters of model.

        """
        suggested_params = copy(self.params)
        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params

        rows_num = len(train_valid_iterator.train) + len(train_valid_iterator.get_validation_data())
        bs = 256

        if rows_num > 50_000:
            bs = 512
        if rows_num > 100_000:
            bs = 1024
        if rows_num > 1_000_000:
            bs = 2048

        suggested_params["bs"] = bs
        suggested_params["snap_params"] = (
            {"k": 1, "early_stopping": True, "patience": 16, "swa": False}
            if self.params["model"] == "snn"
            else self.params["snap_params"]
        )
        return suggested_params

    def get_dataloaders_from_dicts(self, data_dict: Dict, n : int =0):
        """Construct dataloaders depending on stage.

        Args:
            data_dict: Dict with (stage_name, data) (key, value).

        Returns:
            Dataloaders.

        """
        datasets = {}
        for stage, value in data_dict.items():
            data = {
                name: value.data[cols].values
                for name, cols in zip(
                    ["text", "cat", "cont"],
                    [
                        self.params["text_features"],
                        self.params["cat_features"],
                        self.params["cont_features"],
                    ],
                )
                if len(cols) > 0
            }

            datasets[stage] = self.train_params["dataset"](
                fold = n,
                data=data,
                y=value.target.values if stage != "test" else np.ones(len(value.data)),
                w=value.weights.values if value.weights is not None else np.ones(len(value.data)),
                tokenizer=self.train_params["tokenizer"],
                max_length=self.train_params["max_length"],
                stage=stage,
            )

        dataloaders = {
            stage: torch.utils.data.DataLoader(
                datasets[stage],
                batch_size=self.train_params["bs"],
                shuffle=is_shuffle(stage),
                num_workers=self.train_params["num_workers"],
                collate_fn=collate_dict,
                pin_memory=self.train_params["pin_memory"],
                drop_last={"train": True}.get(stage, False),
            )
            for stage, value in data_dict.items()
        }
        return dataloaders

    def fit_predict(self, train_valid_iterator: TrainValidIterator) -> NumpyDataset:
        """Fit and then predict accordig the strategy that uses train_valid_iterator.

        If item uses more then one time it will
        predict mean value of predictions.
        If the element is not used in training then
        the prediction will be ``numpy.nan`` for this item

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Dataset with predicted values.

        """
        if self._params is None:
            self.params = self.init_params_on_input(train_valid_iterator)
        self.params = self._init_params_on_input(train_valid_iterator)
        return super().fit_predict(train_valid_iterator)
    
    def fit_predict_single_fold(self, train: TabularDataset, valid: TabularDataset, n=0):
        """Implements training and prediction on single fold.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Tuple (model, predicted_values).

        """
        seed_everything(self.params["random_state"], self.params["deterministic"])
        task_name = train.task.name
        target = train.target
        self.params["bias"] = self.get_mean_target(target, task_name) if self.params["init_bias"] else None

        if self.params['is_holdout']:
            ds = train
        else:
            ds = vconcatenate([train,valid])
        model = self._infer_params(ds)

        model_path = (
            os.path.join(self.path_to_save, f"{uuid.uuid4()}.pickle") if self.path_to_save is not None else None
        )
        # init datasets
        if self.use_sampler:
            dataloaders = self.get_dataloaders_from_dicts({"train": train.to_pandas(), "val": valid.to_pandas(),"sampler": train.to_pandas()},n)
        else:
            dataloaders = self.get_dataloaders_from_dicts({"train": train.to_pandas(), "val": valid.to_pandas()},n)
            dataloaders['sampler'] = None
        val_pred = model.fit(dataloaders)

        if model_path is None:
            model_path = model.state_dict(model_path)
        else:
            model.state_dict(model_path)

        model.clean()
        del dataloaders, model
        gc.collect()
        torch.cuda.empty_cache()
        return model_path, val_pred

    def predict_single_fold(self, model: any, dataset: TabularDataset) -> np.ndarray:
        """Predict target values for dataset.

        Args:
            model: Neural net object or dict or str.
            dataset: Test dataset.

        Returns:
            Predicted target values.

        """
        seed_everything(self.params["random_state"], self.params["deterministic"])
        if self.use_sampler:
            dataloaders = self.get_dataloaders_from_dicts({"test": dataset.to_pandas(),"sampler": self.train.to_pandas()})
        else:
            dataloaders = self.get_dataloaders_from_dicts({"test": dataset.to_pandas()})
            dataloaders['sampler'] = None


        if isinstance(model, (str, dict)):
            model = self._infer_params().load_state(model)

        pred = model.predict(dataloaders, "test")

        model.clean()
        del dataloaders, model
        gc.collect()
        torch.cuda.empty_cache()

        return pred

    def _get_default_search_spaces(self, suggested_params: Dict, estimated_n_trials: int) -> Dict:
        def sample(estimated_n_trials: int, trial: optuna.trial.Trial, suggested_params: Dict) -> Dict:
            """Implements simple tuning sampling strategy.

            Args:
                trial: Current optuna Trial.
                estimated_n_trials: Estimated trials based on time spent on previous ones.
                suggested_params: Suggested params

            Returns:
                Dict with Sampled params.

            """
            # optionally
            trial_values = copy(suggested_params)

            trial_values["bs"] = trial.suggest_categorical("bs", [2 ** i for i in range(6, 11)])

            weight_decay_bin = trial.suggest_categorical("weight_decay_bin", [0, 1])
            if weight_decay_bin == 0:
                weight_decay = 0
            else:
                weight_decay = trial.suggest_float("weight_decay", low=1e-6, high=1e-2, log=True)

            lr = trial.suggest_float("lr", low=1e-5, high=1e-1, log=True)
            trial_values["opt_params"] = {
                "lr": lr,
                "weight_decay": weight_decay,
            }
            return trial_values

        return sample
