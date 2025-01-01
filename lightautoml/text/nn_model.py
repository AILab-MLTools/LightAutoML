"""Neural Net modules for different data types."""

import logging

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union
import numpy as np
import torch
import torch.nn as nn
from ..tasks.base import Task

from .utils import _dtypes_mapping
logger = logging.getLogger(__name__)


class UniversalDataset:
    """Dataset class for mixed data.

    Args:
        data: Dict with data.
        y: Array of target variable.
        w: Optional array of observation weight.
        tokenizer: Transformers tokenizer.
        max_length: Max sentence length.
        stage: Name of current training / inference stage.

    """

    def __init__(
        self,
        fold: int,
        data: Dict[str, np.ndarray],
        y: np.ndarray,
        w: Optional[np.ndarray] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 256,
        stage: str = "test",
    ):
        self.fold = fold
        self.data = data
        self.y = y
        self.w = w
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stage = stage

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        res = {"fold":self.fold ,"label": self.y[index]}
        res.update({key: value[index] for key, value in self.data.items() if key != "text"})
        if (self.tokenizer is not None) and ("text" in self.data):
            sent = self.data["text"][index, 0]  # only one column
            _split = sent.split("[SEP]")
            sent = _split if len(_split) == 2 else (sent,)
            data = self.tokenizer.encode_plus(
                *sent, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True
            )

            res.update({i: np.array(data[i]) for i in data.keys()})
        if self.w is not None:
            res["weight"] = self.w[index]

        return res


class Clump(nn.Module):
    """Clipping input tensor.

    Args:
        min_v: Min value.
        max_v: Max value.

    """

    def __init__(self, min_v: int = -50, max_v: int = 50):
        super(Clump, self).__init__()

        self.min_v = min_v
        self.max_v = max_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = torch.clamp(x, self.min_v, self.max_v)
        return x
    

class TorchUniversalModel(nn.Module):
    """Mixed data model.

    Class for preparing input for DL model with mixed data.

    Args:
        loss: Callable torch loss with order of arguments (y_true, y_pred).
        task: Task object.
        torch_model: Torch model.
        n_out: Number of output dimensions.
        cont_embedder: Torch module for numeric data.
        cont_params: Dict with numeric model params.
        cat_embedder: Torch module for category data.
        cat_params: Dict with category model params.
        text_embedder: Torch module for text data.
        text_params: Dict with text model params.
        loss_on_logits: Calculate loss on logits or on predictions of model for classification tasks.
        bias: Array with last hidden linear layer bias.

    """

    def __init__(
        self,
        loss: Callable,
        task: Task,
        torch_model: nn.Module,
        n_out: int = 1,
        cont_embedder_: Optional[Any] = None,
        cont_params: Optional[Dict] = None,
        cat_embedder_: Optional[Any] = None,
        cat_params: Optional[Dict] = None,
        text_embedder: Optional[Any] = None,
        text_params: Optional[Dict] = None,
        loss_on_logits: bool = True,
        bias: Union[np.array, torch.Tensor] = None,
        **kwargs,
    ):
        super(TorchUniversalModel, self).__init__()
        self.n_out = n_out
        self.loss = loss
        self.task = task
        self.loss_on_logits = loss_on_logits

        self.cont_embedder = None
        self.cat_embedder = None
        self.text_embedder = None
        self.sampler = None

        n_in = 0
        if cont_embedder_ is not None:
            self.cont_embedder = cont_embedder_(**cont_params)
            n_in += self.cont_embedder.get_out_shape()
        if cat_embedder_ is not None:
            self.cat_embedder = cat_embedder_(**cat_params)
            n_in += self.cat_embedder.get_out_shape()
        if text_embedder is not None:
            self.text_embedder = text_embedder(**text_params)
            n_in += self.text_embedder.get_out_shape()

        self.torch_model = (
            torch_model(
                **{
                    **kwargs,
                    **{
                        "n_in": n_in,
                        "n_out": n_out,
                        "loss": loss,
                        "task": task,
                    },
                }
            )
            if torch_model is not None
            else nn.Sequential(nn.Linear(n_in, n_out))
        )

        if bias is not None:
            self._set_last_layer(self.torch_model, bias)

        self.сlump = Clump()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def _set_last_layer(self, torch_model, bias):
        use_skip = getattr(torch_model, "use_skip", False)
        self._init_last_layers(torch_model, bias, use_skip)

    def _init_last_layers(self, torch_model, bias, use_skip=False):
        try:
            all_layers = list(torch_model.children())
            layers = list(
                filter(
                    lambda x: isinstance(x, nn.Linear) or isinstance(x, nn.Sequential),
                    all_layers,
                )
            )
            if len(layers) == 0:
                last_layer = all_layers[-1]
                self._set_last_layer(last_layer, bias)

            else:
                last_layer = layers[-1]
                while isinstance(last_layer, nn.Sequential):
                    last_layer = list(
                        filter(lambda x: isinstance(x, nn.Linear) or isinstance(x, nn.Sequential), last_layer)
                    )[-1]
                bias = torch.Tensor(bias)
                last_layer.bias.data = bias
                shape = last_layer.weight.data.shape
                last_layer.weight.data = torch.zeros(shape[0], shape[1], requires_grad=True)
            if use_skip:
                if len(layers) <= 1:
                    last_layer = all_layers[-2]
                    self._set_last_layer(last_layer, bias)
                else:
                    pre_last_layer = layers[-2]
                    while isinstance(last_layer, nn.Sequential):
                        pre_last_layer = list(
                            filter(lambda x: isinstance(x, nn.Linear) or isinstance(x, nn.Sequential), pre_last_layer)
                        )[-1]
                    bias = torch.Tensor(bias)
                    pre_last_layer.bias.data = bias
                    shape = pre_last_layer.weight.data.shape
                    pre_last_layer.weight.data = torch.zeros(shape[0], shape[1], requires_grad=True)
        except:
            logger.info3("Last linear layer not founded, so init_bias=False")


    def get_logits(self, inp: Dict[str, torch.Tensor],efficient_bs:int = None) -> torch.Tensor:
        """Forward-pass of model with embeddings."""
        outputs = []
        
            
        if self.cont_embedder is not None:
            outputs.append(self.cont_embedder(inp))

        if self.cat_embedder is not None:
            outputs.append(self.cat_embedder(inp))

        if self.text_embedder is not None:
            outputs.append(self.text_embedder(inp))

        if len(outputs) > 1:
            output = torch.cat(outputs, dim=1)
        else:
            output = outputs[0]
        if efficient_bs is not None:
            logits = self.torch_model(output,efficient_bs)
        else:
            logits = self.torch_model(output)
        return logits

    def get_preds_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Prediction from logits."""
        if self.task.name in ["binary", "multilabel"]:
            out = self.sig(self.сlump(logits))
        elif self.task.name == "multiclass":
            # can't find self.clump when predicting
            out = self.softmax(torch.clamp(logits, -50, 50))
        else:
            out = logits

        return out

    def forward(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward-pass with output loss."""
        efficient_bs = None
        if inp['sampler'] is not None:
            efficient_bs = len(inp['label'])
            candidate_sample = next(inp['sampler'])
            inp = {
                    i: torch.cat([inp[i],
                    (candidate_sample[i].long().to(self.torch_model.device) if _dtypes_mapping[i] == "long" else candidate_sample[i].to(self.torch_model.device))])
                    for i in set(inp.keys())-set(['sampler'])
                }
        x = self.get_logits(inp,efficient_bs)
        if not self.loss_on_logits:
            x = self.get_preds_from_logits(x)

        loss = self.loss(inp["label"].view(inp["label"].shape[0], -1), x, inp.get("weight", None))
        return loss

    def predict(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prediction."""
        efficient_bs = None
        if inp['sampler'] is not None:
            efficient_bs = len(inp['label'])
            candidate_sample = next(inp['sampler'])
            inp = {
                    i: torch.cat([inp[i],
                    (candidate_sample[i].long().to(self.torch_model.device) if _dtypes_mapping[i] == "long" else candidate_sample[i].to(self.torch_model.device))])
                    for i in set(inp.keys())-set(['sampler'])
                }
        x = self.get_logits(inp,efficient_bs)
        x = self.get_preds_from_logits(x)
        return x
