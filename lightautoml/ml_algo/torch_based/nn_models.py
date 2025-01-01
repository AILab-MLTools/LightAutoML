"""Torch models."""

from collections import OrderedDict
from typing import List, Tuple, Type
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from .saint.saint_utils import ColTransformer, RowColTransformer
from .tabnet.utils import TabNetEncoder, _initialize_non_glu
from .autoint.autoint_utils import AttnInteractionBlock, LeakyGate
from .autoint.ghost_norm import GhostBatchNorm
from .fttransformer.fttransformer_utils import Transformer

from .node_nn_model import DenseODSTBlock, MeanPooling


class GaussianNoise(nn.Module):
    """Adds gaussian noise.

    Args:
        stddev: Std of noise.
        device: Device to compute on.

    """

    def __init__(self, stddev: float, device: torch.device):
        super().__init__()
        self.stddev = stddev
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        if self.training:
            return x + torch.randn(x.size(), device=self.device) * self.stddev
        return x


class UniformNoise(nn.Module):
    """Add uniform noise.

    Args:
            stddev: Std of noise.
            device: Device to compute on.

    """

    def __init__(self, stddev: float, device: torch.Tensor):
        super().__init__()
        self.stddev = stddev
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        if self.training:
            return x + (torch.rand(x.size(), device=self.device) - 0.5) * self.stddev
        return x


class DenseLightBlock(nn.Module):
    """Realisation of `'denselight'` model block.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            drop_rate: Dropout rate.
            noise_std: Std of noise.
            act_fun: Activation function.
            use_bn: Use BatchNorm.
            use_noise: Use noise.
            device: Device to compute on.

    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        drop_rate: float = 0.1,
        noise_std: float = 0.05,
        act_fun: nn.Module = nn.ReLU,
        use_bn: bool = True,
        use_noise: bool = False,
        device: torch.device = torch.device("cuda:0"),
        bn_momentum: float = 0.1,
        ghost_batch: Optional[int] = None,
        **kwargs,
    ):
        super(DenseLightBlock, self).__init__()
        self.features = nn.Sequential(OrderedDict([]))
        self.features.add_module("dense", nn.Linear(n_in, n_out, bias=(not use_bn)))
        if use_bn:
            if ghost_batch is None:
                self.features.add_module("norm", nn.BatchNorm1d(n_out, momentum=bn_momentum))
            else:
                self.features.add_module("norm", GhostBatchNorm(n_out, ghost_batch, momentum=bn_momentum))

        self.features.add_module("act", act_fun())

        if drop_rate:
            self.features.add_module("dropout", nn.Dropout(p=drop_rate))
        if use_noise:
            self.features.add_module("noise", GaussianNoise(noise_std, device))

        # self.features.add_module("dense", nn.Linear(n_in, n_out))
        # self.features.add_module("act", act_fun())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        for name, layer in self.features.named_children():
            x = layer(x)
        return x


class DenseLightModel(nn.Module):
    """Realisation of `'denselight'` model.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hidden_size: List of hidden dims.
            drop_rate: Dropout rate for each layer separately or altogether.
            act_fun: Activation function.
            noise_std: Std of noise.
            num_init_features: If not none add fc layer before model with certain dim.
            use_bn: Use BatchNorm.
            use_noise: Use noise.
            concat_input: Concatenate input to all hidden layers. # MLP False
            dropout_first: Use dropout in the first layer or not.
            bn_momentum: BatchNorm momentum
            ghost_batch: If not none use GhoastNorm with ghost_batch.
            leaky_gate: Use LeakyGate or not.
            use_skip: Use another Linear model to blend them after.
            weighted_sum: Use weighted blender or half-half.
            device: Device to compute on.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        hidden_size: List[int] = [
            512,
            750,
        ],
        drop_rate: Union[float, List[float]] = 0.1,
        act_fun: nn.Module = nn.LeakyReLU,
        noise_std: float = 0.05,
        num_init_features: Optional[int] = None,
        use_bn: bool = True,
        use_noise: bool = False,
        concat_input: bool = True,
        dropout_first: bool = True,
        bn_momentum: float = 0.1,
        ghost_batch: Optional[int] = None,
        use_skip: bool = False,
        leaky_gate: bool = False,
        weighted_sum: bool = True,
        device: torch.device = torch.device("cuda:0"),
        **kwargs,
    ):
        super(DenseLightModel, self).__init__()

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        if isinstance(drop_rate, float):
            drop_rate = [drop_rate] * (len(hidden_size) + (1 if dropout_first else 0))

        assert (
            len(hidden_size) == len(drop_rate) if not dropout_first else 1 + len(hidden_size) == len(drop_rate)
        ), "Wrong number hidden_sizes/drop_rates. Must be equal."

        self.concat_input = concat_input
        num_features = n_in if num_init_features is None else num_init_features

        self.features = nn.Sequential(OrderedDict([]))
        if num_init_features is not None:
            self.features.add_module("dense0", nn.Linear(n_in, num_features))

        if leaky_gate:
            self.features.add_module("leakygate0", LeakyGate(num_features))

        if dropout_first and drop_rate[0] > 0:
            self.features.add_module("dropout0", nn.Dropout(drop_rate[0]))
            drop_rate = drop_rate[1:]

        for i, hid_size in enumerate(hidden_size):
            block = DenseLightBlock(
                n_in=num_features,
                n_out=hid_size,
                drop_rate=drop_rate[i],
                noise_std=noise_std,
                act_fun=act_fun,
                use_bn=use_bn,
                use_noise=use_noise,
                device=device,
                bn_momentum=bn_momentum,
                ghost_batch=ghost_batch,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)

            if concat_input:
                num_features = n_in + hid_size
            else:
                num_features = hid_size

        num_features = hidden_size[-1]
        self.fc = nn.Linear(num_features, n_out)
        self.use_skip = use_skip
        if use_skip:
            skip_linear = nn.Linear(n_in, n_out)
            if leaky_gate:
                self.skip_layers = nn.Sequential(LeakyGate(n_in), skip_linear)
            else:
                self.skip_layers = skip_linear
            if weighted_sum:
                self.mix = nn.Parameter(torch.tensor([0.0]))
            else:
                self.mix = torch.tensor([0.0], device=device)
        else:
            self.skip_layers = None
            self.mix = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = X
        input = x.detach().clone()
        for name, layer in self.features.named_children():
            if name not in ["dropout0", "leakygate0", "denseblock1", "dense0"] and self.concat_input:
                x = torch.cat([x, input], 1)
            x = layer(x)
        out = self.fc(x)
        if self.use_skip:
            mix = torch.sigmoid(self.mix)
            skip_out = self.skip_layers(X)
            out = mix * skip_out + (1 - mix) * out
        return out


class MLP(DenseLightModel):
    """Realisation of `'mlp'` model.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hidden_size: List of hidden dims.
            drop_rate: Dropout rate for each layer separately or altogether.
            act_fun: Activation function.
            noise_std: Std of noise.
            num_init_features: If not none add fc layer before model with certain dim.
            use_bn: Use BatchNorm.
            use_noise: Use noise.
            dropout_first: Use dropout in the first layer or not.
            bn_momentum: BatchNorm momentum
            ghost_batch: If not none use GhoastNorm with ghost_batch.
            leaky_gate: Use LeakyGate or not.
            use_skip: Use another Linear model to blend them after.
            weighted_sum: Use weighted blender or half-half.
            device: Device to compute on.
    """

    def __init__(self, *args, **kwargs):
        super(MLP, self).__init__(*args, **{**kwargs, **{"concat_input": False}})


class _LinearLayer(DenseLightBlock):
    """Realisation of `'_linear_layer'` model.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hidden_size: List of hidden dims.
            noise_std: Std of noise.
            num_init_features: If not none add fc layer before model with certain dim.
            device: Device to compute on.

    """

    def __init__(self, *args, **kwargs):
        super(_LinearLayer, self).__init__(
            *args,
            **{
                **kwargs,
                **{
                    "use_bn": True,
                    "use_noise": False,
                    "drop_rate": 0.0,
                    "act_fun": nn.Identity,
                },
            },
        )


class LinearLayer(DenseLightBlock):
    """Realisation of `'linear_layer'` model.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hidden_size: List of hidden dims.
            noise_std: Std of noise.
            num_init_features: If not none add fc layer before model with certain dim.
            device: Device to compute on.

    """

    def __init__(self, *args, **kwargs):
        super(LinearLayer, self).__init__(
            *args,
            **{
                **kwargs,
                **{
                    "use_bn": False,
                    "use_noise": False,
                    "drop_rate": 0.0,
                    "act_fun": nn.Identity,
                },
            },
        )


class DenseLayer(nn.Module):
    """Realisation of `'dense'` model layer.

    Args:
            n_in: Input dim.
            growth_size: Output dim.
            bn_factor: Dim of intermediate fc is increased times `bn_factor` in DenseModel layer.
            drop_rate: Dropout rate.
            act_fun: Activation function.
            use_bn: Use BatchNorm.

    """

    def __init__(
        self,
        n_in: int,
        growth_size: int = 256,
        bn_factor: float = 2,
        drop_rate: float = 0.1,
        act_fun: nn.Module = nn.ReLU,
        use_bn: bool = True,
        **kwargs,
    ):
        super(DenseLayer, self).__init__()

        self.features1 = nn.Sequential(OrderedDict([]))
        self.features2 = nn.Sequential(OrderedDict([]))

        if use_bn:
            self.features1.add_module("norm1", nn.BatchNorm1d(n_in))

        self.features1.add_module("dense1", nn.Linear(n_in, int(bn_factor * n_in)))
        self.features1.add_module("act1", act_fun())

        if use_bn:
            self.features2.add_module("norm2", nn.BatchNorm1d(int(bn_factor * n_in)))

        self.features2.add_module("dense2", nn.Linear(int(bn_factor * n_in), growth_size))
        self.features2.add_module("act2", act_fun())

        if drop_rate:
            self.features2.add_module("dropout", nn.Dropout(drop_rate))

    def forward(self, prev_features: List[torch.Tensor]):
        """Forward-pass."""
        x = self.features1(torch.cat(prev_features, 1))
        x = self.features2(x)
        return x


class Transition(nn.Sequential):
    """Compress input to lower dim.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            growth_size: Output dim of every layer.
            act_fun: Activation function.
            use_bn: Use BatchNorm.

    """

    def __init__(self, n_in: int, n_out: int, act_fun: nn.Module, use_bn: bool = True):
        super(Transition, self).__init__()
        if use_bn:
            self.add_module("norm", nn.BatchNorm1d(n_in))

        self.add_module("dense", nn.Linear(n_in, n_out))
        self.add_module("act", act_fun())


class DenseBlock(nn.Module):
    """Realisation of `'dense'` model block.

    Args:
            n_in: Input dim.
            num_layers: Number of layers.
            bn_factor: Dim of intermediate fc is increased times `bn_factor` in DenseModel layer.
            growth_size: Output dim of every layer.
            drop_rate: Dropout rate.
            act_fun: Activation function.
            use_bn: Use BatchNorm.

    """

    def __init__(
        self,
        num_layers: int,
        n_in: int,
        bn_factor: float,
        growth_size: int,
        drop_rate: float = 0.1,
        act_fun: nn.Module = nn.ReLU,
        use_bn: bool = True,
        **kwargs,
    ):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                n_in + i * growth_size,
                growth_size=growth_size,
                bn_factor=bn_factor,
                drop_rate=drop_rate,
                act_fun=act_fun,
                use_bn=use_bn,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: List[torch.Tensor]):
        """Forward-pass with layer output concatenation in the end."""
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseModel(nn.Module):
    """Realisation of `'dense'` model.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            block_config: List of number of layers within each block
            drop_rate: Dropout rate for each layer separately or altogether.
            num_init_features: If not none add fc layer before model with certain dim.
            compression: portion of neuron to drop after block.
            growth_size: Output dim of every layer.
            bn_factor: Dim of intermediate fc is increased times `bn_factor` in DenseModel layer.
            act_fun: Activation function.
            use_bn: Use BatchNorm.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        block_config: List[int] = [2, 2],
        drop_rate: Union[float, List[float]] = 0.1,
        num_init_features: Optional[int] = None,
        compression: float = 0.5,
        growth_size: int = 256,
        bn_factor: float = 2,
        act_fun: nn.Module = nn.ReLU,
        use_bn: bool = True,
        **kwargs,
    ):
        super(DenseModel, self).__init__()
        assert 0 < compression <= 1, "compression of densenet should be between 0 and 1"

        if isinstance(drop_rate, float):
            drop_rate = [drop_rate] * len(block_config)

        assert len(block_config) == len(drop_rate), "Wrong number hidden_sizes/drop_rates. Must be equal."

        num_features = n_in if num_init_features is None else num_init_features
        self.features = nn.Sequential(OrderedDict([]))
        if num_init_features is not None:
            self.features.add_module("dense0", nn.Linear(n_in, num_features))

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                n_in=num_features,
                bn_factor=bn_factor,
                growth_size=growth_size,
                drop_rate=drop_rate[i],
                act_fun=act_fun,
                use_bn=use_bn,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_size
            if i != len(block_config) - 1:
                trans = Transition(
                    n_in=num_features,
                    n_out=max(10, int(num_features * compression)),
                    act_fun=act_fun,
                    use_bn=use_bn,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = max(10, int(num_features * compression))

        if use_bn:
            self.features.add_module("norm_final", nn.BatchNorm1d(num_features))

        self.fc = nn.Linear(num_features, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x


class ResNetBlock(nn.Module):
    """Realisation of `'resnet'` model block.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hid_factor: Dim of intermediate fc is increased times this factor in ResnetModel layer.
            drop_rate: Dropout rates.
            noise_std: Std of noise.
            act_fun: Activation function.
            use_bn: Use BatchNorm.
            use_noise: Use noise.
            device: Device to compute on.

    """

    def __init__(
        self,
        n_in: int,
        hid_factor: float,
        n_out: int,
        drop_rate: List[float] = [0.1, 0.1],
        noise_std: float = 0.05,
        act_fun: nn.Module = nn.ReLU,
        use_bn: bool = True,
        use_noise: bool = False,
        device: torch.device = torch.device("cuda:0"),
        **kwargs,
    ):
        super(ResNetBlock, self).__init__()
        self.features = nn.Sequential(OrderedDict([]))

        if use_bn:
            self.features.add_module("norm", nn.BatchNorm1d(n_in))
        if use_noise:
            self.features.add_module("noise", GaussianNoise(noise_std, device))

        self.features.add_module("dense1", nn.Linear(n_in, int(hid_factor * n_in)))
        self.features.add_module("act1", act_fun())

        if drop_rate[0]:
            self.features.add_module("drop1", nn.Dropout(p=drop_rate[0]))

        self.features.add_module("dense2", nn.Linear(int(hid_factor * n_in), n_out))

        if drop_rate[1]:
            self.features.add_module("drop2", nn.Dropout(p=drop_rate[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = self.features(x)
        return x


class ResNetModel(nn.Module):
    """The ResNet model from https://github.com/Yura52/rtdl.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hid_factor: Dim of intermediate fc is increased times this factor in ResnetModel layer.
            drop_rate: Dropout rate for each layer separately or altogether.
            noise_std: Std of noise.
            act_fun: Activation function.
            num_init_features: If not none add fc layer before model with certain dim.
            use_bn: Use BatchNorm.
            use_noise: Use noise.
            device: Device to compute on.

    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        hid_factor: List[float] = [2, 2],
        drop_rate: Union[float, List[float], List[List[float]]] = 0.1,
        noise_std: float = 0.05,
        act_fun: nn.Module = nn.ReLU,
        num_init_features: Optional[int] = None,
        use_bn: bool = True,
        use_noise: bool = False,
        device: torch.device = torch.device("cuda:0"),
        **kwargs,
    ):
        super(ResNetModel, self).__init__()
        if isinstance(drop_rate, float):
            drop_rate = [[drop_rate, drop_rate]] * len(hid_factor)
        elif isinstance(drop_rate, list) and len(drop_rate) == 2:
            drop_rate = [drop_rate] * len(hid_factor)
        else:
            assert (
                len(drop_rate) == len(hid_factor) and len(drop_rate[0]) == 2
            ), "Wrong number hidden_sizes/drop_rates. Must be equal."

        num_features = n_in if num_init_features is None else num_init_features
        self.dense0 = nn.Linear(n_in, num_features) if num_init_features is not None else nn.Identity()
        self.features1 = nn.Sequential(OrderedDict([]))

        for i, hd_factor in enumerate(hid_factor):
            block = ResNetBlock(
                n_in=num_features,
                hid_factor=hd_factor,
                n_out=num_features,
                drop_rate=drop_rate[i],
                noise_std=noise_std,
                act_fun=act_fun,
                use_bn=use_bn,
                use_noise=use_noise,
                device=device,
            )
            self.features1.add_module("resnetblock%d" % (i + 1), block)

        self.features2 = nn.Sequential(OrderedDict([]))
        if use_bn:
            self.features2.add_module("norm", nn.BatchNorm1d(num_features))

        self.features2.add_module("act", act_fun())
        self.fc = nn.Linear(num_features, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = self.dense0(x)
        identity = x
        for name, layer in self.features1.named_children():
            if name != "resnetblock1":
                x += identity
                identity = x
            x = layer(x)

        x = self.features2(x)
        x = self.fc(x)
        return x.view(x.shape[0], -1)


class SNN(nn.Module):
    """Realisation of `'snn'` model.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hidden_size: List of hidden dims.
            drop_rate: Dropout rate for each layer separately or altogether.
            num_init_features: If not none add fc layer before model with certain dim.

    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_size: List[int] = [512, 512, 512],
        num_init_features: Optional[int] = None,
        drop_rate: Union[float, List[float]] = 0.1,
        **kwargs,
    ):
        super().__init__()
        if isinstance(drop_rate, float):
            drop_rate = [drop_rate] * len(hidden_size)

        num_features = n_in if num_init_features is None else num_init_features
        layers = OrderedDict([])
        if num_init_features is not None:
            layers["dense-1"] = nn.Linear(n_in, num_features, bias=False)

        for i in range(len(hidden_size) - 1):
            layers[f"dense{i}"] = nn.Linear(num_features, hidden_size[i], bias=False)
            layers[f"selu_{i}"] = nn.SELU()

            if drop_rate[i]:
                layers[f"dropout_{i}"] = nn.AlphaDropout(p=drop_rate[i])
            num_features = hidden_size[i]

        layers[f"dense_{i}"] = nn.Linear(hidden_size[i], n_out, bias=True)
        self.network = nn.Sequential(layers)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = self.network(x)
        x = x.view(x.shape[0], -1)
        return x

    def reset_parameters(self):
        """Init weights."""
        for layer in self.network:
            if not isinstance(layer, nn.Linear):
                continue
            nn.init.normal_(layer.weight, std=1 / np.sqrt(layer.out_features))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)


"""Different Pooling strategies for sequence data."""


class SequenceAbstractPooler(nn.Module):
    """Abstract pooling class."""

    def __init__(self):
        super(SequenceAbstractPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Forward-call."""
        return self.forward(*args, **kwargs)


class SequenceClsPooler(SequenceAbstractPooler):
    """CLS token pooling."""

    def __init__(self):
        super(SequenceClsPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        return x[..., 0, :]


class SequenceMaxPooler(SequenceAbstractPooler):
    """Max value pooling."""

    def __init__(self):
        super(SequenceMaxPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = x.masked_fill(~x_mask, -float("inf"))
        values, _ = torch.max(x, dim=-2)
        return values


class SequenceSumPooler(SequenceAbstractPooler):
    """Sum value pooling."""

    def __init__(self):
        super(SequenceSumPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = x.masked_fill(~x_mask, 0)
        values = torch.sum(x, dim=-2)
        return values


class SequenceAvgPooler(SequenceAbstractPooler):
    """Mean value pooling."""

    def __init__(self):
        super(SequenceAvgPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = x.masked_fill(~x_mask, 0)
        x_active = torch.sum(x_mask, dim=-2)
        x_active = x_active.masked_fill(x_active == 0, 1)
        values = torch.sum(x, dim=-2) / x_active.data
        return values


class SequenceIndentityPooler(SequenceAbstractPooler):
    """Identity pooling."""

    def __init__(self):
        super(SequenceIndentityPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        return x


class SequenceConcatPooler(SequenceAbstractPooler):
    """Concat pooling."""

    def __init__(self):
        super(SequenceConcatPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        pooler1 = SequenceClsPooler()
        pooler2 = SequenceAvgPooler()
        x1 = pooler1(x, x_mask)
        x2 = pooler2(x, x_mask)
        values = torch.cat((x1, x2), dim=1)
        return values


pooling_by_name = {
    "mean": SequenceAvgPooler,
    "sum": SequenceSumPooler,
    "max": SequenceMaxPooler,
    "concat": SequenceConcatPooler,
    "cls": SequenceClsPooler,
}


class NODE(nn.Module):
    """The NODE model from https://github.com/Qwicen.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            layer_dim: num trees in one layer.
            num_layers: number of forests.
            tree_dim: number of response channels in the response of individual tree.
            choice_function: str `entmax` or `sparsmax`
            use_original_head use averaging as a head or put linear layer instead.
            depth: number of splits in every tree.
            drop_rate: Dropout rate for each layer altogether.
            act_fun: Activation function.
            num_init_features: If not none add fc layer before model with certain dim.
            use_bn: Use BatchNorm.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        layer_dim: int = 2048,
        num_layers: int = 1,
        tree_dim: int = 1,
        choice_function="entmax",
        use_original_head: bool = False,
        depth: int = 6,
        drop_rate: float = 0.0,
        act_fun: nn.Module = nn.ReLU,
        num_init_features: Optional[int] = None,
        use_bn: bool = True,
        **kwargs,
    ):
        super(NODE, self).__init__()
        num_features = n_in if num_init_features is None else num_init_features
        self.dense0 = nn.Linear(n_in, num_features) if num_init_features is not None else nn.Identity()
        self.features1 = nn.Sequential(OrderedDict([]))
        block = DenseODSTBlock(
            input_dim=num_features,
            layer_dim=layer_dim,
            num_layers=num_layers,
            tree_dim=tree_dim if not use_original_head else n_out,
            depth=depth,
            choice_function=choice_function,
            input_dropout=drop_rate,
            flatten_output=not use_original_head,
        )
        self.features1.add_module("ODSTForestblock%d", block)
        self.features2 = nn.Sequential(OrderedDict([]))
        if use_original_head:
            last_layer = MeanPooling(n_out, dim=-2)
            self.features2.add_module("head", last_layer)
        else:
            if use_bn:
                self.features2.add_module("norm", nn.BatchNorm1d(layer_dim * num_layers * tree_dim))
            self.features2.add_module("act", act_fun())
            fc = nn.Linear(layer_dim * num_layers * tree_dim, n_out)
            self.features2.add_module("fc", fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = self.dense0(x)
        x = self.features1(x)
        x = self.features2(x)
        return x.view(x.shape[0], -1)


class AutoInt(nn.Module):
    """The AutoInt model from https://github.com/jrfiedler/xynn.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            layer_dim: num trees in one layer.
            num_layers: number of forests.
            tree_dim: number of response channels in the response of individual tree.
            use_original_head use averaging as a head or put linear layer instead.
            depth: number of splits in every tree.
            drop_rate: Dropout rate for each layer altogether.
            act_fun: Activation function.
            num_init_features: If not none add fc layer before model with certain dim.
            use_bn: Use BatchNorm.
    """

    def __init__(
        self,
        n_in: int,
        embedding_size: int,
        n_out: int = 1,
        attn_embedding_size: int = 8,
        attn_num_layers: int = 3,
        attn_num_heads: int = 2,
        attn_activation: Optional[Type[nn.Module]] = None,
        attn_use_residual: bool = True,
        attn_dropout: float = 0.1,
        attn_normalize: bool = True,
        attn_use_mlp: bool = True,
        mlp_hidden_sizes: Union[int, Tuple[int, ...], List[int]] = (512, 256, 128, 64),
        mlp_activation: Type[nn.Module] = nn.LeakyReLU,
        mlp_use_bn: bool = True,
        mlp_bn_momentum: float = 0.1,
        mlp_ghost_batch: Optional[int] = 16,
        mlp_dropout: float = 0.0,
        mlp_use_skip: bool = True,
        use_leaky_gate: bool = True,
        weighted_sum: bool = True,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        super(AutoInt, self).__init__()
        super().__init__()
        device = torch.device(device)

        if use_leaky_gate:
            self.attn_gate = LeakyGate(n_in * embedding_size, device=device)
        else:
            self.attn_gate = nn.Identity()

        self.attn_interact = AttnInteractionBlock(
            field_input_size=embedding_size,
            field_output_size=attn_embedding_size,
            num_layers=attn_num_layers,
            num_heads=attn_num_heads,
            activation=attn_activation,
            use_residual=attn_use_residual,
            dropout=attn_dropout,
            normalize=attn_normalize,
            ghost_batch_size=mlp_ghost_batch,
            device=device,
        )

        self.attn_final = MLP(
            n_in=n_in * attn_embedding_size * attn_num_heads,
            hidden_size=(mlp_hidden_sizes if mlp_hidden_sizes and attn_use_mlp else []),
            n_out=n_out,
            act_fun=mlp_activation,
            drop_rate=mlp_dropout,
            use_bn=mlp_use_bn,
            bn_momentum=mlp_bn_momentum,
            ghost_batch=mlp_ghost_batch,
            leaky_gate=use_leaky_gate,
            use_skip=mlp_use_skip,
            device=device,
        )

        if mlp_hidden_sizes:
            self.mlp = MLP(
                n_in=n_in * embedding_size,
                hidden_size=mlp_hidden_sizes,
                n_out=n_out,
                act_fun=mlp_activation,
                drop_rate=mlp_dropout,
                use_bn=mlp_use_bn,
                bn_momentum=mlp_bn_momentum,
                ghost_batch=mlp_ghost_batch,
                leaky_gate=use_leaky_gate,
                use_skip=mlp_use_skip,
                device=device,
            )
            self.use_skip = True
            if weighted_sum:
                self.mix = nn.Parameter(torch.tensor([0.0], device=device))
            else:
                self.mix = torch.tensor([0.0], device=device)
        else:
            self.mlp = None
            self.mix = None

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        """Transform the input tensor.

        Args:
            embedded : torch.Tensor
                embedded fields

        Returns:
            torch.Tensor

        """
        out = self.attn_gate(embedded)
        out = self.attn_interact(out)
        out = self.attn_final(out.reshape((out.shape[0], -1)))
        if self.mlp is not None:
            embedded_2d = embedded.reshape((embedded.shape[0], -1))
            mix = torch.sigmoid(self.mix)
            out = mix * out + (1 - mix) * self.mlp(embedded_2d)
        return out


class FTTransformer(nn.Module):
    """FT Transformer (https://arxiv.org/abs/2106.11959v2) from https://github.com/lucidrains/tab-transformer-pytorch/tree/main.

    Args:
            pooling: Pooling used for the last step.
            n_out: Output dimension, 1 for binary prediction.
            embedding_size: Embeddings size.
            depth: Number of Attention Blocks inside Transformer.
            heads: Number of heads in Attention.
            attn_dropout: Post-Attention dropout.
            ff_dropout: Feed-Forward Dropout.
            dim_head: Attention head dimension.
            num_enc_layers: Number of Transformer layers.
            device: Device to compute on.
    """

    def __init__(
        self,
        *,
        pooling: str = "mean",
        n_out: int = 1,
        embedding_size: int = 32,
        depth: int = 4,
        heads: int = 1,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        dim_head: int = 32,
        num_enc_layers: int = 2,
        device: Union[str, torch.device] = "cuda:0",
        **kwargs,
    ):
        super(FTTransformer, self).__init__()
        self.device = device
        self.pooling = pooling_by_name[pooling]()

        # transformer
        self.transformer = nn.Sequential(
            *nn.ModuleList(
                [
                    Transformer(
                        dim=embedding_size,
                        depth=depth,
                        heads=heads,
                        dim_head=dim_head,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                    )
                    for _ in range(num_enc_layers)
                ]
            )
        )

        # to logits
        if pooling == "concat":
            self.to_logits = nn.Sequential(nn.BatchNorm1d(embedding_size * 2), nn.Linear(embedding_size * 2, n_out))
        else:
            self.to_logits = nn.Sequential(nn.BatchNorm1d(embedding_size), nn.Linear(embedding_size, n_out))

        self.cls_token = nn.Embedding(2, embedding_size)

    def forward(self, embedded):
        """Transform the input tensor.

        Args:
            embedded : torch.Tensor
                embedded fields

        Returns:
            torch.Tensor

        """
        cls_token = torch.unsqueeze(
            self.cls_token(torch.ones(embedded.shape[0], dtype=torch.int).to(self.device)), dim=1
        )
        x = torch.cat((cls_token, embedded), dim=1)
        x = self.transformer(x)
        x_mask = torch.ones(x.shape, dtype=torch.bool).to(self.device)
        pool_tokens = self.pooling(x=x, x_mask=x_mask)
        if isinstance(self.pooling, SequenceIndentityPooler):
            pool_tokens = pool_tokens[:, 0]

        logits = self.to_logits(pool_tokens)
        return logits


class TabNet(torch.nn.Module):
    """Implementation of TabNet from https://github.com/dreamquark-ai/tabnet.

    Args:
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : torch matrix
            Matrix of size (n_groups, input_dim), m_ij = importance within group i of feature j
    """

    def __init__(
        self,
        n_in,
        n_out,
        n_d=32,
        n_a=32,
        n_steps=1,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="entmax",
        group_attention_matrix=None,
        **kwargs,
    ):
        super(TabNet, self).__init__()
        self.input_dim = n_in
        self.output_dim = n_out
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = nn.BatchNorm1d(self.input_dim, momentum=0.01)

        self.encoder = TabNetEncoder(
            input_dim=n_in,
            output_dim=n_out,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            group_attention_matrix=group_attention_matrix,
        )

        self.final_mapping = nn.Linear(n_d, n_out, bias=True)
        _initialize_non_glu(self.final_mapping, n_d, n_out)

    def forward(self, x):
        """Forward-pass."""
        res = 0
        steps_output, M_loss = self.encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        out = self.final_mapping(res)
        return out

    def forward_masks(self, x):
        """Magic forward-pass of encoder that returns masks."""
        return self.encoder.forward_masks(x)


class SAINT(nn.Module):
    """Implementation of Saint from https://github.com/yandex-research/tabular-dl-tabr.

    Args:
        n_in : int
            Number of features
        n_out : int or list of int for multi task classification
            Dimension of network output
        embedding_size : embedding_size
            Dimension of the embedding
        depth : int
            Number of Attention Blocks.
        heads : int
            Number of heads in Attention.
        dim_head : int
            Attention head dimension.
        mlp_hidden_mults : int | tuple[int]
            Multiply hidden state of MLP.
        ffn_mult : int
            Multiply hidden state of feed forward layer.
        attn_dropout : float
            Post-Attention dropout.
        ff_dropout : int
            Feed-Forward Dropout.
        mlp_dropout : float
            MLP Dropout.
        attentiontype : str
            Either "colrow" or "row" : this is the masking attention to use
        device : torch.device
        kwargs : kwargs
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        embedding_size: int = 10,
        depth: int = 2,
        heads: int = 8,
        dim_head=16,
        mlp_hidden_mults=(4, 2),
        ffn_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        mlp_dropout=0.0,
        attentiontype="colrow",
        pooling: str = "cls",
        device: torch.device = torch.device("cuda:0"),
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.cls_token = nn.Embedding(2, embedding_size)
        self.attentiontype = attentiontype
        if attentiontype == "col":
            self.transformer = ColTransformer(
                dim=embedding_size,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )
        elif attentiontype in ["row", "colrow"]:
            self.transformer = RowColTransformer(
                dim=embedding_size,
                nfeats=n_in + 1,  # num featurs
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                ffn_mult=ffn_mult,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype,
            )

        l_rate = (n_in + 1) // 8  # input_size = (dim * self.num_categories)  + (dim * num_continuous)
        hidden_dimensions = list(map(lambda t: l_rate * t, mlp_hidden_mults))
        self.pooling = pooling_by_name[pooling]()
        self.mlp = MLP(
            n_in=embedding_size * 2 if pooling == "concat" else embedding_size,
            n_out=n_out,
            hidden_size=hidden_dimensions,
            drop_rate=mlp_dropout,
            use_bn=False,
            dropout_first=False,
        )

    def forward(self, embedded: torch.Tensor, bs: int) -> torch.Tensor:
        """Transform the input tensor.

        Args:
            embedded : torch.Tensor
                embedded fields
            bs : batch size without sapler`s part

        Returns:
            torch.Tensor

        """
        mask = torch.zeros((len(embedded), len(embedded)), device=self.device, dtype=torch.bool)
        mask[torch.arange(bs), torch.arange(bs)] = 1
        # NOTE that it was:
        # mask[:bs, bs:] = 1
        # mask[bs:, bs:] = 1
        # probably misprint
        mask[:bs, bs:] = 1
        mask[bs:, :bs] = 1

        cls_token = torch.unsqueeze(
            self.cls_token(torch.ones(embedded.shape[0], dtype=torch.int).to(self.device)), dim=1
        )
        x = torch.cat((cls_token, embedded), dim=1)
        x = self.transformer(x, mask_samples=mask)

        # NOTE modified to simple X -> Y supervised model

        # cat_outs = self.mlp1(x[:,:self.num_categories,:])
        # con_outs = self.mlp2(x[:,self.num_categories:,:])
        # return cat_outs, con_outs
        x_mask = torch.ones(x.shape, dtype=torch.bool).to(self.device)
        pool_tokens = self.pooling(x=x, x_mask=x_mask)
        logits = self.mlp(pool_tokens)
        return logits
