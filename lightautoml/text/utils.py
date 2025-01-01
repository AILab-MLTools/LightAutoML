"""Text utility script."""

import os
import random

from typing import Dict
from typing import List
from typing import Sequence

import numpy as np
import torch

from sklearn.utils.murmurhash import murmurhash3_32


_dtypes_mapping = {
    "label": "float",
    "cat": "long",
    "cont": "float",
    "weight": "float",
    "input_ids": "long",
    "attention_mask": "long",
    "token_type_ids": "long",
    "text": "float",  # embeddings
    "length": "long",
    "fold": "long"
}


def inv_sigmoid(x: np.ndarray) -> np.ndarray:
    """Inverse sigmoid transformation.

    Args:
        x: Input array.

    Returns:
        Transformed array.

    """
    return np.log(x / (1 - x))


def inv_softmax(x: np.ndarray) -> np.ndarray:
    """Variant of inverse softmax transformation with zero constant term.

    Args:
        x: Input array.

    Returns:
        Transformed array.

    """
    eps = 1e-7
    x = np.abs(x)
    arr = (x + eps) / (np.sum(x) + eps)
    arr = np.log(arr)
    return arr


def is_shuffle(stage: str) -> bool:
    """Whether shuffle input.

    Args:
        stage: Train, val, test.

    Returns:
        Bool value.

    """
    is_sh = {"train": True, "val": False, "test": False, "sampler": True}
    return is_sh[stage]


def seed_everything(seed: int = 42, deterministic: bool = True):
    """Set random seed and cudnn params.

    Args:
        seed: Random state.
        deterministic: cudnn backend.

    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True


def parse_devices(dvs, is_dp: bool = False) -> tuple:
    """Parse devices and convert first to the torch device.

    Args:
        dvs: List, string with device ids or torch.device.
        is_dp: Use data parallel - additionally returns device ids.

    Returns:
        First torch device and list of gpu ids.

    """
    device = []
    ids = []
    if (not torch.cuda.is_available()) or (dvs is None):
        return torch.device("cpu"), None

    if not isinstance(dvs, (list, tuple)):
        dvs = [dvs]

    for _device in dvs:
        if isinstance(_device, str):
            if _device.startswith("cuda:"):
                ids.append(int(_device.split("cuda:")[-1]))
            elif _device == "cuda":
                ids.append(0)
            elif _device == "cpu":
                return torch.device("cpu"), None
            else:
                ids.append(int(_device))
                _device = torch.device(int(_device))

        elif isinstance(_device, int):
            ids.append(_device)
            _device = torch.device("cuda:{}".format(_device))
        elif isinstance(_device, torch.device):
            if _device.type == "cpu":
                return _device, None
            else:
                if _device.index is None:
                    ids.append(0)
                else:
                    ids.append(_device.index)
        else:
            raise ValueError("Unknown device type: {}".format(_device))

        device.append(_device)

    return device[0], ids if (len(device) > 1) and is_dp else None


def custom_collate(batch: List[np.ndarray]) -> torch.Tensor:
    """Puts each data field into a tensor with outer dimension batch size."""
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        numel = sum([x.numel() for x in batch])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    else:
        return torch.from_numpy(np.array(batch)).float()


def collate_dict(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """custom_collate for dicts."""
    keys = list(batch[0].keys())
    transposed_data = list(map(list, zip(*[tuple([i[name] for name in i.keys()]) for i in batch])))
    return {key: custom_collate(transposed_data[n]) for n, key in enumerate(keys)}


def single_text_hash(x: str) -> str:
    """Get text hash.

    Args:
        x: Text.

    Returns:
        String text hash.

    """
    numhash = murmurhash3_32(x, seed=13)
    texthash = str(numhash) if numhash > 0 else "m" + str(abs(numhash))
    return texthash


def get_textarr_hash(x: Sequence[str]) -> str:
    """Get hash of array with texts.

    Args:
        x: Text array.

    Returns:
        Hash of array.

    """
    full_hash = single_text_hash(str(x))
    n = 0
    for text in x:
        if text != "":
            full_hash += "_" + single_text_hash(text)
            n += 1
            if n >= 3:
                break

    return full_hash
