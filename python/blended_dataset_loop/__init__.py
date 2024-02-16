from pathlib import PurePath
import numpy as np
from typing import Union

from .blended_dataset_loop import ffi, lib


def sample(number_to_sample: np.ndarray, prefix_path: Union[PurePath, str]):
    assert isinstance(
        number_to_sample, np.ndarray
    ), "expected `number_to_sample` to be a numpy array"
    assert (
        len(number_to_sample.shape) == 1
    ), "expected `number_to_sample` to be one-dimensional"
    assert isinstance(prefix_path, PurePath) or isinstance(
        prefix_path, str
    ), "expected `prefix_path` to be a PurePath or str"

    number_to_sample_len = len(number_to_sample)
    number_to_sample_buf = ffi.from_buffer(number_to_sample)
    number_to_sample_c = ffi.cast("uint64_t *", number_to_sample_buf)
    path_utf8 = str(prefix_path).encode("utf-8")
    prefix_path_c = ffi.cast("uint8_t *", ffi.from_buffer(path_utf8))

    lib.sample(number_to_sample_c, number_to_sample_len, prefix_path_c, len(path_utf8))


__all__ = ["sample"]
