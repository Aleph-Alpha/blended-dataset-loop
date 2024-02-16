import numpy as np
import blended_dataset_loop
import json

from pathlib import Path


def expect_input(prefix_path: Path, expected):
    with open(f"{prefix_path}.input.json") as file:
        actual = json.load(file)
    assert actual == expected


def expect_meta(prefix_path: Path, expected):
    with open(f"{prefix_path}.meta.json") as file:
        actual = json.load(file)
    assert actual == expected


def expect_bin(prefix_path: Path, expected):
    with open(f"{prefix_path}.meta.json") as file:
        meta = json.load(file)
    contents = np.fromfile(f"{prefix_path}.bin", dtype=meta["dtype"]).reshape(tuple(meta["shape"]))
    assert np.array_equal(contents, expected)


def test_simple(tmp_path: Path):
    prefix_path = tmp_path / "simple"
    number_to_sample = np.array([1, 2, 3, 4], dtype="uint64")
    blended_dataset_loop.sample(number_to_sample, str(prefix_path))
    expect_input(prefix_path, [1,2,3,4])
    expect_meta(prefix_path, {"dtype":"uint64","total_count":10,"shape":[10,2]})
    expect_bin(prefix_path, [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [2, 1], [1, 1], [3, 2], [2, 2], [3, 3]])
