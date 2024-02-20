import numpy as np
import blended_dataset_loop
import json

from pathlib import Path


def expect_input(cache_filename_stem: Path, expected):
    with open(f"{cache_filename_stem}.input.json") as file:
        actual = json.load(file)
    assert actual == expected


def expect_meta(cache_filename_stem: Path, expected):
    with open(f"{cache_filename_stem}.meta.json") as file:
        actual = json.load(file)
    assert actual == expected


def expect_bin(cache_filename_stem: Path, expected):
    with open(f"{cache_filename_stem}.meta.json") as file:
        meta = json.load(file)
    contents = np.fromfile(f"{cache_filename_stem}.bin", dtype=meta["dtype"]).reshape(
        tuple(meta["shape"])
    )
    assert np.array_equal(contents, expected)


def test_simple(tmp_path: Path):
    cache_filename_stem = tmp_path / "simple"
    number_to_sample = np.array([1, 2, 3, 4], dtype="uint64")
    blended_dataset_loop.sample(number_to_sample, str(cache_filename_stem))
    expect_input(cache_filename_stem, [1, 2, 3, 4])
    expect_meta(
        cache_filename_stem, {"dtype": "uint64", "total_count": 10, "shape": [10, 2]}
    )
    expect_bin(
        cache_filename_stem,
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [3, 1],
            [2, 1],
            [1, 1],
            [3, 2],
            [2, 2],
            [3, 3],
        ],
    )
