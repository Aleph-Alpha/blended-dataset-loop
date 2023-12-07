import json
import numpy as np


def load_blended_indices(prefix_path):
    meta = json.load(open(f"{prefix_path}.meta.json"))
    inputs = json.load(open(f"{prefix_path}.input.json"))
    return inputs, np.memmap(f"{prefix_path}.result.bin", dtype=meta["dtype"], mode='r', shape=tuple(meta["shape"]))



if __name__ == '__main__':
    inputs, mmap = load_blended_indices("./example")
    print(inputs)
    print(mmap)
