# Blended Dataset Loop

This repository contains a simple loop to compute a balanced ordering of dataset indices to train on.
The resulting ordering ensures the data distribution is similar among batches.
The loop is implemented in Rust for performance reasons and can be consumed as part of a Python package.

The package uses `cffi` (rather than e.g. `PyO3`) in order to be compatible with different Python versions.

## Requirements

* Conda (for Python)
* Cargo with nightly Rust

## Setup

Create a conda environment as follows:

```sh
conda create -n blended_dataset_loop python=3.9 -y
conda activate blended_dataset_loop
```

Install Rust nightly

```sh
rustup override set nightly-2024-02-03
```

Install the Python dev-dependencies:

```sh
pip install 'maturin[patchelf]'
pip install '.[dev]'
```

## Develop

After changing the Rust code, run:

```sh
maturin develop
```

or, for release mode

```sh
maturin develop --release
```
