# Dev Setup

## Conda Environment

Create the development environment from the repo root:

```bash
mamba env create -f environment.yml
conda activate dada2-gpu-dev
```

## CPU-Only Build

Build the standalone shared library without CUDA:

```bash
make clean libdada2.so
```

## GPU-Enabled Build

The Conda environment installs `nvcc`, CUDA headers, and `libcudart`.
Point `CUDA_HOME` at the active Conda prefix when building:

```bash
CUDA_HOME=$CONDA_PREFIX make clean libdada2.so
```

## Quick Checks

Verify the Python bindings can load:

```bash
python - <<'PY'
import sys
sys.path.insert(0, '.')
from py._cdada import gpu_available
print("gpu_available =", gpu_available())
PY
```

Run Python tests:

```bash
pytest
```

## Notes

- CPU-only development does not require the CUDA packages in `environment.yml`.
- GPU builds require a working NVIDIA driver in addition to the Conda CUDA toolchain.
- For Conda-provided CUDA, `CUDA_HOME=$CONDA_PREFIX` is the expected build setting.

