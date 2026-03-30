# Dev Setup

## Conda Environment

Create the development environment from the repo root:

```bash
mamba env create -f environment.yml
conda activate dada2-dev
```

## CPU-Only Build

Build the standalone shared library:

```bash
make clean libdada2.so
```

## Quick Checks

Verify the Python bindings can load:

```bash
python - <<'PY'
import sys
sys.path.insert(0, '.')
import py
print("py package import ok")
PY
```

Run Python tests:

```bash
pytest
```

## Notes

- This branch is CPU-only.
