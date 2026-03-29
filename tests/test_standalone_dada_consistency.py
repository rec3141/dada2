import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "py"

spec = importlib.util.spec_from_file_location(
    "dada2_py",
    PKG_ROOT / "__init__.py",
    submodule_search_locations=[str(PKG_ROOT)],
)
dada2_py = importlib.util.module_from_spec(spec)
sys.modules["dada2_py"] = dada2_py
assert spec.loader is not None
spec.loader.exec_module(dada2_py)

from dada2_py.io import derep_fastq


def test_standalone_abundances_match_assigned_reads():
    files = [
        str(REPO_ROOT / "inst" / "extdata" / "sam1F.fastq.gz"),
        str(REPO_ROOT / "inst" / "extdata" / "sam2F.fastq.gz"),
    ]

    py_dada = importlib.import_module("dada2_py.dada")
    _cdada = importlib.import_module("dada2_py._cdada")
    _cdada.gpu_available = lambda: False

    err = py_dada.learn_errors(files, verbose=False)
    derep = derep_fastq(files[0], verbose=False)
    res = py_dada.dada(derep, err=err, verbose=False)

    assigned = np.asarray(res["map"]) >= 0
    assigned_reads = int(derep["abundances"][assigned].sum())
    denoised_reads = int(sum(res["denoised"].values()))
    cluster_reads = int(np.asarray(res["cluster_abunds"]).sum())

    assert denoised_reads == assigned_reads
    assert cluster_reads == assigned_reads
