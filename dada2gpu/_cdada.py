"""ctypes bindings to libdada2.so C API."""

import ctypes as ct
import numpy as np
import os

# Load the shared library
_lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libdada2.so")
if not os.path.exists(_lib_path):
    _lib_path = "libdada2.so"
_lib = ct.CDLL(_lib_path)


class DadaResult(ct.Structure):
    _fields_ = [
        ("nclust", ct.c_int),
        ("nraw", ct.c_int),
        ("maxlen", ct.c_int),
        ("cluster_seqs", ct.POINTER(ct.c_char_p)),
        ("cluster_abunds", ct.POINTER(ct.c_int)),
        ("cluster_n0", ct.POINTER(ct.c_int)),
        ("cluster_n1", ct.POINTER(ct.c_int)),
        ("cluster_nunq", ct.POINTER(ct.c_int)),
        ("cluster_pval", ct.POINTER(ct.c_double)),
        ("ncol_trans", ct.c_int),
        ("trans", ct.POINTER(ct.c_int)),
        ("map", ct.POINTER(ct.c_int)),
        ("pval", ct.POINTER(ct.c_double)),
    ]


_lib.dada2_run.restype = ct.POINTER(DadaResult)
_lib.dada2_run.argtypes = [
    ct.POINTER(ct.c_char_p),  # seqs
    ct.POINTER(ct.c_int),     # abundances
    ct.POINTER(ct.c_int),     # priors
    ct.c_int,                  # nraw
    ct.POINTER(ct.c_double),  # err_mat
    ct.c_int,                  # ncol_err
    ct.POINTER(ct.c_double),  # quals
    ct.c_int,                  # maxlen
    ct.c_int, ct.c_int, ct.c_int,  # match, mismatch, gap_pen
    ct.c_int, ct.c_double, ct.c_int,  # use_kmers, kdist_cutoff, band_size
    ct.c_double, ct.c_double, ct.c_double,  # omegaA, omegaP, omegaC
    ct.c_int, ct.c_int,       # detect_singletons, max_clust
    ct.c_double, ct.c_int, ct.c_int,  # min_fold, min_hamming, min_abund
    ct.c_int, ct.c_int, ct.c_int,  # use_quals, vectorized_alignment, homo_gap_pen
    ct.c_int, ct.c_int,       # multithread, verbose
    ct.c_int, ct.c_int, ct.c_int,  # SSE, gapless, greedy
]

_lib.dada2_result_free.restype = None
_lib.dada2_result_free.argtypes = [ct.POINTER(DadaResult)]

_lib.dada2_gpu_available.restype = ct.c_int
_lib.dada2_gpu_available.argtypes = []


def gpu_available():
    return bool(_lib.dada2_gpu_available())


def run_dada(seqs, abundances, err_mat, quals=None, priors=None,
             match=5, mismatch=-4, gap_pen=-8,
             use_kmers=True, kdist_cutoff=0.42, band_size=16,
             omega_a=1e-40, omega_p=1e-4, omega_c=1e-40,
             detect_singletons=False, max_clust=0,
             min_fold=1, min_hamming=1, min_abund=1,
             use_quals=True, vectorized_alignment=True, homo_gap_pen=-8,
             multithread=True, verbose=False,
             sse=2, gapless=True, greedy=True):
    """Call the C++ dada2 algorithm on dereplicated sequences.

    Args:
        seqs: list of str, unique DNA sequences (ACGT only)
        abundances: array-like of int, abundance per unique sequence
        err_mat: numpy array (16, ncol), error rate matrix, row-major
        quals: numpy array (nraw, maxlen) of avg quality scores, or None
        priors: array-like of int (0/1), or None

    Returns:
        dict with keys: cluster_seqs, cluster_abunds, trans, map, pval, etc.
    """
    nraw = len(seqs)
    if nraw == 0:
        return {"cluster_seqs": [], "cluster_abunds": np.array([], dtype=np.int32),
                "trans": np.zeros((16, 0), dtype=np.int32), "map": np.array([], dtype=np.int32),
                "pval": np.array([], dtype=np.float64)}

    # Convert sequences to C strings
    seq_arr = (ct.c_char_p * nraw)()
    for i, s in enumerate(seqs):
        seq_arr[i] = s.encode("ascii") if isinstance(s, str) else s

    # Abundances
    abund_arr = np.asarray(abundances, dtype=np.int32)

    # Priors
    if priors is None:
        prior_arr = np.zeros(nraw, dtype=np.int32)
    else:
        prior_arr = np.asarray(priors, dtype=np.int32)

    # Error matrix (16 x ncol, row-major, C-contiguous)
    err = np.ascontiguousarray(err_mat, dtype=np.float64)
    ncol_err = err.shape[1]

    # Quality matrix (C++ rounds to uint8_t internally via round())
    if quals is not None:
        q = np.ascontiguousarray(quals, dtype=np.float64)
        maxlen = q.shape[1]
        q_ptr = q.ctypes.data_as(ct.POINTER(ct.c_double))
    else:
        maxlen = max(len(s) for s in seqs)
        q_ptr = ct.POINTER(ct.c_double)()

    res_ptr = _lib.dada2_run(
        seq_arr,
        abund_arr.ctypes.data_as(ct.POINTER(ct.c_int)),
        prior_arr.ctypes.data_as(ct.POINTER(ct.c_int)),
        ct.c_int(nraw),
        err.ctypes.data_as(ct.POINTER(ct.c_double)),
        ct.c_int(ncol_err),
        q_ptr,
        ct.c_int(maxlen),
        ct.c_int(match), ct.c_int(mismatch), ct.c_int(gap_pen),
        ct.c_int(int(use_kmers)), ct.c_double(kdist_cutoff), ct.c_int(band_size),
        ct.c_double(omega_a), ct.c_double(omega_p), ct.c_double(omega_c),
        ct.c_int(int(detect_singletons)), ct.c_int(max_clust),
        ct.c_double(min_fold), ct.c_int(min_hamming), ct.c_int(min_abund),
        ct.c_int(int(use_quals)), ct.c_int(int(vectorized_alignment)),
        ct.c_int(homo_gap_pen),
        ct.c_int(int(multithread)), ct.c_int(int(verbose)),
        ct.c_int(sse), ct.c_int(int(gapless)), ct.c_int(int(greedy)),
    )

    if not res_ptr:
        raise RuntimeError("dada2_run returned NULL")

    res = res_ptr.contents
    nc = res.nclust
    nr = res.nraw

    # Extract results using fast buffer copies
    result = {
        "cluster_seqs": [res.cluster_seqs[i].decode("ascii") for i in range(nc)],
        "cluster_abunds": np.ctypeslib.as_array(res.cluster_abunds, shape=(nc,)).copy(),
        "cluster_n0": np.ctypeslib.as_array(res.cluster_n0, shape=(nc,)).copy(),
        "cluster_n1": np.ctypeslib.as_array(res.cluster_n1, shape=(nc,)).copy(),
        "cluster_nunq": np.ctypeslib.as_array(res.cluster_nunq, shape=(nc,)).copy(),
        "trans": np.ctypeslib.as_array(res.trans, shape=(16 * res.ncol_trans,)).copy().reshape(16, res.ncol_trans) if res.ncol_trans > 0 else np.zeros((16, 0), dtype=np.int32),
        "map": np.ctypeslib.as_array(res.map, shape=(nr,)).copy(),
        "pval": np.ctypeslib.as_array(res.pval, shape=(nr,)).copy(),
    }

    _lib.dada2_result_free(res_ptr)
    return result


# =========================================================================
# Taxonomy assignment
# =========================================================================

class TaxResult(ct.Structure):
    _fields_ = [
        ("nseq", ct.c_int),
        ("nlevel", ct.c_int),
        ("rval", ct.POINTER(ct.c_int)),
        ("rboot", ct.POINTER(ct.c_int)),
    ]


_lib.dada2_assign_taxonomy.restype = ct.POINTER(TaxResult)
_lib.dada2_assign_taxonomy.argtypes = [
    ct.POINTER(ct.c_char_p),  # seqs
    ct.c_int,                  # nseq
    ct.POINTER(ct.c_char_p),  # refs
    ct.c_int,                  # nref
    ct.POINTER(ct.c_int),     # ref_to_genus
    ct.POINTER(ct.c_int),     # genusmat
    ct.c_int,                  # ngenus
    ct.c_int,                  # nlevel
    ct.c_int,                  # verbose
]

_lib.dada2_tax_result_free.restype = None
_lib.dada2_tax_result_free.argtypes = [ct.POINTER(TaxResult)]


def run_taxonomy(seqs, refs, ref_to_genus, genusmat, ngenus, nlevel, verbose=True):
    """Run dada2 taxonomy assignment via C library.

    Args:
        seqs: list of query sequences (str)
        refs: list of reference sequences (str)
        ref_to_genus: numpy array (nref,) int32, 0-indexed genus ID per ref
        genusmat: numpy array (ngenus, nlevel) int32, genus-to-level mapping
        ngenus: int
        nlevel: int
        verbose: bool

    Returns:
        dict with:
            rval: numpy array (nseq,) int32, 1-indexed best genus per query (0=NA)
            rboot: numpy array (nseq, nlevel) int32, bootstrap counts
    """
    nseq = len(seqs)
    nref = len(refs)

    # Convert sequences to C strings
    seq_arr = (ct.c_char_p * nseq)()
    for i, s in enumerate(seqs):
        seq_arr[i] = s.encode("ascii") if isinstance(s, str) else s

    ref_arr = (ct.c_char_p * nref)()
    for i, s in enumerate(refs):
        ref_arr[i] = s.encode("ascii") if isinstance(s, str) else s

    # Reference-to-genus mapping
    rtg = np.ascontiguousarray(ref_to_genus, dtype=np.int32)

    # Genus matrix (row-major)
    gmat = np.ascontiguousarray(genusmat, dtype=np.int32)

    res_ptr = _lib.dada2_assign_taxonomy(
        seq_arr, nseq,
        ref_arr, nref,
        rtg.ctypes.data_as(ct.POINTER(ct.c_int)),
        gmat.ctypes.data_as(ct.POINTER(ct.c_int)),
        ngenus, nlevel,
        ct.c_int(int(verbose))
    )

    if not res_ptr:
        raise RuntimeError("dada2_assign_taxonomy returned NULL")

    res = res_ptr.contents
    result = {
        "rval": np.ctypeslib.as_array(res.rval, shape=(nseq,)).copy(),
        "rboot": np.ctypeslib.as_array(res.rboot, shape=(nseq * nlevel,)).copy().reshape(nseq, nlevel),
    }

    _lib.dada2_tax_result_free(res_ptr)
    return result
