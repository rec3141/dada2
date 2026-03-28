"""Main DADA2 denoising pipeline."""

import sys
import os
import numpy as np
from . import _cdada
from .io import derep_fastq
from .error import loess_errfun, get_initial_err

DADA_OPTS = {
    "OMEGA_A": 1e-40,
    "OMEGA_P": 1e-4,
    "OMEGA_C": 1e-40,
    "DETECT_SINGLETONS": False,
    "USE_KMERS": True,
    "KDIST_CUTOFF": 0.42,
    "MAX_CONSIST": 10,
    "MATCH": 5,
    "MISMATCH": -4,
    "GAP_PENALTY": -8,
    "BAND_SIZE": 16,
    "VECTORIZED_ALIGNMENT": True,
    "MAX_CLUST": 0,
    "MIN_FOLD": 1,
    "MIN_HAMMING": 1,
    "MIN_ABUNDANCE": 1,
    "USE_QUALS": True,
    "HOMOPOLYMER_GAP_PENALTY": None,
    "SSE": 2,
    "GAPLESS": True,
    "GREEDY": True,
}


def _run_one_sample(args):
    """Worker function for parallel dada() calls.

    Must be a top-level function for multiprocessing pickling.
    Re-imports _cdada in each subprocess (ctypes can't be pickled).
    """
    drp, err, opts, max_clust = args
    from dada2gpu import _cdada as _cd

    seqs = drp["seqs"]
    abundances = drp["abundances"]
    quals = drp["quals"]
    nraw = len(seqs)

    if nraw == 0:
        return {"cluster_seqs": [], "cluster_abunds": np.array([]),
                "trans": np.zeros((16, err.shape[1]), dtype=np.int32),
                "map": np.array([]), "pval": np.array([])}

    # Extend error matrix if needed
    max_q_obs = int(np.nanmax(quals)) + 1 if not np.all(np.isnan(quals)) else 0
    erri = err.copy()
    if max_q_obs > erri.shape[1]:
        extra = np.tile(erri[:, -1:], (1, max_q_obs - erri.shape[1]))
        erri = np.hstack([erri, extra])

    homo_gap = opts["GAP_PENALTY"] if opts["HOMOPOLYMER_GAP_PENALTY"] is None else opts["HOMOPOLYMER_GAP_PENALTY"]

    res = _cd.run_dada(
        seqs, abundances, erri, quals,
        match=opts["MATCH"], mismatch=opts["MISMATCH"], gap_pen=opts["GAP_PENALTY"],
        use_kmers=opts["USE_KMERS"], kdist_cutoff=opts["KDIST_CUTOFF"],
        band_size=opts["BAND_SIZE"],
        omega_a=opts["OMEGA_A"], omega_p=opts["OMEGA_P"], omega_c=opts["OMEGA_C"],
        detect_singletons=opts["DETECT_SINGLETONS"], max_clust=max_clust,
        min_fold=opts["MIN_FOLD"], min_hamming=opts["MIN_HAMMING"],
        min_abund=opts["MIN_ABUNDANCE"],
        use_quals=opts["USE_QUALS"], vectorized_alignment=opts["VECTORIZED_ALIGNMENT"],
        homo_gap_pen=homo_gap,
        multithread=False, verbose=False,
        sse=opts["SSE"], gapless=opts["GAPLESS"], greedy=opts["GREEDY"],
    )

    res["denoised"] = {seq: ab for seq, ab in
                       zip(res["cluster_seqs"], res["cluster_abunds"])}
    return res


def set_dada_opt(**kwargs):
    for k, v in kwargs.items():
        if k in DADA_OPTS:
            DADA_OPTS[k] = v
        else:
            raise KeyError(f"Unknown DADA option: {k}")


def get_dada_opt(key=None):
    if key is None:
        return dict(DADA_OPTS)
    return DADA_OPTS[key]


def dada(derep, err=None, error_estimation_function=None, self_consist=False,
         verbose=True, **opts):
    """Run DADA2 denoising on one or more dereplicated samples.

    Args:
        derep: dict from derep_fastq(), or list of dicts, or FASTQ filepath(s)
        err: numpy array (16, ncol) error matrix, or None for self-consistent learning
        error_estimation_function: callable(trans) -> err_matrix, default loess_errfun
        self_consist: bool, iterate until error model converges
        verbose: bool

    Returns:
        dict (single sample) or list of dicts, each with:
            denoised: dict {seq: abundance}
            cluster_seqs, cluster_abunds, trans, map, pval, err_in, err_out
    """
    if error_estimation_function is None:
        error_estimation_function = loess_errfun

    # Normalize input to list of derep dicts
    if isinstance(derep, str):
        derep = [derep_fastq(derep, verbose=verbose)]
    elif isinstance(derep, dict):
        derep = [derep]
    elif isinstance(derep, list) and len(derep) > 0 and isinstance(derep[0], str):
        derep = [derep_fastq(f, verbose=verbose) for f in derep]

    single = len(derep) == 1

    # Merge options
    o = dict(DADA_OPTS)
    o.update(opts)

    homo_gap = o["GAP_PENALTY"] if o["HOMOPOLYMER_GAP_PENALTY"] is None else o["HOMOPOLYMER_GAP_PENALTY"]

    # Initialize error matrix (matching R: all 1.0)
    initialize_err = False
    if self_consist and err is None:
        max_q = 0
        for d in derep:
            max_q = max(max_q, d["quals"].shape[1] if d["quals"].size > 0 else 41)
        err = get_initial_err(max(41, max_q))
        initialize_err = True

    if err is None:
        raise ValueError("Error matrix (err) must be provided unless self_consist=True")

    err_history = []
    nconsist = 0 if initialize_err else 1  # R starts at 0 for init, 1 otherwise

    # Determine parallelism strategy:
    # - GPU available: run sequentially (GPU is already fast, shared resource)
    # - CPU only: use ProcessPoolExecutor (ctypes holds GIL, threads don't help)
    use_gpu = _cdada.gpu_available()
    n_workers = int(os.environ.get("DADA2_WORKERS", "0"))
    if n_workers == 0:
        n_workers = min(len(derep), os.cpu_count() or 1)
    use_parallel = not use_gpu and len(derep) > 1 and n_workers > 1

    # Create a persistent process pool (reused across self-consistency iterations)
    pool = None
    if use_parallel:
        from concurrent.futures import ProcessPoolExecutor
        os.environ["OMP_NUM_THREADS"] = "1"
        pool = ProcessPoolExecutor(max_workers=n_workers)
    elif use_gpu:
        os.environ["OMP_NUM_THREADS"] = "1"

    while True:
        nconsist += 1
        if verbose and self_consist:
            sys.stdout.write(f"   selfConsist step {nconsist}")
            sys.stdout.flush()

        # R uses MAX_CLUST=1 on the initialization pass (nconsist==1 after init)
        max_clust_iter = 1 if initialize_err else o["MAX_CLUST"]

        # Extend error matrix once for all samples (upstream optimization)
        max_q_all = max((int(np.nanmax(d["quals"])) + 1 if d["quals"].size and not np.all(np.isnan(d["quals"])) else 0) for d in derep)
        erri = err.copy()
        if max_q_all > erri.shape[1]:
            extra = np.tile(erri[:, -1:], (1, max_q_all - erri.shape[1]))
            erri = np.hstack([erri, extra])

        if use_parallel and pool is not None:
            # Parallel: dispatch each sample to a persistent worker pool
            work_args = [(drp, erri, o, max_clust_iter) for drp in derep]
            results = list(pool.map(_run_one_sample, work_args))

            if verbose and self_consist:
                sys.stdout.write("." * len(derep))
                sys.stdout.flush()
        else:
            # Sequential: single sample, GPU mode, or single worker
            results = []
            for i, drp in enumerate(derep):
                if verbose and self_consist:
                    sys.stdout.write(".")
                    sys.stdout.flush()

                res = _run_one_sample((drp, erri, o, max_clust_iter))
                results.append(res)

        trans_list = [r["trans"] for r in results]

        if verbose and self_consist:
            print()

        # Accumulate transitions
        cur_trans = _accumulate_trans(trans_list)

        # Estimate new error rates
        new_err = error_estimation_function(cur_trans)

        # Check convergence
        if not self_consist:
            break

        converged = any(np.allclose(new_err, h, atol=1e-10) for h in err_history)
        if converged:
            if verbose:
                print(f"Convergence after {nconsist} rounds.")
            break

        if nconsist >= o["MAX_CONSIST"]:
            if verbose:
                print(f"Self-consistency loop terminated before convergence.")
            break

        err_history.append(err.copy())
        err = new_err

        # After initialization pass: set self-transitions to 1.0 (matching R)
        if initialize_err:
            err[0, :] = 1.0   # A2A
            err[5, :] = 1.0   # C2C
            err[10, :] = 1.0  # G2G
            err[15, :] = 1.0  # T2T
            initialize_err = False

    # Shut down persistent pool
    if pool is not None:
        pool.shutdown(wait=False)

    # Attach error info to results
    for res in results:
        res["err_in"] = err
        res["err_out"] = new_err

    if single:
        return results[0]
    return results


def learn_errors(fastq_files, nbases=1e8, error_estimation_function=None,
                 verbose=True, **opts):
    """Learn error rates from FASTQ files.

    Args:
        fastq_files: list of FASTQ file paths
        nbases: target number of bases to use for learning
        error_estimation_function: callable, default loess_errfun
        verbose: bool

    Returns:
        numpy array (16, ncol) of learned error rates
    """
    if isinstance(fastq_files, str):
        fastq_files = [fastq_files]

    # Accumulate samples until we have enough bases
    dereps = []
    total_bases = 0
    for fl in fastq_files:
        drp = derep_fastq(fl, verbose=verbose)
        dereps.append(drp)
        n_reads = drp["abundances"].sum()
        seqlen = len(drp["seqs"][0]) if drp["seqs"] else 0
        total_bases += n_reads * seqlen
        if total_bases >= nbases:
            break

    if verbose:
        n_reads_total = sum(d["abundances"].sum() for d in dereps)
        print(f"{int(total_bases)} total bases in {n_reads_total} reads "
              f"from {len(dereps)} samples will be used for learning the error rates.")
        print("Initializing error rates to maximum possible estimate.")

    # Run dada with self-consistency
    results = dada(dereps, err=None, error_estimation_function=error_estimation_function,
                   self_consist=True, verbose=verbose, **opts)

    if isinstance(results, dict):
        results = [results]

    return results[0]["err_out"]


def _accumulate_trans(trans_list):
    """Sum transition matrices, zero-padding to max column count."""
    if not trans_list:
        return np.zeros((16, 0), dtype=np.int32)

    max_ncol = max(t.shape[1] for t in trans_list if t.size > 0)
    if max_ncol == 0:
        return np.zeros((16, 0), dtype=np.int32)

    acc = np.zeros((16, max_ncol), dtype=np.int64)
    for t in trans_list:
        if t.size == 0:
            continue
        nc = t.shape[1]
        acc[:, :nc] += t

    return acc.astype(np.int32)
