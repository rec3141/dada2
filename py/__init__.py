"""dada2 Python bindings - GPU-accelerated DADA2 amplicon denoising in Python."""

from .dada import dada, learn_errors, DADA_OPTS, set_dada_opt, get_dada_opt
from .io import derep_fastq
from .error import loess_errfun, loess_errfun_r, noqual_errfun, inflate_err
from ._cdada import gpu_available, nwalign, eval_pair, pair_consensus, rc
from .paired import merge_pairs
from .chimera import remove_bimera_denovo
from .utils import (
    assign_species,
    add_species,
    collapse_no_mismatch,
    make_sequence_table,
    plot_quality_profile,
    uniquesto_fasta,
    write_fasta,
    is_phix,
    seq_complexity,
    get_sequences,
    get_uniques,
)

__version__ = "0.1.0"
