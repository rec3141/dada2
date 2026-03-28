"""dada2gpu - GPU-accelerated DADA2 amplicon denoising in Python."""

from .dada import dada, learn_errors, DADA_OPTS, set_dada_opt, get_dada_opt
from .io import derep_fastq
from .error import loess_errfun, noqual_errfun, inflate_err
from ._cdada import gpu_available

__version__ = "0.1.0"
