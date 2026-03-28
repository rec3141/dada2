"""FASTQ reading and dereplication."""

import gzip
import numpy as np
from collections import defaultdict


def _open_fastq(filepath):
    """Open a FASTQ file, auto-detecting gzip compression."""
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rt")
    return open(filepath, "r")


def _parse_phred(qual_str):
    """Convert FASTQ quality string to Phred scores (Phred+33)."""
    return np.array([ord(c) - 33 for c in qual_str], dtype=np.float64)


def derep_fastq(filepath, verbose=False):
    """Dereplicate a FASTQ file.

    Returns:
        dict with keys:
            uniques: dict {sequence_str: abundance_int}
            quals: numpy array (n_uniques x max_seqlen), average quality per position
            map: list[int], maps each read to its unique index (0-indexed)
            seqs: list[str], unique sequences sorted by abundance (descending)
    """
    counts = defaultdict(int)
    qual_sums = defaultdict(lambda: None)
    read_map = []

    # Parse FASTQ (4 lines per record)
    n_reads = 0
    with _open_fastq(filepath) as f:
        while True:
            header = f.readline().rstrip()
            if not header:
                break
            seq = f.readline().rstrip().upper()
            f.readline()  # +
            qual_str = f.readline().rstrip()

            counts[seq] += 1
            q = _parse_phred(qual_str)
            if qual_sums[seq] is None:
                qual_sums[seq] = q.copy()
            else:
                qs = qual_sums[seq]
                if len(q) > len(qs):
                    new_qs = np.zeros(len(q), dtype=np.float64)
                    new_qs[:len(qs)] = qs
                    new_qs[:len(q)] += q  # overlap part
                    qual_sums[seq] = new_qs
                elif len(q) < len(qs):
                    qs[:len(q)] += q
                else:
                    qs += q
            read_map.append(seq)
            n_reads += 1

    if verbose:
        print(f"Read {n_reads} reads, {len(counts)} unique sequences")

    # Sort by abundance (descending)
    sorted_seqs = sorted(counts.keys(), key=lambda s: counts[s], reverse=True)

    # Build index
    seq_to_idx = {s: i for i, s in enumerate(sorted_seqs)}

    # Build quality matrix (average quals)
    maxlen = max(len(s) for s in sorted_seqs) if sorted_seqs else 0
    n_uniques = len(sorted_seqs)
    qual_mat = np.full((n_uniques, maxlen), np.nan, dtype=np.float64)
    abundances = np.zeros(n_uniques, dtype=np.int32)

    for i, seq in enumerate(sorted_seqs):
        ab = counts[seq]
        abundances[i] = ab
        qs = qual_sums[seq]
        slen = len(seq)
        # Average quality: sum / count
        qual_mat[i, :slen] = qs[:slen] / ab

    # Build read-to-unique map (0-indexed)
    rmap = np.array([seq_to_idx[s] for s in read_map], dtype=np.int32)

    return {
        "seqs": sorted_seqs,
        "uniques": {s: counts[s] for s in sorted_seqs},
        "abundances": abundances,
        "quals": qual_mat,
        "map": rmap,
    }
