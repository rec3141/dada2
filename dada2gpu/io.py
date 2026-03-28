"""FASTQ reading and dereplication."""

import gzip
import numpy as np


def derep_fastq(filepath, verbose=False):
    """Dereplicate a FASTQ file.

    Returns:
        dict with keys:
            seqs: list[str], unique sequences sorted by abundance (descending)
            abundances: numpy int32 array
            quals: numpy float64 array (n_uniques x max_seqlen), average quality
            map: numpy int32 array, maps each read to its unique index (0-indexed)
    """
    # Read entire file at once (much faster than line-by-line for gzip)
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rb") as f:
        raw = f.read()

    # Split into lines, extract every 4th (seq) and every 4th+3 (qual)
    lines = raw.split(b'\n')
    # Remove trailing empty line
    if lines and lines[-1] == b'':
        lines.pop()
    n_reads = len(lines) // 4

    # Phase 1: Build dedup index from sequences
    seq_to_idx = {}
    first_seen = []
    counts_list = []
    read_uid = np.empty(n_reads, dtype=np.int32)

    for i in range(n_reads):
        seq = lines[i * 4 + 1].upper()  # bytes
        idx = seq_to_idx.get(seq)
        if idx is None:
            idx = len(first_seen)
            seq_to_idx[seq] = idx
            first_seen.append(seq)
            counts_list.append(0)
        counts_list[idx] += 1
        read_uid[i] = idx

    n_uniques = len(first_seen)
    counts = np.array(counts_list, dtype=np.int32)
    maxlen = max(len(s) for s in first_seen) if first_seen else 0

    # Phase 2: Accumulate quality scores using numpy frombuffer
    qual_sums = np.zeros((n_uniques, maxlen), dtype=np.float64)

    for i in range(n_reads):
        uid = read_uid[i]
        qline = lines[i * 4 + 3]
        q = np.frombuffer(qline, dtype=np.uint8).astype(np.float64)
        q -= 33.0
        slen = len(q)
        qual_sums[uid, :slen] += q

    # Phase 3: Sort by abundance descending
    sort_idx = np.argsort(-counts)
    sorted_seqs = [first_seen[i].decode('ascii') for i in sort_idx]
    sorted_counts = counts[sort_idx]
    sorted_quals = qual_sums[sort_idx]

    # Average and NaN-pad
    for i in range(n_uniques):
        slen = len(sorted_seqs[i])
        if sorted_counts[i] > 0:
            sorted_quals[i, :slen] /= sorted_counts[i]
        sorted_quals[i, slen:] = np.nan

    # Remap read indices
    inv_sort = np.empty(n_uniques, dtype=np.int32)
    inv_sort[sort_idx] = np.arange(n_uniques, dtype=np.int32)
    rmap = inv_sort[read_uid]

    if verbose:
        print(f"Read {n_reads} reads, {n_uniques} unique sequences")

    return {
        "seqs": sorted_seqs,
        "abundances": sorted_counts,
        "quals": sorted_quals,
        "map": rmap,
    }
