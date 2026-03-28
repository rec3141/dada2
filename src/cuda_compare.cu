/*
 * cuda_compare.cu - GPU-accelerated sequence comparison for DADA2
 *
 * Implements a fused kernel that performs kmer distance screening,
 * banded Needleman-Wunsch alignment, and lambda (error probability)
 * computation for all raw sequences against a cluster center.
 *
 * This replaces b_compare_parallel() from cluster.cpp on GPU.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include "cuda_compare.h"

/* ---------------- Constants ---------------- */

#define KMER_SIZE 5
#define N_KMER 1024          /* 4^KMER_SIZE */
#define MAX_SEQLEN 300       /* Max sequence length for GPU kernel local arrays */
#define BLOCK_SIZE 32        /* Threads per block (reduced for large local memory in NW) */
#define GAP_GLYPH 9999

/* Error matrix in constant memory: 16 transitions x max 93 quality cols */
#define MAX_ERR_NCOL 93
__constant__ double d_err_mat[16 * MAX_ERR_NCOL];
__constant__ unsigned int d_err_ncol;

/* ---------------- GPU Context ---------------- */

struct GpuContext {
    /* Device pointers */
    char *d_seqs;              /* nraw * max_seqlen */
    uint8_t *d_quals;          /* nraw * max_seqlen */
    uint8_t *d_kmer8;          /* nraw * N_KMER */
    unsigned int *d_lengths;   /* nraw */
    unsigned int *d_reads;     /* nraw */
    int *d_locks;              /* nraw */

    /* Output buffers */
    double *d_lambdas;         /* nraw */
    unsigned int *d_hammings;  /* nraw */

    /* Dimensions */
    unsigned int max_nraw;
    unsigned int max_seqlen;
    unsigned int nraw;         /* current number of raws */
};

/* ---------------- Lifecycle ---------------- */

extern "C" GpuContext* gpu_context_create(unsigned int max_nraw, unsigned int max_seqlen) {
    GpuContext *ctx = (GpuContext *) malloc(sizeof(GpuContext));
    if (!ctx) return NULL;

    if (max_seqlen > MAX_SEQLEN) max_seqlen = MAX_SEQLEN;

    ctx->max_nraw = max_nraw;
    ctx->max_seqlen = max_seqlen;
    ctx->nraw = 0;

    cudaMalloc(&ctx->d_seqs, (size_t)max_nraw * max_seqlen);
    cudaMalloc(&ctx->d_quals, (size_t)max_nraw * max_seqlen);
    cudaMalloc(&ctx->d_kmer8, (size_t)max_nraw * N_KMER);
    cudaMalloc(&ctx->d_lengths, max_nraw * sizeof(unsigned int));
    cudaMalloc(&ctx->d_reads, max_nraw * sizeof(unsigned int));
    cudaMalloc(&ctx->d_locks, max_nraw * sizeof(int));
    cudaMalloc(&ctx->d_lambdas, max_nraw * sizeof(double));
    cudaMalloc(&ctx->d_hammings, max_nraw * sizeof(unsigned int));

    return ctx;
}

extern "C" void gpu_context_destroy(GpuContext *ctx) {
    if (!ctx) return;
    cudaFree(ctx->d_seqs);
    cudaFree(ctx->d_quals);
    cudaFree(ctx->d_kmer8);
    cudaFree(ctx->d_lengths);
    cudaFree(ctx->d_reads);
    cudaFree(ctx->d_locks);
    cudaFree(ctx->d_lambdas);
    cudaFree(ctx->d_hammings);
    free(ctx);
}

/* ---------------- Data Upload ---------------- */

extern "C" void gpu_upload_raws(GpuContext *ctx,
                                const char *all_seqs,
                                const uint8_t *all_quals,
                                const uint8_t *all_kmer8,
                                const unsigned int *lengths,
                                const unsigned int *reads,
                                unsigned int nraw) {
    ctx->nraw = nraw;
    size_t seq_bytes = (size_t)nraw * ctx->max_seqlen;
    cudaMemcpy(ctx->d_seqs, all_seqs, seq_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_quals, all_quals, seq_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_kmer8, all_kmer8, (size_t)nraw * N_KMER, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_lengths, lengths, nraw * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_reads, reads, nraw * sizeof(unsigned int), cudaMemcpyHostToDevice);
    /* Initialize locks to 0 */
    cudaMemset(ctx->d_locks, 0, nraw * sizeof(int));
}

extern "C" void gpu_upload_err_mat(GpuContext *ctx, const double *err_mat,
                                   unsigned int nrow, unsigned int ncol) {
    (void)ctx; /* err_mat goes to constant memory, not ctx */
    if (ncol > MAX_ERR_NCOL) ncol = MAX_ERR_NCOL;
    cudaMemcpyToSymbol(d_err_mat, err_mat, nrow * ncol * sizeof(double));
    cudaMemcpyToSymbol(d_err_ncol, &ncol, sizeof(unsigned int));
}

extern "C" void gpu_upload_locks(GpuContext *ctx, const int *locks, unsigned int nraw) {
    cudaMemcpy(ctx->d_locks, locks, nraw * sizeof(int), cudaMemcpyHostToDevice);
}

/* ---------------- Device Helper Functions ---------------- */

/*
 * Kmer distance: compute min-sum overlap between two kmer8 vectors.
 * Returns distance in [0,1]. Center kmer8 is in shared memory.
 */
__device__ double kmer_dist_gpu(const uint8_t *center_kmer8,
                                const uint8_t *raw_kmer8,
                                unsigned int len_center,
                                unsigned int len_raw) {
    unsigned int dotsum = 0;
    for (int k = 0; k < N_KMER; k++) {
        uint8_t a = center_kmer8[k];
        uint8_t b = raw_kmer8[k];
        dotsum += (a < b) ? a : b;
    }
    unsigned int min_len = (len_center < len_raw) ? len_center : len_raw;
    unsigned int denom = min_len - KMER_SIZE + 1;
    if (denom == 0) return 1.0;
    double dot = (double)dotsum / (double)denom;
    return 1.0 - dot;
}

/*
 * Standard banded Needleman-Wunsch alignment (ends-free).
 * Uses standard (i,j) DP matrix with band constraint.
 * Fills traceback into local p[] array, returns alignment score.
 *
 * s1 = center (reference), s2 = raw (query)
 * Sequences are int-encoded: A=1,C=2,G=3,T=4
 *
 * The DP matrix is stored in thread-local memory (physically L1-cached global).
 * Only the banded region is stored: for each row i, columns from
 * max(0, j_center-band) to min(len2, j_center+band) where j_center = i*len2/len1.
 *
 * To keep memory bounded, we use a row-major layout with 2*band+1 columns per row.
 */

/* Maximum band size supported */
#define MAX_BAND 32

/*
 * Simplified banded NW that directly extracts substitutions.
 *
 * Instead of building full alignment strings, the traceback directly
 * identifies substitution positions (ref pos, ref nt, raw nt) and
 * builds the map from ref positions to raw positions.
 *
 * Returns number of substitutions found.
 * sub_pos0[]: ref positions of substitutions (0-indexed)
 * sub_nt0[]:  ref nucleotide at each sub
 * sub_nt1[]:  raw nucleotide at each sub
 * map[]:      ref_pos -> raw_pos mapping
 * nsubs:      number of substitutions
 */
/*
 * Exact port of CPU nwalign_vectorized2() to CUDA device function.
 * Uses the same rotated anti-diagonal coordinate system, swap logic,
 * ends-free boundary recalculation, and tie-breaking precedence.
 *
 * Instead of producing alignment strings, the traceback directly
 * extracts substitution positions (fused al2subs).
 *
 * Memory: for 250bp seqs, band=16: nrow=501, ncol~19 -> ~19KB d + ~10KB p
 */

/* Max rotated matrix dimensions.
 * ncol depends on band and length difference:
 *   ncol = 2 + start_col + ((len2-len1+band) < len2 ? (len2-len1+band) : len2) / 2
 * For same-length: ncol ≈ band + 4.
 * For different lengths: ncol can be up to band + (len2-len1)/2 + 4.
 * Since len diff is typically < band for banded alignment, use 2*band + 6 as safe max. */
#define MAX_NCOL_ROT (2 * MAX_BAND + 6)
#define MAX_NROW_ROT (2 * MAX_SEQLEN + 1)

__device__ int nwalign_and_extract_subs(
    const char *s1_in, unsigned int len1_in,  /* center/ref (original input) */
    const char *s2_in, unsigned int len2_in,  /* raw/query (original input) */
    int match, int mismatch, int gap_pen,
    int band,
    uint16_t *sub_pos0, char *sub_nt0, char *sub_nt1,
    uint16_t *map)  /* len1_in entries */
{
    if (len1_in > MAX_SEQLEN || len2_in > MAX_SEQLEN) {
        for (unsigned int pp = 0; pp < len1_in; pp++) map[pp] = pp < len2_in ? pp : GAP_GLYPH;
        return 0;
    }

    /* Local copies for potential swap */
    const char *s1 = s1_in, *s2 = s2_in;
    int len1 = (int)len1_in, len2 = (int)len2_in;
    bool do_swap = false;

    /* Ensure s1 is the shorter sequence (matching CPU exactly) */
    if (len1 > len2) {
        const char *tmp = s1; s1 = s2; s2 = tmp;
        int t = len1; len1 = len2; len2 = t;
        do_swap = true;
    }
    if (band < 0) band = len2;
    int16_t end_gap_p = 0;  /* dada2 always calls with end_gap_p=0 */
    int16_t gap_p = (int16_t)gap_pen;

    /* Rotated matrix dimensions (exact CPU formula) */
    int start_col = 1 + (1 + (band < len1 ? band : len1)) / 2;
    int ncol = 2 + start_col + ((len2 - len1 + band) < len2 ? (len2 - len1 + band) : len2) / 2;
    int nrow = len1 + len2 + 1;

    if (ncol > MAX_NCOL_ROT || nrow > MAX_NROW_ROT) {
        /* Fallback for oversized — shouldn't happen with MAX_SEQLEN/MAX_BAND */
        for (unsigned int pp = 0; pp < len1_in; pp++) map[pp] = pp < len2_in ? pp : GAP_GLYPH;
        return 0;
    }

    int16_t d[MAX_NROW_ROT * MAX_NCOL_ROT];
    int8_t  p[MAX_NROW_ROT * MAX_NCOL_ROT];
    int16_t diag_buf[MAX_NCOL_ROT];

    /* Fill boundaries with -infinity */
    int16_t fill_val = INT16_MIN - (mismatch < gap_pen ? (mismatch < match ? mismatch : (match < 0 ? match : 0))
                                                       : (gap_pen < match ? gap_pen : (match < 0 ? match : 0)));
    /* Simpler: just use a very negative value */
    fill_val = -30000;
    for (int row = 0; row < nrow; row++) {
        d[row * ncol] = fill_val;
        d[row * ncol + 1] = fill_val;
        d[row * ncol + ncol - 2] = fill_val;
        d[row * ncol + ncol - 1] = fill_val;
    }

    /* Starting point */
    d[start_col] = 0;
    p[start_col] = 0;

    /* Fill "left column" (ends-free gaps in s2 direction) */
    {
        int row = 1, col = start_col - 1;
        int16_t d_free = end_gap_p;
        int limit = 1 + (band < len1 ? band : len1);
        while (row < limit) {
            d[row * ncol + col] = d_free;
            p[row * ncol + col] = 3;
            if (row % 2 == 0) col--;
            row++;
            d_free += end_gap_p;
        }
    }

    /* Fill "top row" (ends-free gaps in s1 direction) */
    {
        int row = 1, col = start_col;
        int16_t d_free = end_gap_p;
        int limit = 1 + ((band + len2 - len1) < len2 ? (band + len2 - len1) : len2);
        while (row < limit) {
            d[row * ncol + col] = d_free;
            p[row * ncol + col] = 2;
            if (row % 2 == 1) col++;
            row++;
            d_free += end_gap_p;
        }
    }

    /* Fill DP matrix — exact CPU logic */
    int col_min = start_col, col_max = start_col;
    int i_max = 0, j_min = 0;
    int even = 1;
    bool recalc_left = false, recalc_right = false;

    for (int row = 2; row <= len1 + len2; row++) {
        /* Compute diagonal buffer */
        for (int col = col_min, ii = i_max, jj = j_min; col < 1 + col_max; col++, ii--, jj++) {
            diag_buf[col] = d[(row - 2) * ncol + col] + (s1[ii] == s2[jj] ? (int16_t)match : (int16_t)mismatch);
        }

        /* DP fill with precedence matching CPU */
        int16_t *ptr_left = &d[(row - 1) * ncol + col_min - even];
        int16_t *ptr_diag = &diag_buf[col_min];
        int16_t *ptr_up   = &d[(row - 1) * ncol + col_min + 1 - even];
        int n_cells = col_max - col_min + 1;

        if (do_swap) {
            /* dploop_vec_swap: left >= up >= diag */
            for (int k = 0; k < n_cells; k++) {
                int16_t left = ptr_left[k] + gap_p;
                int16_t diag = ptr_diag[k];
                int16_t up   = ptr_up[k] + gap_p;
                int16_t entry = left >= up ? left : up;
                int8_t pentry = left >= up ? 2 : 3;
                pentry = entry >= diag ? pentry : 1;
                entry  = entry >= diag ? entry  : diag;
                d[row * ncol + col_min + k] = entry;
                p[row * ncol + col_min + k] = pentry;
            }
        } else {
            /* dploop_vec: up >= left >= diag */
            for (int k = 0; k < n_cells; k++) {
                int16_t left = ptr_left[k] + gap_p;
                int16_t diag = ptr_diag[k];
                int16_t up   = ptr_up[k] + gap_p;
                int16_t entry = up >= left ? up : left;
                int8_t pentry = up >= left ? 3 : 2;
                pentry = entry >= diag ? pentry : 1;
                entry  = entry >= diag ? entry  : diag;
                d[row * ncol + col_min + k] = entry;
                p[row * ncol + col_min + k] = pentry;
            }
        }

        /* Band boundary offsets (exact CPU logic) */
        if (row == (band < len1 ? band : len1)) {
            col_min--; i_max++; j_min--;
        }
        if (row == ((band + len2 - len1) < len2 ? (band + len2 - len1) : len2)) {
            col_max++;
        }

        /* Ends-free recalculation (critical for matching CPU) */
        if (end_gap_p > gap_p) {
            /* Left column */
            if (recalc_left) {
                int16_t d_free = ptr_left[0] + end_gap_p;
                if (d_free > d[row * ncol + col_min]) {
                    d[row * ncol + col_min] = d_free;
                    p[row * ncol + col_min] = 2;
                } else if (!do_swap && d_free == d[row * ncol + col_min] && p[row * ncol + col_min] == 1) {
                    p[row * ncol + col_min] = 2;
                } else if (do_swap && d_free == d[row * ncol + col_min] && p[row * ncol + col_min] != 2) {
                    p[row * ncol + col_min] = 2;
                }
            }
            if (i_max == len1 - 1) recalc_left = true;
            /* Right column */
            if (recalc_right) {
                int16_t d_free = ptr_up[col_max - col_min] + end_gap_p;
                if (d_free > d[row * ncol + col_max]) {
                    d[row * ncol + col_max] = d_free;
                    p[row * ncol + col_max] = 3;
                } else if (!do_swap && d_free == d[row * ncol + col_max] && p[row * ncol + col_max] != 3) {
                    p[row * ncol + col_max] = 3;
                } else if (do_swap && d_free == d[row * ncol + col_max] && p[row * ncol + col_max] == 1) {
                    p[row * ncol + col_max] = 3;
                }
            }
            if ((row + 1) / 2 + col_max - start_col == len2) recalc_right = true;
        }

        /* Update indices (exact CPU logic) */
        if (row < band && row < len1) {
            if (even) col_min--;
            i_max++;
        } else if (i_max < len1 - 1) {
            if (band % 2 == 0) {
                if (even) j_min++;
                else i_max++;
            } else {
                if (even) { col_min--; i_max++; }
                else { col_min++; j_min++; }
            }
        } else {
            if (!even) col_min++;
            j_min++;
        }

        if (row < ((band + len2 - len1) < len2 ? (band + len2 - len1) : len2)) {
            if (!even) col_max++;
        } else if ((row + 1) / 2 + col_max - start_col < len2) {
            if ((band + len2 - len1) % 2 == 0) {
                if (even) col_max--;
                else col_max++;
            }
        } else {
            if (even) col_max--;
        }

        even = 1 - even;
    }

    /* Traceback (exact CPU logic) — collect moves in reverse */
    int8_t moves[2 * MAX_SEQLEN + 2];
    int nmoves = 0;
    int ti = len1, tj = len2;

    while (ti > 0 || tj > 0) {
        int8_t ptr = p[(ti + tj) * ncol + (2 * start_col + tj - ti) / 2];
        switch (ptr) {
            case 1: moves[nmoves++] = 1; ti--; tj--; break;
            case 2: moves[nmoves++] = 2; tj--; break;
            case 3: moves[nmoves++] = 3; ti--; break;
            default: moves[nmoves++] = 1; ti--; tj--; break; /* emergency */
        }
        if (nmoves >= 2 * MAX_SEQLEN) break;
    }

    /* If swapped, swap the move meanings:
     * In the swapped DP, s1 and s2 are swapped, so:
     *   move=2 (gap in s1) -> gap in the original s2 (the shorter)
     *   move=3 (gap in s2) -> gap in the original s1 (the longer)
     * The CPU swaps the alignment strings (al0 <-> al1) after traceback.
     * Equivalently, we swap move 2 and 3 meanings in the extraction. */

    /* Extract substitutions from reversed moves.
     * After swap: al[0] = s1_in (center), al[1] = s2_in (raw)
     * Without swap: al[0] = s1 = s1_in, al[1] = s2 = s2_in
     * With swap: al[0] = s2 (was s1 in DP) = s1_in, al[1] = s1 (was s2 in DP) = s2_in
     *
     * In the CPU: traceback produces al0 from s1 (DP), al1 from s2 (DP).
     * Then if swap: al0 and al1 are swapped.
     * So: al[0] always corresponds to s1_in (center), al[1] to s2_in (raw).
     *
     * Move meanings in DP space:
     *   1 = diagonal: both s1(DP) and s2(DP) advance
     *   2 = left: gap in s1(DP), s2(DP) advances -> al0='-', al1=s2[j]
     *   3 = up: s1(DP) advances, gap in s2(DP) -> al0=s1[i], al1='-'
     *
     * After swap (al0<->al1):
     *   1 = diagonal: both advance (same)
     *   2 = was gap in s1(DP)=s2_in -> now: al0=s2[j]=s2_in nucleotide, al1='-' -> s1_in has gap
     *   3 = was gap in s2(DP)=s1_in -> now: al0='-', al1=s1[i]=s1_in nucleotide -> s2_in has gap
     * Wait, this is getting confusing. Let me think differently.
     *
     * In al2subs, is_nt0 means al[0] has a nucleotide (not gap).
     * al[0] = center (s1_in). So:
     *   Without swap: move 1 -> is_nt0=true,is_nt1=true
     *                 move 2 -> is_nt0=false (gap in al0=s1), is_nt1=true (al1=s2 has nt)
     *                 move 3 -> is_nt0=true (al0=s1 has nt), is_nt1=false (gap in al1=s2)
     *   With swap:    al0 and al1 are swapped, so
     *                 move 1 -> is_nt0=true, is_nt1=true (same)
     *                 move 2 -> gap in DP-s1=s2_in -> original: al[0]=s1_in gets gap? No...
     *
     * Actually let me just track it through carefully.
     * CPU traceback builds al0 from s1(DP) and al1 from s2(DP).
     * Case 1: ti--, tj-- -> al0[k] = s1[--ti], al1[k] = s2[--tj]
     * Case 2: tj-- -> al0[k] = '-', al1[k] = s2[--tj]
     * Case 3: ti-- -> al0[k] = s1[--ti], al1[k] = '-'
     * Then if swap: ptr_char = al0; al0 = al1; al1 = ptr_char;
     * So after swap:
     *   al[0] (center=s1_in) was al1 (from s2(DP)=s1_in) ✓
     *   al[1] (raw=s2_in) was al0 (from s1(DP)=s2_in) ✓
     *
     * For sub extraction, we need:
     *   is_nt0 = al[0][k] is a nucleotide (center=s1_in has nt at this position)
     *   is_nt1 = al[1][k] is a nucleotide (raw=s2_in has nt at this position)
     *
     * Without swap:
     *   move 1: is_nt0=true(s1_in nt), is_nt1=true(s2_in nt)
     *   move 2: is_nt0=false(gap), is_nt1=true(s2_in=s2(DP) nt)
     *   move 3: is_nt0=true(s1_in=s1(DP) nt), is_nt1=false(gap)
     *
     * With swap (al0<->al1 means center comes from s2(DP), raw from s1(DP)):
     *   move 1: is_nt0=true, is_nt1=true (both nt)
     *   move 2: al0 was '-' (gap in s1(DP)=s2_in=raw) -> after swap: al[1]='-' -> is_nt1=false
     *           al1 was s2(DP)=s1_in=center nt -> after swap: al[0]=nt -> is_nt0=true
     *   move 3: al0 was s1(DP)=s2_in=raw nt -> after swap: al[1]=nt -> is_nt1=true
     *           al1 was '-' (gap in s2(DP)=s1_in=center) -> after swap: al[0]='-' -> is_nt0=false
     *
     * Summary (both cases in terms of is_nt0=center has nt, is_nt1=raw has nt):
     *   Without swap: move1 -> (T,T), move2 -> (F,T), move3 -> (T,F)
     *   With swap:    move1 -> (T,T), move2 -> (T,F), move3 -> (F,T)
     *   i.e., with swap, move2 and move3 meanings swap!
     */
    int nsubs = 0;
    int i0 = -1, i1 = -1;  /* positions in s1_in (center) and s2_in (raw) */

    for (int m = nmoves - 1; m >= 0; m--) {
        int8_t mv = moves[m];
        bool is_nt_center, is_nt_raw;

        if (!do_swap) {
            is_nt_center = (mv == 1 || mv == 3);
            is_nt_raw    = (mv == 1 || mv == 2);
        } else {
            is_nt_center = (mv == 1 || mv == 2);
            is_nt_raw    = (mv == 1 || mv == 3);
        }

        if (is_nt_center) i0++;
        if (is_nt_raw) i1++;

        if (is_nt_center) {
            map[i0] = is_nt_raw ? (uint16_t)i1 : (uint16_t)GAP_GLYPH;
        }
        if (is_nt_center && is_nt_raw && i0 < (int)len1_in && i1 < (int)len2_in) {
            char c0 = s1_in[i0], c1 = s2_in[i1];
            if (c0 != c1 && c0 != 5 && c1 != 5) {
                sub_pos0[nsubs] = (uint16_t)i0;
                sub_nt0[nsubs] = c0;
                sub_nt1[nsubs] = c1;
                nsubs++;
            }
        }
    }
    return nsubs;
}

/*
 * Gapless alignment: simple position-by-position comparison.
 * Sequences must be same length.
 */
__device__ int gapless_extract_subs(
    const char *s1, unsigned int len1,
    const char *s2, unsigned int len2,
    uint16_t *sub_pos0,
    char *sub_nt0,
    char *sub_nt1,
    uint16_t *map)
{
    unsigned int minlen = (len1 < len2) ? len1 : len2;
    int nsubs = 0;

    for (unsigned int pos = 0; pos < minlen; pos++) {
        map[pos] = (uint16_t)pos;
        char c0 = s1[pos];
        char c1 = s2[pos];
        if (c0 != c1 && c0 != 5 && c1 != 5) {
            sub_pos0[nsubs] = (uint16_t)pos;
            sub_nt0[nsubs] = c0;
            sub_nt1[nsubs] = c1;
            nsubs++;
        }
    }
    /* Any extra positions in s1 map to GAP_GLYPH */
    for (unsigned int pos = minlen; pos < len1; pos++) {
        map[pos] = GAP_GLYPH;
    }
    return nsubs;
}

/*
 * Compute lambda from substitution data and error matrix.
 * Mirrors compute_lambda_ts() from pval.cpp.
 *
 * lambda = product over all positions in raw of errMat[transition][quality]
 * where transition = nt0*4 + nt1 (0-15 index)
 */
__device__ double compute_lambda_gpu(
    const char *raw_seq,
    const uint8_t *raw_qual,
    unsigned int raw_len,
    int nsubs,
    const uint16_t *sub_pos0,
    const char *sub_nt0,
    const char *sub_nt1,
    const uint16_t *map,
    unsigned int len0,         /* center/ref length */
    int use_quals,
    unsigned int ncol)
{
    if (raw_len > MAX_SEQLEN) return 0.0;

    /* Build transition vector: default is self-transition */
    unsigned int tvec[MAX_SEQLEN];
    unsigned int qind[MAX_SEQLEN];

    for (unsigned int pos1 = 0; pos1 < raw_len; pos1++) {
        int nti1 = ((int)raw_seq[pos1]) - 1;
        if (nti1 < 0 || nti1 > 3) return 0.0;  /* non-ACGT */
        tvec[pos1] = nti1 * 4 + nti1;  /* self-transition */
        qind[pos1] = use_quals ? (unsigned int)raw_qual[pos1] : 0;
        if (qind[pos1] >= ncol) qind[pos1] = ncol - 1;
    }

    /* Override at substitution positions */
    for (int s = 0; s < nsubs; s++) {
        uint16_t pos0 = sub_pos0[s];
        if (pos0 >= len0) continue;
        uint16_t pos1 = map[pos0];
        if (pos1 == GAP_GLYPH || pos1 >= raw_len) continue;

        int nti0 = ((int)sub_nt0[s]) - 1;
        int nti1 = ((int)sub_nt1[s]) - 1;
        if (nti0 < 0 || nti0 > 3 || nti1 < 0 || nti1 > 3) continue;
        tvec[pos1] = nti0 * 4 + nti1;
    }

    /* Compute lambda = product of error rates */
    double lambda = 1.0;
    for (unsigned int pos1 = 0; pos1 < raw_len; pos1++) {
        lambda *= d_err_mat[tvec[pos1] * ncol + qind[pos1]];
    }

    return lambda;
}

/* ---- GPU kmer screen kernel ----
 * Only computes kmer distances and greedy checks.
 * Outputs: d_passed[tid] = 1 if pair passed screen, 0 if shrouded/skipped.
 * Alignment and lambda are computed on CPU for pairs that pass.
 * This ensures byte-identical alignment with CPU nwalign_vectorized2. */

__global__ void kmer_screen_kernel(
    const uint8_t *d_kmer8,
    const unsigned int *d_lengths,
    const unsigned int *d_reads,
    const int *d_locks,
    unsigned int center_index,
    unsigned int nraw,
    double kdist_cutoff,
    int use_kmers,
    int greedy, unsigned int center_reads,
    int *d_passed)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nraw) return;

    __shared__ uint8_t center_kmer8[N_KMER];
    for (int k = threadIdx.x; k < N_KMER; k += blockDim.x) {
        center_kmer8[k] = d_kmer8[center_index * N_KMER + k];
    }
    __syncthreads();

    /* Greedy checks */
    if (greedy && d_reads[tid] > center_reads) { d_passed[tid] = 0; return; }
    if (greedy && d_locks[tid]) { d_passed[tid] = 0; return; }

    /* Kmer distance screening */
    if (use_kmers) {
        const uint8_t *raw_kmer8 = &d_kmer8[tid * N_KMER];
        double kdist = kmer_dist_gpu(center_kmer8, raw_kmer8,
                                     d_lengths[center_index], d_lengths[tid]);
        if (kdist > kdist_cutoff) { d_passed[tid] = 0; return; }
    }

    d_passed[tid] = 1;  /* Passed screen, needs alignment on CPU */
}

/* ---------------- Host API ---------------- */

extern "C" void gpu_compare(GpuContext *ctx,
                             unsigned int center_index,
                             unsigned int nraw,
                             int match, int mismatch, int gap_pen,
                             int band_size,
                             double kdist_cutoff,
                             int use_kmers, int use_quals, int gapless,
                             int greedy, unsigned int center_reads,
                             unsigned int ncol_err,
                             double *lambdas,
                             unsigned int *hammings) {
    if (!ctx || nraw == 0) return;

    /* Allocate device array for screen results */
    int *d_passed;
    cudaMalloc(&d_passed, nraw * sizeof(int));

    int grid = (nraw + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kmer_screen_kernel<<<grid, BLOCK_SIZE>>>(
        ctx->d_kmer8, ctx->d_lengths, ctx->d_reads, ctx->d_locks,
        center_index, nraw,
        kdist_cutoff, use_kmers, greedy, center_reads,
        d_passed);

    /* Copy screen results to host */
    int *passed = (int *)malloc(nraw * sizeof(int));
    cudaMemcpy(passed, d_passed, nraw * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_passed);

    /* Initialize all outputs to shrouded */
    for (unsigned int i = 0; i < nraw; i++) {
        lambdas[i] = 0.0;
        hammings[i] = (unsigned int)(-1);
    }

    /* For pairs that passed kmer screen, the caller (b_compare_gpu)
     * will handle alignment and lambda on CPU. We store the screen
     * results in the lambdas array: -1.0 means "passed, needs alignment".
     * 0.0 means "shrouded/skipped". */
    for (unsigned int i = 0; i < nraw; i++) {
        if (passed[i]) {
            lambdas[i] = -1.0;  /* sentinel: needs CPU alignment */
        }
    }
    free(passed);
}

extern "C" int gpu_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return 0;
    return (count > 0) ? 1 : 0;
}
