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
#define BLOCK_SIZE 32        /* Threads per block (reduced for large local memory in NW) */
#define GAP_GLYPH 9999

/* Error matrix in constant memory: 16 transitions x max 256 quality cols.
 * Must accommodate all possible ncol values (quality scores 0-255). */
#define MAX_ERR_NCOL 256
__constant__ double d_err_mat[16 * MAX_ERR_NCOL];
__constant__ unsigned int d_err_ncol;

/* ---------------- GPU Context ---------------- */

struct GpuContext {
    /* Device pointers */
    char *d_seqs;              /* nraw * max_seqlen */
    uint8_t *d_quals;          /* nraw * max_seqlen */
    uint8_t *d_kmer8;          /* nraw * N_KMER */
    uint16_t *d_kord;          /* nraw * max_seqlen */
    unsigned int *d_lengths;   /* nraw */
    unsigned int *d_reads;     /* nraw */
    int *d_locks;              /* nraw */

    /* Output buffers */
    double *d_lambdas;         /* nraw */
    unsigned int *d_hammings;  /* nraw */
    int *d_needs_nw;           /* nraw: 1 if pair needs banded NW */

    /* Dimensions */
    unsigned int max_nraw;
    unsigned int max_seqlen;
    unsigned int nraw;         /* current number of raws */
};

/* ---------------- Lifecycle ---------------- */

extern "C" GpuContext* gpu_context_create(unsigned int max_nraw, unsigned int max_seqlen) {
    GpuContext *ctx = (GpuContext *) malloc(sizeof(GpuContext));
    if (!ctx) return NULL;

    ctx->max_nraw = max_nraw;
    ctx->max_seqlen = max_seqlen;
    ctx->nraw = 0;

    cudaMalloc(&ctx->d_seqs, (size_t)max_nraw * max_seqlen);
    cudaMalloc(&ctx->d_quals, (size_t)max_nraw * max_seqlen);
    cudaMalloc(&ctx->d_kmer8, (size_t)max_nraw * N_KMER);
    cudaMalloc(&ctx->d_kord, (size_t)max_nraw * max_seqlen * sizeof(uint16_t));
    cudaMalloc(&ctx->d_lengths, max_nraw * sizeof(unsigned int));
    cudaMalloc(&ctx->d_reads, max_nraw * sizeof(unsigned int));
    cudaMalloc(&ctx->d_locks, max_nraw * sizeof(int));
    cudaMalloc(&ctx->d_lambdas, max_nraw * sizeof(double));
    cudaMalloc(&ctx->d_hammings, max_nraw * sizeof(unsigned int));
    cudaMalloc(&ctx->d_needs_nw, max_nraw * sizeof(int));

    return ctx;
}

extern "C" void gpu_context_destroy(GpuContext *ctx) {
    if (!ctx) return;
    cudaFree(ctx->d_seqs);
    cudaFree(ctx->d_quals);
    cudaFree(ctx->d_kmer8);
    cudaFree(ctx->d_lengths);
    cudaFree(ctx->d_reads);
    cudaFree(ctx->d_kord);
    cudaFree(ctx->d_locks);
    cudaFree(ctx->d_lambdas);
    cudaFree(ctx->d_hammings);
    cudaFree(ctx->d_needs_nw);
    free(ctx);
}

/* ---------------- Data Upload ---------------- */

extern "C" void gpu_upload_raws(GpuContext *ctx,
                                const char *all_seqs,
                                const uint8_t *all_quals,
                                const uint8_t *all_kmer8,
                                const uint16_t *all_kord,    /* nraw * ctx->max_seqlen, or NULL */
                                const unsigned int *lengths,
                                const unsigned int *reads,
                                unsigned int nraw) {
    ctx->nraw = nraw;
    size_t seq_bytes = (size_t)nraw * ctx->max_seqlen;
    cudaMemcpy(ctx->d_seqs, all_seqs, seq_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_quals, all_quals, seq_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_kmer8, all_kmer8, (size_t)nraw * N_KMER, cudaMemcpyHostToDevice);
    if (all_kord)
        cudaMemcpy(ctx->d_kord, all_kord, (size_t)nraw * ctx->max_seqlen * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_lengths, lengths, nraw * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_reads, reads, nraw * sizeof(unsigned int), cudaMemcpyHostToDevice);
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
 * Kord distance: compare kmer order vectors position-by-position.
 * Returns fraction of mismatched positions (0=identical order, 1=completely different).
 * Matches CPU kord_dist_SSEi logic.
 */
__device__ double kord_dist_gpu(const uint16_t *kord1, unsigned int len1,
                                const uint16_t *kord2, unsigned int len2) {
    unsigned int minlen = len1 < len2 ? len1 : len2;
    if (minlen <= KMER_SIZE) return 1.0;
    unsigned int n_pos = minlen - KMER_SIZE + 1;
    unsigned int matches = 0;
    for (unsigned int i = 0; i < n_pos; i++) {
        if (kord1[i] == kord2[i]) matches++;
    }
    return 1.0 - (double)matches / (double)n_pos;
}

/* ---- Fused GPU kernel: kmer screen + kord check + gapless lambda ----
 * Pass 1 of the 2-pass approach. For ALL pairs:
 *   - Compute kmer distance (screen out distant pairs)
 *   - Compute kord distance to detect if banded NW is needed
 *   - Compute gapless lambda (fast, no warp divergence)
 *   - Flag pairs where kord != kmer dist (need CPU banded NW in pass 2)
 *
 * Output:
 *   d_lambdas:  gapless lambda (will be overwritten by CPU for flagged pairs)
 *   d_hammings: gapless hamming (will be overwritten by CPU for flagged pairs)
 *   d_needs_nw: 1 if pair needs banded NW on CPU, 0 otherwise
 */
__global__ void compare_kernel_2pass(
    const char *d_seqs,
    const uint8_t *d_quals,
    const uint8_t *d_kmer8,
    const uint16_t *d_kord,
    const unsigned int *d_lengths,
    const unsigned int *d_reads,
    const int *d_locks,
    unsigned int center_index,
    unsigned int nraw,
    unsigned int max_seqlen,
    double kdist_cutoff,
    int use_kmers, int use_quals, int gapless,
    int greedy, unsigned int center_reads,
    unsigned int ncol_err,
    double *d_lambdas,
    unsigned int *d_hammings,
    int *d_needs_nw)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nraw) return;

    __shared__ uint8_t center_kmer8[N_KMER];
    for (int k = threadIdx.x; k < N_KMER; k += blockDim.x)
        center_kmer8[k] = d_kmer8[center_index * N_KMER + k];
    __syncthreads();

    d_needs_nw[tid] = 0;

    /* Greedy checks */
    if (greedy && d_reads[tid] > center_reads) { d_lambdas[tid] = 0.0; d_hammings[tid] = (unsigned)-1; return; }
    if (greedy && d_locks[tid]) { d_lambdas[tid] = 0.0; d_hammings[tid] = (unsigned)-1; return; }

    unsigned int len_c = d_lengths[center_index], len_r = d_lengths[tid];

    /* Kmer distance screen */
    double kdist = 0.0;
    if (use_kmers) {
        kdist = kmer_dist_gpu(center_kmer8, &d_kmer8[tid * N_KMER], len_c, len_r);
        if (kdist > kdist_cutoff) { d_lambdas[tid] = 0.0; d_hammings[tid] = (unsigned)-1; return; }
    }

    /* Kord distance — detect if banded NW alignment is needed */
    if (use_kmers && gapless) {
        double kodist = kord_dist_gpu(&d_kord[center_index * max_seqlen], len_c,
                                      &d_kord[tid * max_seqlen], len_r);
        /* CPU logic: if (gapless && kodist == kdist) → use gapless, else → use banded NW */
        if (kodist != kdist) {
            d_needs_nw[tid] = 1;  /* Flag for CPU pass 2 */
        }
    }

    /* Gapless alignment + lambda (computed for ALL passing pairs, even NW-flagged) */
    const char *cs = &d_seqs[center_index * max_seqlen];
    const char *rs = &d_seqs[tid * max_seqlen];
    const uint8_t *rq = &d_quals[tid * max_seqlen];

    unsigned int minlen = len_c < len_r ? len_c : len_r;
    double lambda = 1.0;
    unsigned int nsubs = 0;

    for (unsigned int pos = 0; pos < minlen; pos++) {
        int nti0 = ((int)cs[pos]) - 1;
        int nti1 = ((int)rs[pos]) - 1;
        if (nti0 < 0 || nti0 > 3 || nti1 < 0 || nti1 > 3) { lambda = 0.0; break; }
        unsigned int qind = use_quals ? (unsigned int)rq[pos] : 0;
        if (qind >= ncol_err) qind = ncol_err - 1;
        unsigned int tvec = (nti0 == nti1) ? (nti1 * 4 + nti1) : (nti0 * 4 + nti1);
        lambda *= d_err_mat[tvec * ncol_err + qind];
        if (nti0 != nti1) nsubs++;
    }

    if (lambda < 0.0 || lambda > 1.0) lambda = 0.0;
    d_lambdas[tid] = lambda;
    d_hammings[tid] = nsubs;
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

    int grid = (nraw + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* 2-pass kernel: GPU computes gapless lambda for all pairs,
     * flags those needing banded NW (kord != kmer distance).
     * CPU (in b_compare_gpu) handles the flagged ~5% with banded NW. */
    compare_kernel_2pass<<<grid, BLOCK_SIZE>>>(
        ctx->d_seqs, ctx->d_quals, ctx->d_kmer8, ctx->d_kord,
        ctx->d_lengths, ctx->d_reads, ctx->d_locks,
        center_index, nraw, ctx->max_seqlen,
        kdist_cutoff, use_kmers, use_quals, gapless,
        greedy, center_reads, ncol_err,
        ctx->d_lambdas, ctx->d_hammings, ctx->d_needs_nw);

    cudaMemcpy(lambdas, ctx->d_lambdas, nraw * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hammings, ctx->d_hammings, nraw * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    /* Copy needs_nw flags — encode in lambdas as -1.0 sentinel for CPU pass 2 */
    int *needs_nw = (int *)malloc(nraw * sizeof(int));
    cudaMemcpy(needs_nw, ctx->d_needs_nw, nraw * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (unsigned int i = 0; i < nraw; i++) {
        if (needs_nw[i]) {
            lambdas[i] = -1.0;  /* sentinel: CPU must recompute with banded NW */
        }
    }
    free(needs_nw);
}

extern "C" int gpu_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return 0;
    return (count > 0) ? 1 : 0;
}
