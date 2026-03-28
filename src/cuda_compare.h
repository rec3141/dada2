#ifndef CUDA_COMPARE_H
#define CUDA_COMPARE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle for persistent GPU resources */
typedef struct GpuContext GpuContext;

/* Lifecycle */
GpuContext* gpu_context_create(unsigned int max_nraw, unsigned int max_seqlen);
void gpu_context_destroy(GpuContext *ctx);

/* Upload raw sequence data to GPU (once per dada_uniques call) */
void gpu_upload_raws(GpuContext *ctx,
                     const char *all_seqs,        /* nraw * max_seqlen, padded */
                     const uint8_t *all_quals,     /* nraw * max_seqlen, padded */
                     const uint8_t *all_kmer8,     /* nraw * 1024 */
                     const unsigned int *lengths,  /* nraw */
                     const unsigned int *reads,    /* nraw */
                     unsigned int nraw);

/* Upload error matrix (16 x ncol) to GPU constant memory */
void gpu_upload_err_mat(GpuContext *ctx, const double *err_mat,
                        unsigned int nrow, unsigned int ncol);

/* Upload lock states (changes during greedy mode) */
void gpu_upload_locks(GpuContext *ctx, const int *locks, unsigned int nraw);

/*
 * Execute fused comparison kernel: kmer distance + NW align + lambda
 * for all nraw sequences against center_index.
 *
 * Output arrays (host-allocated, size nraw):
 *   lambdas[i]  = error probability for raw i (0 if shrouded/skipped)
 *   hammings[i] = number of substitutions (UINT_MAX if shrouded/skipped)
 */
void gpu_compare(GpuContext *ctx,
                 unsigned int center_index,
                 unsigned int nraw,
                 int match, int mismatch, int gap_pen,
                 int band_size,
                 double kdist_cutoff,
                 int use_kmers, int use_quals, int gapless,
                 int greedy, unsigned int center_reads,
                 unsigned int ncol_err,
                 double *lambdas,           /* out, host */
                 unsigned int *hammings);   /* out, host */

/* Check if a CUDA-capable GPU is available (returns 1 if yes) */
int gpu_available(void);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPARE_H */
