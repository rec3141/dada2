/*
 * dada2_capi.cpp - Pure C API wrapper for DADA2 core algorithm.
 * Replaces Rmain.cpp for standalone (non-R) builds.
 * Compiled only with -DNO_RCPP.
 */
#ifndef NO_RCPP
/* This file is only for standalone builds. When compiled as R package,
 * Rmain.cpp provides the entry point instead. */
#else

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "dada.h"
#include "dada2_capi.h"

#ifdef HAVE_CUDA
#include "cuda_compare.h"
#endif

/* Forward declaration of the standalone run_dada */
static B *run_dada_c(Raw **raws, int nraw, double *err_mat, int ncol_err,
                     int match, int mismatch, int gap_pen, int homo_gap_pen,
                     bool use_kmers, double kdist_cutoff, int band_size,
                     double omegaA, double omegaP, bool detect_singletons,
                     int max_clust, double min_fold, int min_hamming, int min_abund,
                     bool use_quals, bool vectorized_alignment,
                     bool multithread, bool verbose, int SSE, bool gapless, bool greedy
#ifdef HAVE_CUDA
                     , GpuContext *gpu_ctx, unsigned int gpu_max_seqlen
#endif
                     );

/* Build transition matrix from clustering (mirrors b_make_transition_by_quality_matrix) */
static void fill_trans_matrix(B *b, Sub **subs, bool has_quals, int ncol_err,
                              int **out_trans, int *out_ncol) {
    int ncol = ncol_err;
    int *mat = (int *)calloc(16 * ncol, sizeof(int));
    if (!mat) { *out_trans = NULL; *out_ncol = 0; return; }

    for (unsigned int i = 0; i < b->nclust; i++) {
        for (unsigned int r = 0; r < b->bi[i]->nraw; r++) {
            Raw *raw = b->bi[i]->raw[r];
            if (!raw->correct) continue;  /* Skip uncorrected sequences (matches R) */
            Sub *sub = subs[raw->index];
            if (!sub) continue;

            /* For each position in the center (reference) sequence */
            Raw *center = b->bi[i]->center;
            for (unsigned int pos0 = 0; pos0 < center->length; pos0++) {
                uint16_t pos1 = sub->map[pos0];
                if (pos1 == GAP_GLYPH || pos1 >= raw->length) continue;

                int nti0 = ((int)b->bi[i]->center->seq[pos0]) - 1;
                int nti1 = ((int)raw->seq[pos1]) - 1;
                if (nti0 < 0 || nti0 > 3 || nti1 < 0 || nti1 > 3) continue;

                int qind = has_quals ? (int)raw->qual[pos1] : 0;
                if (qind >= ncol) qind = ncol - 1;

                int row = nti0 * 4 + nti1;
                mat[row * ncol + qind] += raw->reads;
            }
        }
    }
    *out_trans = mat;
    *out_ncol = ncol;
}

extern "C" DadaResult* dada2_run(
    const char **seqs,
    const int  *abundances,
    const int  *priors,
    int         nraw,
    const double *err_mat,
    int         ncol_err,
    const double *quals,
    int         maxlen,
    int match, int mismatch, int gap_pen,
    int use_kmers, double kdist_cutoff,
    int band_size,
    double omegaA, double omegaP, double omegaC,
    int detect_singletons,
    int max_clust,
    double min_fold, int min_hamming, int min_abund,
    int use_quals,
    int vectorized_alignment,
    int homo_gap_pen,
    int multithread,
    int verbose,
    int SSE, int gapless, int greedy)
{
    unsigned int i, r, index, pos;
    Raw *raw;

    if (nraw == 0) { fprintf(stderr, "dada2_run: zero input sequences.\n"); return NULL; }
    if (ncol_err < 1) { fprintf(stderr, "dada2_run: invalid error matrix.\n"); return NULL; }

    bool has_quals = (quals != NULL);

    /* Determine actual maxlen from sequences */
    unsigned int actual_maxlen = 0;
    for (index = 0; index < (unsigned)nraw; index++) {
        unsigned int slen = strlen(seqs[index]);
        if (slen > actual_maxlen) actual_maxlen = slen;
    }
    if (maxlen == 0) maxlen = actual_maxlen;

    /* Construct Raw structures */
    char seq_buf[SEQLEN];
    double qual_buf[SEQLEN];
    Raw **raws = (Raw **)malloc(nraw * sizeof(Raw *));
    if (!raws) { fprintf(stderr, "Memory allocation failed.\n"); return NULL; }

    for (index = 0; index < (unsigned)nraw; index++) {
        strncpy(seq_buf, seqs[index], SEQLEN - 1);
        seq_buf[SEQLEN - 1] = '\0';
        nt2int(seq_buf, seq_buf);
        if (has_quals) {
            for (pos = 0; pos < strlen(seqs[index]); pos++) {
                qual_buf[pos] = quals[(size_t)index * maxlen + pos];
            }
            raws[index] = raw_new(seq_buf, qual_buf, abundances[index], priors ? priors[index] : 0);
        } else {
            raws[index] = raw_new(seq_buf, NULL, abundances[index], priors ? priors[index] : 0);
        }
        raws[index]->index = index;
    }

    /* Build kmer structures */
    uint8_t *k8 = NULL;
    uint16_t *k16 = NULL;
    uint16_t *kord = NULL;
    if (use_kmers) {
        size_t n_kmer = 1 << (2 * KMER_SIZE);
        k8 = (uint8_t *)malloc(nraw * n_kmer * sizeof(uint8_t));
        k16 = (uint16_t *)malloc(nraw * n_kmer * sizeof(uint16_t));
        kord = (uint16_t *)malloc(nraw * actual_maxlen * sizeof(uint16_t));
        if (!k8 || !k16 || !kord) { fprintf(stderr, "Memory allocation failed.\n"); return NULL; }
        for (index = 0; index < (unsigned)nraw; index++) {
            raw = raws[index];
            raw->kmer8 = &k8[index * n_kmer];
            assign_kmer8(raw->kmer8, raw->seq, KMER_SIZE);
            raw->kmer = &k16[index * n_kmer];
            assign_kmer(raw->kmer, raw->seq, KMER_SIZE);
            raw->kord = &kord[index * actual_maxlen];
            assign_kmer_order(raw->kord, raw->seq, KMER_SIZE);
        }
    } else {
        for (index = 0; index < (unsigned)nraw; index++) {
            raws[index]->kmer8 = NULL;
            raws[index]->kmer = NULL;
            raws[index]->kord = NULL;
        }
    }

    /* Convert error matrix to mutable copy for internal use */
    double *err_mat_c = (double *)malloc(16 * ncol_err * sizeof(double));
    if (!err_mat_c) { fprintf(stderr, "Memory allocation failed.\n"); return NULL; }
    memcpy(err_mat_c, err_mat, 16 * ncol_err * sizeof(double));

#ifdef HAVE_CUDA
    /* GPU setup */
    GpuContext *gpu_ctx = NULL;
    bool use_gpu = dada2_gpu_available() && (homo_gap_pen == gap_pen);
    if (use_gpu) {
        if (verbose) printf("GPU: Initializing CUDA context for %d sequences (maxlen=%d)...\n", nraw, actual_maxlen);
        gpu_ctx = gpu_context_create(nraw, actual_maxlen);
        if (gpu_ctx) {
            char *all_seqs = (char *)calloc((size_t)nraw * actual_maxlen, 1);
            uint8_t *all_quals = (uint8_t *)calloc((size_t)nraw * actual_maxlen, 1);
            unsigned int *lengths = (unsigned int *)malloc(nraw * sizeof(unsigned int));
            unsigned int *reads_arr = (unsigned int *)malloc(nraw * sizeof(unsigned int));
            if (!all_seqs || !all_quals || !lengths || !reads_arr) {
                fprintf(stderr, "GPU memory allocation failed.\n");
                gpu_context_destroy(gpu_ctx); gpu_ctx = NULL;
            } else {
                for (index = 0; index < (unsigned)nraw; index++) {
                    memcpy(&all_seqs[(size_t)index * actual_maxlen], raws[index]->seq, raws[index]->length);
                    if (raws[index]->qual)
                        memcpy(&all_quals[(size_t)index * actual_maxlen], raws[index]->qual, raws[index]->length);
                    lengths[index] = raws[index]->length;
                    reads_arr[index] = raws[index]->reads;
                }
                gpu_upload_raws(gpu_ctx, all_seqs, all_quals,
                                use_kmers ? k8 : NULL,
                                use_kmers ? kord : NULL,
                                lengths, reads_arr, nraw);
                free(all_seqs); free(all_quals); free(lengths); free(reads_arr);
                if (verbose) printf("GPU: Data uploaded successfully.\n");
            }
        }
    }
#endif

    /* Run DADA algorithm */
    B *bb = run_dada_c(raws, nraw, err_mat_c, ncol_err,
                       match, mismatch, gap_pen, homo_gap_pen,
                       use_kmers, kdist_cutoff, band_size,
                       omegaA, omegaP, detect_singletons,
                       max_clust, min_fold, min_hamming, min_abund,
                       use_quals, vectorized_alignment,
                       multithread, verbose, SSE, gapless, greedy
#ifdef HAVE_CUDA
                       , gpu_ctx, actual_maxlen
#endif
                       );

#ifdef HAVE_CUDA
    if (gpu_ctx) { gpu_context_destroy(gpu_ctx); gpu_ctx = NULL; }
#endif

    /* Build final alignments for output */
    Sub **subs = (Sub **)malloc(bb->nraw * sizeof(Sub *));
    if (!subs) { fprintf(stderr, "Memory allocation failed.\n"); return NULL; }
    for (i = 0; i < bb->nclust; i++) {
        for (r = 0; r < bb->bi[i]->nraw; r++) {
            raw = bb->bi[i]->raw[r];
            subs[raw->index] = sub_new(bb->bi[i]->center, raw, match, mismatch, gap_pen, homo_gap_pen,
                                       false, 1.0, band_size, vectorized_alignment, SSE, gapless);
        }
    }

    /* Assign final p-values */
    for (i = 0; i < bb->nclust; i++) {
        for (r = 0; r < bb->bi[i]->nraw; r++) {
            raw = bb->bi[i]->raw[r];
            if (bb->bi[i]->center == raw) {
                raw->p = 1.0;
            } else {
                raw->p = calc_pA(raw->reads, raw->comp.lambda * bb->bi[i]->reads, true);
                if (raw->p < omegaC) { raw->correct = false; }
            }
        }
    }

    /* Construct DadaResult */
    DadaResult *res = (DadaResult *)calloc(1, sizeof(DadaResult));
    if (!res) { fprintf(stderr, "Memory allocation failed.\n"); return NULL; }

    res->nclust = bb->nclust;
    res->nraw = bb->nraw;
    res->maxlen = actual_maxlen;

    /* Cluster sequences and abundances */
    res->cluster_seqs = (char **)malloc(bb->nclust * sizeof(char *));
    res->cluster_abunds = (int *)malloc(bb->nclust * sizeof(int));
    res->cluster_n0 = (int *)calloc(bb->nclust, sizeof(int));
    res->cluster_n1 = (int *)calloc(bb->nclust, sizeof(int));
    res->cluster_nunq = (int *)malloc(bb->nclust * sizeof(int));
    res->cluster_pval = (double *)malloc(bb->nclust * sizeof(double));

    char oseq[SEQLEN];
    for (i = 0; i < bb->nclust; i++) {
        /* Find most abundant raw as representative */
        Raw *max_raw = bb->bi[i]->center;
        int2nt(oseq, max_raw->seq);
        res->cluster_seqs[i] = strdup(oseq);
        res->cluster_abunds[i] = bb->bi[i]->reads;
        res->cluster_nunq[i] = bb->bi[i]->nraw;
        res->cluster_pval[i] = bb->bi[i]->birth_pval;

        /* Count n0 (abundance of center) and n1 (2nd most abundant) */
        unsigned int max1 = 0, max2 = 0;
        for (r = 0; r < bb->bi[i]->nraw; r++) {
            unsigned int rd = bb->bi[i]->raw[r]->reads;
            if (rd > max1) { max2 = max1; max1 = rd; }
            else if (rd > max2) { max2 = rd; }
        }
        res->cluster_n0[i] = max1;
        res->cluster_n1[i] = max2;
    }

    /* Transition matrix */
    fill_trans_matrix(bb, subs, has_quals, ncol_err, &res->trans, &res->ncol_trans);

    /* Map: raw -> cluster (0-indexed, -1 for uncorrected) */
    res->map = (int *)malloc(bb->nraw * sizeof(int));
    res->pval = (double *)malloc(bb->nraw * sizeof(double));
    for (i = 0; i < bb->nclust; i++) {
        for (r = 0; r < bb->bi[i]->nraw; r++) {
            raw = bb->bi[i]->raw[r];
            res->map[raw->index] = raw->correct ? (int)i : -1;
            res->pval[raw->index] = raw->p;
        }
    }

    /* Cleanup */
    for (index = 0; index < bb->nraw; index++) sub_free(subs[index]);
    free(subs);
    b_free(bb);
    for (index = 0; index < (unsigned)nraw; index++) raw_free(raws[index]);
    free(raws);
    if (use_kmers) { free(k8); free(k16); free(kord); }
    free(err_mat_c);

    return res;
}

extern "C" void dada2_result_free(DadaResult *res) {
    if (!res) return;
    if (res->cluster_seqs) {
        for (int i = 0; i < res->nclust; i++) free(res->cluster_seqs[i]);
        free(res->cluster_seqs);
    }
    free(res->cluster_abunds);
    free(res->cluster_n0);
    free(res->cluster_n1);
    free(res->cluster_nunq);
    free(res->cluster_pval);
    free(res->trans);
    free(res->map);
    free(res->pval);
    free(res);
}

extern "C" int dada2_gpu_available(void) {
#ifdef HAVE_CUDA
    return gpu_available();
#else
    return 0;
#endif
}

/* Standalone run_dada (mirrors Rmain.cpp::run_dada but takes flat arrays) */
static B *run_dada_c(Raw **raws, int nraw, double *err_mat, int ncol_err,
                     int match, int mismatch, int gap_pen, int homo_gap_pen,
                     bool use_kmers, double kdist_cutoff, int band_size,
                     double omegaA, double omegaP, bool detect_singletons,
                     int max_clust, double min_fold, int min_hamming, int min_abund,
                     bool use_quals, bool vectorized_alignment,
                     bool multithread, bool verbose, int SSE, bool gapless, bool greedy
#ifdef HAVE_CUDA
                     , GpuContext *gpu_ctx, unsigned int gpu_max_seqlen
#endif
                     ) {
    int newi = 0, nshuffle = 0;
    bool shuffled = false;

    B *bb = b_new(raws, nraw, omegaA, omegaP, use_quals);

    /* Initial comparison - all raws vs cluster 0, no kmer screen */
#ifdef HAVE_CUDA
    if (gpu_ctx) {
        gpu_upload_err_mat(gpu_ctx, err_mat, 16, ncol_err);
        b_compare_gpu(bb, 0, err_mat, ncol_err, gpu_ctx, gpu_max_seqlen,
                      match, mismatch, gap_pen, use_kmers, 1.0, band_size, gapless, greedy, verbose);
    } else
#endif
    {
        b_compare_omp(bb, 0, err_mat, ncol_err, match, mismatch, gap_pen, homo_gap_pen,
                      use_kmers, 1.0, band_size, vectorized_alignment, SSE, gapless, greedy, verbose);
    }

    b_p_update(bb, greedy, detect_singletons);

    if (max_clust < 1) max_clust = bb->nraw;

    while ((bb->nclust < (unsigned)max_clust) && (newi = b_bud(bb, min_fold, min_hamming, min_abund, verbose))) {
        if (verbose) printf("\nNew Cluster C%d:", newi);

#ifdef HAVE_CUDA
        if (gpu_ctx) {
            b_compare_gpu(bb, newi, err_mat, ncol_err, gpu_ctx, gpu_max_seqlen,
                          match, mismatch, gap_pen, use_kmers, kdist_cutoff, band_size, gapless, greedy, verbose);
        } else
#endif
        {
            b_compare_omp(bb, newi, err_mat, ncol_err, match, mismatch, gap_pen, homo_gap_pen,
                          use_kmers, kdist_cutoff, band_size, vectorized_alignment, SSE, gapless, greedy, verbose);
        }

        nshuffle = 0;
        do {
            shuffled = b_shuffle2(bb);
            if (verbose) printf("S");
        } while (shuffled && ++nshuffle < MAX_SHUFFLE);
        if (verbose && nshuffle >= MAX_SHUFFLE) printf("Warning: Reached maximum (%d) shuffles.\n", MAX_SHUFFLE);

        b_p_update(bb, greedy, detect_singletons);
    }

    if (verbose) printf("\nALIGN: %d aligns, %d shrouded (%d raw).\n", bb->nalign, bb->nshroud, bb->nraw);

    return bb;
}
#endif /* NO_RCPP */
