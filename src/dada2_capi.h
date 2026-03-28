#ifndef DADA2_CAPI_H
#define DADA2_CAPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int      nclust;
    int      nraw;
    int      maxlen;

    /* Per-cluster arrays (nclust entries) */
    char   **cluster_seqs;      /* ACGT sequences, null-terminated */
    int     *cluster_abunds;
    int     *cluster_n0;
    int     *cluster_n1;
    int     *cluster_nunq;
    double  *cluster_pval;

    /* Transition matrix: 16 x ncol_trans, row-major */
    int      ncol_trans;
    int     *trans;

    /* Map: nraw entries, 0-indexed cluster assignment (-1 for uncorrected) */
    int     *map;

    /* Per-raw p-values */
    double  *pval;
} DadaResult;

DadaResult* dada2_run(
    const char **seqs,
    const int  *abundances,
    const int  *priors,
    int         nraw,
    const double *err_mat,      /* 16 x ncol_err, row-major */
    int         ncol_err,
    const double *quals,        /* nraw x maxlen, row-major (avg Q per pos), or NULL */
    int         maxlen,
    /* alignment params */
    int match, int mismatch, int gap_pen,
    int use_kmers, double kdist_cutoff,
    int band_size,
    /* statistical params */
    double omegaA, double omegaP, double omegaC,
    int detect_singletons,
    int max_clust,
    double min_fold, int min_hamming, int min_abund,
    int use_quals,
    int vectorized_alignment,
    int homo_gap_pen,
    int multithread,
    int verbose,
    int SSE, int gapless, int greedy
);

void dada2_result_free(DadaResult *res);
int dada2_gpu_available(void);

/* Taxonomy assignment result */
typedef struct {
    int      nseq;      /* number of query sequences */
    int      nlevel;    /* number of taxonomy levels */
    int     *rval;      /* best genus index per query (nseq), 1-indexed, 0=NA */
    int     *rboot;     /* bootstrap counts (nseq x nlevel), row-major */
} TaxResult;

/* Assign taxonomy using naive Bayesian kmer classifier.
 *
 * seqs:         query sequences (nseq)
 * nseq:         number of query sequences
 * refs:         reference sequences (nref)
 * nref:         number of references
 * ref_to_genus: 0-indexed genus ID per reference (nref)
 * genusmat:     genus-to-level assignment matrix (ngenus x nlevel), row-major
 * ngenus:       number of unique genera
 * nlevel:       number of taxonomy levels
 * verbose:      print progress
 *
 * Returns a TaxResult* that must be freed with dada2_tax_result_free().
 */
TaxResult* dada2_assign_taxonomy(
    const char **seqs,
    int nseq,
    const char **refs,
    int nref,
    const int *ref_to_genus,
    const int *genusmat,
    int ngenus,
    int nlevel,
    int verbose
);

void dada2_tax_result_free(TaxResult *res);

#ifdef __cplusplus
}
#endif

#endif /* DADA2_CAPI_H */
