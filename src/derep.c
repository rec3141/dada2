/*
 * derep.c - Fast FASTQ dereplication in C.
 * Reads gzipped FASTQ, deduplicates sequences, averages quality scores.
 * Returns per-read map (read_idx -> unique_idx) for paired-end merging.
 * Called from Python via ctypes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

/* Simple hash map for sequence deduplication */
#define HASH_SIZE (1 << 20)  /* 1M buckets */
#define MAX_SEQ_LEN 1024
#define MAP_INIT_CAP 65536

typedef struct Entry {
    char *seq;
    int count;
    double *qual_sum;
    int seq_len;
    int insert_id;      /* first-seen order (before abundance sort) */
    struct Entry *next;
} Entry;

typedef struct {
    int n_uniques;
    int n_reads;
    int max_seq_len;
    /* Sorted by abundance (descending) */
    char **seqs;        /* n_uniques null-terminated strings */
    int *abundances;    /* n_uniques */
    double *quals;      /* n_uniques * max_seq_len, row-major, NaN-padded */
    int *map;           /* n_reads: read_idx -> sorted unique_idx (0-indexed) */
} DerepResult;

static int cmp_entry_desc(const void *a, const void *b) {
    int ca = (*(Entry **)a)->count, cb = (*(Entry **)b)->count;
    return (cb > ca) - (cb < ca);
}

/* FNV-1a hash — good avalanche properties for short DNA strings */
static unsigned int hash_seq(const char *s, int len) {
    unsigned int h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h & (HASH_SIZE - 1);
}

DerepResult* derep_fastq_c(const char *filepath) {
    gzFile gz = gzopen(filepath, "rb");
    if (!gz) return NULL;

    Entry **table = (Entry **)calloc(HASH_SIZE, sizeof(Entry *));

    char line[MAX_SEQ_LEN * 2];
    int n_reads = 0, n_uniques = 0, max_len = 0;

    /* Growable array for per-read map (read_idx -> insert_id) */
    int map_cap = MAP_INIT_CAP;
    int *raw_map = (int *)malloc(map_cap * sizeof(int));

    /* Parse FASTQ: 4 lines per record */
    while (gzgets(gz, line, sizeof(line))) {
        /* Line 1: header (skip) */
        /* Line 2: sequence */
        if (!gzgets(gz, line, sizeof(line))) break;
        int slen = strlen(line);
        while (slen > 0 && (line[slen-1] == '\n' || line[slen-1] == '\r')) slen--;
        line[slen] = '\0';
        /* Uppercase */
        for (int i = 0; i < slen; i++)
            if (line[i] >= 'a' && line[i] <= 'z') line[i] -= 32;

        if (slen >= MAX_SEQ_LEN) {
            fprintf(stderr, "derep_fastq_c: sequence length %d exceeds MAX_SEQ_LEN %d, truncating\n",
                    slen, MAX_SEQ_LEN);
            slen = MAX_SEQ_LEN - 1;
            line[slen] = '\0';
        }
        if (slen > max_len) max_len = slen;

        /* Line 3: + (skip) */
        char plus[MAX_SEQ_LEN * 2];
        if (!gzgets(gz, plus, sizeof(plus))) break;

        /* Line 4: quality */
        char qline[MAX_SEQ_LEN * 2];
        if (!gzgets(gz, qline, sizeof(qline))) break;
        int qlen = strlen(qline);
        while (qlen > 0 && (qline[qlen-1] == '\n' || qline[qlen-1] == '\r')) qlen--;

        /* Hash lookup */
        unsigned int h = hash_seq(line, slen);
        Entry *e = table[h];
        while (e) {
            if (e->seq_len == slen && memcmp(e->seq, line, slen) == 0) break;
            e = e->next;
        }

        if (!e) {
            /* New unique */
            e = (Entry *)malloc(sizeof(Entry));
            e->seq = (char *)malloc(slen + 1);
            memcpy(e->seq, line, slen + 1);
            e->seq_len = slen;
            e->count = 0;
            e->qual_sum = (double *)calloc(slen, sizeof(double));
            e->insert_id = n_uniques;
            e->next = table[h];
            table[h] = e;
            n_uniques++;
        }

        e->count++;
        int mlen = slen < qlen ? slen : qlen;
        for (int i = 0; i < mlen; i++)
            e->qual_sum[i] += (double)((unsigned char)qline[i] - 33);

        /* Record map: this read -> insert_id (will remap to sorted index later) */
        if (n_reads >= map_cap) {
            map_cap *= 2;
            raw_map = (int *)realloc(raw_map, map_cap * sizeof(int));
        }
        raw_map[n_reads] = e->insert_id;

        n_reads++;
    }
    gzclose(gz);

    /* Collect all entries into array */
    Entry **all = (Entry **)malloc(n_uniques * sizeof(Entry *));
    int idx = 0;
    for (int i = 0; i < HASH_SIZE; i++) {
        Entry *e = table[i];
        while (e) {
            all[idx++] = e;
            e = e->next;
        }
    }

    /* Sort by abundance descending */
    qsort(all, n_uniques, sizeof(Entry *), cmp_entry_desc);

    /* Build insert_id -> sorted_index remap table */
    int *remap = (int *)malloc(n_uniques * sizeof(int));
    for (int i = 0; i < n_uniques; i++)
        remap[all[i]->insert_id] = i;

    /* Remap raw_map from insert_id to sorted index */
    int *sorted_map = (int *)malloc(n_reads * sizeof(int));
    for (int i = 0; i < n_reads; i++)
        sorted_map[i] = remap[raw_map[i]];
    free(raw_map);
    free(remap);

    /* Build result */
    DerepResult *res = (DerepResult *)calloc(1, sizeof(DerepResult));
    res->n_uniques = n_uniques;
    res->n_reads = n_reads;
    res->max_seq_len = max_len;
    res->seqs = (char **)malloc(n_uniques * sizeof(char *));
    res->abundances = (int *)malloc(n_uniques * sizeof(int));
    res->quals = (double *)malloc((size_t)n_uniques * max_len * sizeof(double));
    res->map = sorted_map;

    /* Fill quals with NaN */
    for (size_t i = 0; i < (size_t)n_uniques * max_len; i++)
        res->quals[i] = 0.0 / 0.0;  /* NaN */

    for (int i = 0; i < n_uniques; i++) {
        res->seqs[i] = all[i]->seq;  /* transfer ownership */
        res->abundances[i] = all[i]->count;
        /* Average quality */
        for (int j = 0; j < all[i]->seq_len; j++)
            res->quals[(size_t)i * max_len + j] = all[i]->qual_sum[j] / all[i]->count;
        free(all[i]->qual_sum);
        free(all[i]);
    }
    free(all);

    free(table);
    return res;
}

void derep_result_free(DerepResult *res) {
    if (!res) return;
    for (int i = 0; i < res->n_uniques; i++) free(res->seqs[i]);
    free(res->seqs);
    free(res->abundances);
    free(res->quals);
    free(res->map);
    free(res);
}
