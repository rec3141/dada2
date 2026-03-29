/*
 * derep.c - Fast FASTQ dereplication in C.
 * Reads gzipped FASTQ, deduplicates sequences, averages quality scores.
 * Quality scores are accumulated in lexical sequence order to match
 * R's derepFastq (which uses srsort), ensuring identical float
 * accumulation and thus identical rounded uint8 quality values.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#define HASH_SIZE (1 << 20)
#define MAX_SEQ_LEN 1024

typedef struct Entry {
    char *seq;
    int count;
    long *qual_sum;  /* integer accumulation to match R's rowsum on integer matrix */
    int seq_len;
    int insert_id;
    struct Entry *next;
} Entry;

typedef struct {
    int seq_offset;   /* offset into pool */
    int qual_offset;  /* offset into pool */
    int seq_len;
    int orig_index;   /* file-order index for map */
} ReadRec;

typedef struct {
    int n_uniques;
    int n_reads;
    int max_seq_len;
    char **seqs;
    int *abundances;
    double *quals;
    int *map;
} DerepResult;

static int cmp_entry_desc(const void *a, const void *b) {
    int ca = (*(Entry **)a)->count, cb = (*(Entry **)b)->count;
    return (cb > ca) - (cb < ca);
}

static unsigned int hash_seq(const char *s, int len) {
    unsigned int h = 2166136261u;
    for (int i = 0; i < len; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h & (HASH_SIZE - 1);
}

/* Global pool pointer for sort comparator */
static char *g_pool;
static int cmp_read_by_seq(const void *a, const void *b) {
    const ReadRec *ra = (const ReadRec *)a;
    const ReadRec *rb = (const ReadRec *)b;
    int minlen = ra->seq_len < rb->seq_len ? ra->seq_len : rb->seq_len;
    int c = memcmp(g_pool + ra->seq_offset, g_pool + rb->seq_offset, minlen);
    if (c != 0) return c;
    return ra->seq_len - rb->seq_len;
}

DerepResult* derep_fastq_c(const char *filepath) {
    gzFile gz = gzopen(filepath, "rb");
    if (!gz) return NULL;

    /* Phase 1: Read all reads into memory (offsets into pool) */
    int read_cap = 65536, pool_cap = 65536 * 300, pool_used = 0, n_reads = 0, max_len = 0;
    ReadRec *reads = (ReadRec *)malloc(read_cap * sizeof(ReadRec));
    char *pool = (char *)malloc(pool_cap);
    char buf[MAX_SEQ_LEN * 2];

    while (gzgets(gz, buf, sizeof(buf))) {  /* header */
        /* sequence */
        if (!gzgets(gz, buf, sizeof(buf))) break;
        int slen = strlen(buf);
        while (slen > 0 && (buf[slen-1] == '\n' || buf[slen-1] == '\r')) slen--;
        buf[slen] = '\0';
        for (int i = 0; i < slen; i++)
            if (buf[i] >= 'a' && buf[i] <= 'z') buf[i] -= 32;
        if (slen >= MAX_SEQ_LEN) { slen = MAX_SEQ_LEN - 1; buf[slen] = '\0'; }
        if (slen > max_len) max_len = slen;

        /* + line */
        char plus[MAX_SEQ_LEN * 2];
        if (!gzgets(gz, plus, sizeof(plus))) break;

        /* quality */
        char qbuf[MAX_SEQ_LEN * 2];
        if (!gzgets(gz, qbuf, sizeof(qbuf))) break;
        int qlen = strlen(qbuf);
        while (qlen > 0 && (qbuf[qlen-1] == '\n' || qbuf[qlen-1] == '\r')) qlen--;

        /* Grow pool */
        int need = slen + 1 + qlen + 1;
        while (pool_used + need > pool_cap) { pool_cap *= 2; pool = (char *)realloc(pool, pool_cap); }

        int seq_off = pool_used;
        memcpy(pool + pool_used, buf, slen + 1); pool_used += slen + 1;
        int qual_off = pool_used;
        memcpy(pool + pool_used, qbuf, qlen + 1); pool_used += qlen + 1;

        /* Grow reads */
        if (n_reads >= read_cap) { read_cap *= 2; reads = (ReadRec *)realloc(reads, read_cap * sizeof(ReadRec)); }
        reads[n_reads].seq_offset = seq_off;
        reads[n_reads].qual_offset = qual_off;
        reads[n_reads].seq_len = slen;
        reads[n_reads].orig_index = n_reads;
        n_reads++;
    }
    gzclose(gz);

    /* Phase 2: Sort reads by sequence (lexical) to match R's srsort */
    g_pool = pool;
    qsort(reads, n_reads, sizeof(ReadRec), cmp_read_by_seq);

    /* Phase 3: Dedup + accumulate qualities in sorted order */
    Entry **table = (Entry **)calloc(HASH_SIZE, sizeof(Entry *));
    int n_uniques = 0;
    int *orig_to_insert = (int *)malloc(n_reads * sizeof(int));  /* orig_index -> insert_id */

    for (int i = 0; i < n_reads; i++) {
        char *seq = pool + reads[i].seq_offset;
        int slen = reads[i].seq_len;
        char *qline = pool + reads[i].qual_offset;
        int qlen = strlen(qline);

        unsigned int h = hash_seq(seq, slen);
        Entry *e = table[h];
        while (e) { if (e->seq_len == slen && memcmp(e->seq, seq, slen) == 0) break; e = e->next; }

        if (!e) {
            e = (Entry *)malloc(sizeof(Entry));
            e->seq = (char *)malloc(slen + 1);
            memcpy(e->seq, seq, slen + 1);
            e->seq_len = slen;
            e->count = 0;
            e->qual_sum = (long *)calloc(slen, sizeof(long));
            e->insert_id = n_uniques++;
            e->next = table[h];
            table[h] = e;
        }

        e->count++;
        int mlen = slen < qlen ? slen : qlen;
        for (int j = 0; j < mlen; j++)
            e->qual_sum[j] += (long)((unsigned char)qline[j] - 33);

        orig_to_insert[reads[i].orig_index] = e->insert_id;
    }

    /* Phase 4: Sort by abundance, build remap, build result */
    Entry **all = (Entry **)malloc(n_uniques * sizeof(Entry *));
    int idx = 0;
    for (int i = 0; i < HASH_SIZE; i++) {
        Entry *e = table[i];
        while (e) { all[idx++] = e; e = e->next; }
    }
    qsort(all, n_uniques, sizeof(Entry *), cmp_entry_desc);

    int *remap = (int *)malloc(n_uniques * sizeof(int));
    for (int i = 0; i < n_uniques; i++) remap[all[i]->insert_id] = i;

    /* Map: original file order -> abundance-sorted unique index */
    int *sorted_map = (int *)malloc(n_reads * sizeof(int));
    for (int i = 0; i < n_reads; i++)
        sorted_map[i] = remap[orig_to_insert[i]];
    free(orig_to_insert);
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

    for (size_t i = 0; i < (size_t)n_uniques * max_len; i++)
        res->quals[i] = 0.0 / 0.0;

    for (int i = 0; i < n_uniques; i++) {
        res->seqs[i] = all[i]->seq;
        res->abundances[i] = all[i]->count;
        for (int j = 0; j < all[i]->seq_len; j++)
            res->quals[(size_t)i * max_len + j] = (double)all[i]->qual_sum[j] / (double)all[i]->count;
        free(all[i]->qual_sum);
        free(all[i]);
    }
    free(all);
    free(table);
    free(reads);
    free(pool);

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
