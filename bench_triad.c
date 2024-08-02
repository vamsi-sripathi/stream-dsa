#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "immintrin.h"
#include "float.h"
#include "time.h"
#include "omp.h"

#ifdef USE_DSA
#include "dml/dml.h"
#endif

#define NTRIALS             (100)
#define ALIGN               (4*1024)
#define MAX_THREADS         (56)
#define QUEUE_DEPTH         (2)
#define BLK_SIZE_IN_BYTES   (1*1024*1024)
#define SCALAR_CONSTANT     (3.0)

#undef DEMOTE_BUFFERS

#if !defined(USE_CPU) && !defined(USE_DSA)
#error USE_CPU of USE_DSA should be defined
#endif

void demote_buffers(long long *p_n, double *p_src1, double *p_src2, double *p_dst)
{
  long long n = *p_n;
  for (long long i=0; i<n; i+=8) {
#if 1
    _mm_clflushopt(p_src1); 
    _mm_clflushopt(p_src2); 
    _mm_clflushopt(p_dst); 
#else
    _mm_cldemote(p_src1);
    _mm_cldemote(p_src2);
    _mm_cldemote(p_dst);
#endif
    p_src1 += 8;
    p_src2 += 8;
    p_dst  += 8;
  }
}
      
void seq_triad(long long *p_n, double *p_a, double *p_b, double *p_c)
{
  long long n = *p_n;

  for (long long i=0; i<n; i++) {
    p_c[i] = p_a[i] + SCALAR_CONSTANT * p_b[i];
  }
}

void cpu_omp_triad(long long *p_n, double *p_a, double *p_b, double *p_c)
{
  long long n = *p_n;

#pragma omp parallel for
  for (long long i=0; i<n; i++) {
    p_c[i] = p_a[i] + SCALAR_CONSTANT * p_b[i];
  }
}

int check_triad_results(long long *p_n, double *p_exp, double *p_obs)
{
  for (long long i=0; i<*p_n; i++) {
    if (p_exp[i] != p_obs[i]) {
      printf ("\tERROR: Results differ: Expected[%ld] = %lf, Observed[%ld] = %lf\n",
              i, p_exp[i], i, p_obs[i]);
      return 1;
    }
  }
  return 0;
}


#if defined(USE_DSA)
dml_job_t* init_dml()
{
  dml_job_t *p_dml_job = NULL;
  dml_status_t status;
  uint32_t size;

  status  = dml_get_job_size(DML_PATH_HW, &size);
  if (DML_STATUS_OK != status) {
    exit(1);
  }

  p_dml_job = (dml_job_t *) malloc(size);
  status  = dml_init_job(DML_PATH_HW, p_dml_job);
  if (DML_STATUS_OK != status) {
    exit(1);
  }

  return p_dml_job;
}

void sync_copy(long long *p_n, double *p_src, double *p_dst, dml_job_t *p_dml_job)
{
  dml_status_t status;

  p_dml_job->operation             = DML_OP_MEM_MOVE;
  p_dml_job->flags                 = DML_FLAG_COPY_ONLY|DML_FLAG_PREFETCH_CACHE;
  p_dml_job->source_first_ptr      = (void *)p_src;
  p_dml_job->destination_first_ptr = (void *)p_dst;
  p_dml_job->source_length         = (*p_n)*sizeof(double);
  p_dml_job->destination_length    = (*p_n)*sizeof(double);

  status = dml_execute_job(p_dml_job);

  if (status) {
    printf ("\tdml_execute_job status failed, status = %u\n", status);
  }
}

void async_copy(long long *p_n, double *p_src, double *p_dst, dml_job_t *p_dml_job)
{
  dml_status_t status;

  p_dml_job->operation             = DML_OP_MEM_MOVE;
  p_dml_job->flags                 = DML_FLAG_COPY_ONLY|DML_FLAG_PREFETCH_CACHE;
  p_dml_job->source_first_ptr      = (void *)p_src;
  p_dml_job->destination_first_ptr = (void *)p_dst;
  p_dml_job->source_length         = (*p_n)*sizeof(double);
  p_dml_job->destination_length    = (*p_n)*sizeof(double);

  status = dml_submit_job(p_dml_job);

  if (status) {
    printf ("\tdml_submit_job status failed, status = %u\n", status); fflush(0);
  }
}

void dsa_omp_triad(long long *p_n, double *p_src1, double *p_src2, double *p_dst,
                   double *p_tmp1, double *p_tmp2,
                   dml_job_t **p_dml_jobs_t1, dml_job_t **p_dml_jobs_t2)
{
  long long n = *p_n;

#pragma omp parallel
  {
    int nthrs = omp_get_num_threads();
    int ithr  = omp_get_thread_num();

    long long chunk = n/nthrs;
    long long tail  = n - (chunk*nthrs);
    long long start = ithr * chunk;
    if ((tail) && (ithr == nthrs-1)) {
      chunk += tail;
    }

    double *p_t_src1 = p_src1 + start;
    double *p_t_src2 = p_src2 + start;
    double *p_t_dst  = p_dst  + start;

    double *p_t_tmp1 = p_tmp1 + (ithr * ((BLK_SIZE_IN_BYTES*QUEUE_DEPTH)/sizeof(double)));
    double *p_t_tmp2 = p_tmp2 + (ithr * ((BLK_SIZE_IN_BYTES*QUEUE_DEPTH)/sizeof(double)));

    dml_job_t **p_t_dml_jobs_t1 = p_dml_jobs_t1 + (ithr*QUEUE_DEPTH);
    dml_job_t **p_t_dml_jobs_t2 = p_dml_jobs_t2 + (ithr*QUEUE_DEPTH);

    long long blk_elems = BLK_SIZE_IN_BYTES/sizeof(double);
    long long num_blks  = chunk/blk_elems;
    tail = chunk - (blk_elems*num_blks);

    for (int i=0; i<QUEUE_DEPTH; i++) {
      async_copy(&blk_elems, p_t_src1+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp1+(i*blk_elems), p_t_dml_jobs_t1[i]);
      async_copy(&blk_elems, p_t_src2+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp2+(i*blk_elems), p_t_dml_jobs_t2[i]);
    }

    long long tmp_elems = QUEUE_DEPTH * blk_elems;
    seq_triad(&tmp_elems, p_t_src1, p_t_src2, p_t_dst);

    int i;
    for (i=QUEUE_DEPTH; i<(num_blks-QUEUE_DEPTH); i++) {
      dml_wait_job(p_t_dml_jobs_t1[i%QUEUE_DEPTH]);
      dml_wait_job(p_t_dml_jobs_t2[i%QUEUE_DEPTH]);

      seq_triad(&blk_elems, p_t_tmp1+((i%QUEUE_DEPTH)*blk_elems), p_t_tmp2+((i%QUEUE_DEPTH)*blk_elems), p_t_dst+(i*blk_elems));

      async_copy(&blk_elems, p_t_src1+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp1+((i%QUEUE_DEPTH)*blk_elems), p_t_dml_jobs_t1[i%QUEUE_DEPTH]);
      async_copy(&blk_elems, p_t_src2+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp2+((i%QUEUE_DEPTH)*blk_elems), p_t_dml_jobs_t2[i%QUEUE_DEPTH]);
    }

    for (int j=i; j<(i+QUEUE_DEPTH); j++) {
      dml_wait_job(p_t_dml_jobs_t1[j%QUEUE_DEPTH]);
      dml_wait_job(p_t_dml_jobs_t2[j%QUEUE_DEPTH]);
      seq_triad(&blk_elems, p_t_tmp1+((j%QUEUE_DEPTH)*blk_elems), p_t_tmp2+((j%QUEUE_DEPTH)*blk_elems), p_t_dst+(j*blk_elems));
    }

    if (tail) {
      seq_triad(&tail, p_t_src1+(num_blks*blk_elems),  p_t_src2+(num_blks*blk_elems),  p_t_dst+(num_blks*blk_elems));
    }
  }
}

void dsa_omp_triad_v2(long long *p_n, double *p_src1, double *p_src2, double *p_dst,
                      double *p_tmp1, double *p_tmp2,
                      dml_job_t **p_dml_jobs_t1, dml_job_t **p_dml_jobs_t2)
{
  long long n = *p_n;

#pragma omp parallel
  {
    int nthrs = omp_get_num_threads();
    int ithr  = omp_get_thread_num();

    long long chunk = n/nthrs;
    long long tail  = n - (chunk*nthrs);
    long long start = ithr * chunk;
    if ((tail) && (ithr == nthrs-1)) {
      chunk += tail;
    }

    double *p_t_src1 = p_src1 + start;
    double *p_t_src2 = p_src2 + start;
    double *p_t_dst  = p_dst  + start;

    double *p_t_tmp1 = p_tmp1 + (ithr * ((BLK_SIZE_IN_BYTES*QUEUE_DEPTH)/sizeof(double)));

    dml_job_t **p_t_dml_jobs_t1 = p_dml_jobs_t1 + (ithr*QUEUE_DEPTH);

    long long blk_elems = BLK_SIZE_IN_BYTES/sizeof(double);
    long long num_blks  = chunk/blk_elems;
    tail = chunk - (blk_elems*num_blks);

    for (int i=0; i<QUEUE_DEPTH; i++) {
      async_copy(&blk_elems, p_t_src1+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp1+(i*blk_elems), p_t_dml_jobs_t1[i]);
    }

    long long tmp_elems = QUEUE_DEPTH * blk_elems;
    seq_triad(&tmp_elems, p_t_src1, p_t_src2, p_t_dst);

    int i;
    for (i=QUEUE_DEPTH; i<(num_blks-QUEUE_DEPTH); i++) {
      dml_wait_job(p_t_dml_jobs_t1[i%QUEUE_DEPTH]);

      seq_triad(&blk_elems, p_t_tmp1+((i%QUEUE_DEPTH)*blk_elems), p_t_src2+(i*blk_elems), p_t_dst+(i*blk_elems));

      async_copy(&blk_elems, p_t_src1+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp1+((i%QUEUE_DEPTH)*blk_elems), p_t_dml_jobs_t1[i%QUEUE_DEPTH]);
    }

    for (int j=i; j<(i+QUEUE_DEPTH); j++) {
      dml_wait_job(p_t_dml_jobs_t1[j%QUEUE_DEPTH]);
      seq_triad(&blk_elems, p_t_tmp1+((j%QUEUE_DEPTH)*blk_elems), p_t_src2+(j*blk_elems), p_t_dst+(j*blk_elems));
    }

    if (tail) {
      seq_triad(&tail, p_t_src1+(num_blks*blk_elems),  p_t_src2+(num_blks*blk_elems),  p_t_dst+(num_blks*blk_elems));
    }
  }
}

#endif

int main (int argc, char **argv)
{
  double *p_src1 = NULL, *p_src2 = NULL, *p_dst = NULL, *p_dst_ref = NULL, *p_t_elapsed = NULL;
  double t_start, size_mB, size_gB;
  long long num_bytes, num_elems;

  if (argc != 2) {
    printf("\nUSAGE: %s <size> in bytes\n", argv[0]);
    exit(1);
  }
  num_bytes = atol(argv[1]);

  if ((num_bytes%sizeof(double))) {
    printf ("ERROR: input size should be multiple of 8 (sizeof(double))\n");
    printf ("Exiting..");
    exit (1);
  }

  num_elems = num_bytes/sizeof(double);

  p_src1      = (double *)_mm_malloc(num_bytes, ALIGN);
  p_src2      = (double *)_mm_malloc(num_bytes, ALIGN);
  p_dst       = (double *)_mm_malloc(num_bytes, ALIGN);
  p_dst_ref   = (double *)_mm_malloc(num_bytes, ALIGN);
  p_t_elapsed = (double *)_mm_malloc(sizeof(double)*NTRIALS, ALIGN);

  srand48(42);

#pragma omp parallel for
  for (long long i=0; i<num_elems; i++) {
    p_src1[i] = drand48();
    p_src2[i] = drand48();
    p_dst[i]  = 0.;
  }

#if defined (USE_DSA)
  double *p_tmp1   = NULL;
  double *p_tmp2   = NULL;
  dml_job_t** p_dml_jobs_t1 = NULL;
  dml_job_t** p_dml_jobs_t2 = NULL;

  p_dml_jobs_t1  = (dml_job_t **)_mm_malloc(sizeof(dml_job_t *)*MAX_THREADS*QUEUE_DEPTH, ALIGN);
  p_dml_jobs_t2  = (dml_job_t **)_mm_malloc(sizeof(dml_job_t *)*MAX_THREADS*QUEUE_DEPTH, ALIGN);
  p_tmp1         = (double *)_mm_malloc(MAX_THREADS*BLK_SIZE_IN_BYTES*QUEUE_DEPTH, ALIGN);
  p_tmp2         = (double *)_mm_malloc(MAX_THREADS*BLK_SIZE_IN_BYTES*QUEUE_DEPTH, ALIGN);

//#pragma omp parallel for
  for (int i=0; i<MAX_THREADS*QUEUE_DEPTH; i++) {
    p_dml_jobs_t1[i] = init_dml();
    p_dml_jobs_t2[i] = init_dml();
  }

#pragma omp parallel for
  for (long long i=0; i<(MAX_THREADS*BLK_SIZE_IN_BYTES*QUEUE_DEPTH)/sizeof(double); i++) {
    p_tmp1[i] = 5.;
    p_tmp2[i] = 5.;
  }
#endif

  seq_triad(&num_elems, p_src1, p_src2, p_dst_ref);

  size_mB = num_bytes/1.e6;
  size_gB = size_mB/1.e3;

  for (int i=0; i<NTRIALS; i++) {
#if defined (DEMOTE_BUFFERS)
    demote_buffers(&num_elems, p_src1, p_src2, p_dst);
#endif

    t_start = dsecnd();

#if defined (USE_CPU)
    cpu_omp_triad(&num_elems, p_src1, p_src2, p_dst);
#elif defined (USE_DSA)
    /* dsa_omp_triad(&num_elems, p_src1, p_src2, p_dst, p_tmp1, p_tmp2, */
    /*               p_dml_jobs_t1, p_dml_jobs_t2); */
    dsa_omp_triad_v2(&num_elems, p_src1, p_src2, p_dst, p_tmp1, NULL,
                     p_dml_jobs_t1, NULL);
#endif

    p_t_elapsed[i] = dsecnd() - t_start;
    printf ("Iter[%d]: N = %ld, Size(MB) = %.2f, Time(us) = %.2f, Bandwidth(GB/s) = %.2f\n",
            i, num_elems, size_mB, p_t_elapsed[i]*1.e6, (3.*size_gB)/p_t_elapsed[i]); fflush(0);

    if (check_triad_results(&num_elems, p_dst_ref, p_dst)) {
      printf ("Validation failed!\n"); fflush(0);
      goto bailout;
    }
  }

  double t_best = FLT_MAX;
  double t_total = 0.;
  for (int i=0; i<NTRIALS; i++) {
    if (t_best > p_t_elapsed[i]) {
      t_best = p_t_elapsed[i];
    }
    t_total += p_t_elapsed[i];
  }
  double t_avg = t_total/NTRIALS;

  printf ("N = %ld, Size(MB) = %.2f, Best Time(us) = %.2f, Best Bandwidth(GB/s) = %.2f\n",
          num_elems, size_mB, t_best*1.e6, (3.*size_gB)/(t_best));

bailout:
  _mm_free(p_src1);
  _mm_free(p_src2);
  _mm_free(p_dst);
  _mm_free(p_dst_ref);
  _mm_free(p_t_elapsed);
#ifdef USE_DSA
  _mm_free(p_tmp1);
  _mm_free(p_tmp2);
  _mm_free(p_dml_jobs_t1);
  _mm_free(p_dml_jobs_t2);
#endif

  return 0;
}
