#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "immintrin.h"
#include "float.h"
#include "time.h"
#include "math.h"
#include "omp.h"
#include <unistd.h>
#include <numaif.h>

#ifdef USE_DSA
#include "dml/dml.h"
#endif

#define NTRIALS             (100)
#define ALIGN               (4*1024)
#define MAX_THREADS         (56)
#define QUEUE_DEPTH         (4)
#define BLK_SIZE_IN_BYTES   (1*1024*1024)
#define MAX_NUMA_NODES      (8)

#undef DEMOTE_BUFFERS

#if !defined(USE_CPU) && !defined(USE_DSA)
#error USE_CPU of USE_DSA should be defined
#endif

void demote_buffers(long long *p_n, double *p_src1, double *p_src2)
{
  long long n = *p_n;
  for (long long i=0; i<n; i+=8) {
#if 1
    _mm_clflushopt(p_src1); 
    _mm_clflushopt(p_src2); 
#else
    _mm_cldemote(p_src1);
    _mm_cldemote(p_src2);
#endif
    p_src1 += 8;
    p_src2 += 8;
  }
   _mm_mfence();
}

double seq_dot(long long *p_n, double *p_src1, double *p_src2)
{
  double tmp = 0.;

  for (long long i=0; i<*p_n; i++) {
    tmp += p_src1[i] * p_src2[i];
  }

  return tmp;
}

double cpu_omp_dot(long long *p_n, double *p_src1, double *p_src2)
{
  long long n = *p_n;
  double tmp = 0.;

#pragma omp parallel for reduction(+:tmp)
  for (long long i=0; i<n; i++) {
    tmp += p_src1[i] * p_src2[i];
  }

  return tmp;
}

double cpu_omp_dot_nthrs(long long *p_n, double *p_src1, double *p_src2, int nthrs)
{
  long long n = *p_n;
  double tmp = 0.;

#pragma omp parallel for reduction(+:tmp) num_threads(nthrs)
  for (long long i=0; i<n; i++) {
    tmp += p_src1[i] * p_src2[i];
  }

  return tmp;
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

void async_copy_numa(long long *p_n, double *p_src, double *p_dst, int numa_id, dml_job_t *p_dml_job)
{
  dml_status_t status;

  p_dml_job->operation             = DML_OP_MEM_MOVE;
  p_dml_job->flags                 = DML_FLAG_COPY_ONLY|DML_FLAG_PREFETCH_CACHE;
  p_dml_job->source_first_ptr      = (void *)p_src;
  p_dml_job->destination_first_ptr = (void *)p_dst;
  p_dml_job->source_length         = (*p_n)*sizeof(double);
  p_dml_job->destination_length    = (*p_n)*sizeof(double);
  p_dml_job->numa_id               = numa_id;

  status = dml_submit_job(p_dml_job);

  if (status) {
    printf ("\tdml_submit_job status failed, status = %u\n", status); fflush(0);
  }
}


double dsa_omp_dot(long long *p_n, double *p_src1, double *p_src2,
                   double *p_tmp1, double *p_tmp2,
                   dml_job_t **p_dml_jobs_t1, dml_job_t **p_dml_jobs_t2)
{
  long long n = *p_n;
  double tmp = 0.;
  double res_thrs[MAX_THREADS] = {0};

#pragma omp parallel
  {
    int nthrs = omp_get_num_threads();
    int ithr  = omp_get_thread_num();
    double res = 0.;

    long long chunk = n/nthrs;
    long long tail  = n - (chunk*nthrs);
    long long start = ithr * chunk;
    if ((tail) && (ithr == nthrs-1)) {
      chunk += tail;
    }

    double *p_t_src1 = p_src1 + start;
    double *p_t_src2 = p_src2 + start;

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
    res += seq_dot(&tmp_elems, p_t_src1, p_t_src2);

    int i;
    double t_start, t_dsa_elapsed=0., t_cpu_elapsed=0.;
    for (i=QUEUE_DEPTH; i<(num_blks-QUEUE_DEPTH); i++) {
      /* t_start = omp_get_wtime(); */
      dml_wait_job(p_t_dml_jobs_t1[i%QUEUE_DEPTH]);
      dml_wait_job(p_t_dml_jobs_t2[i%QUEUE_DEPTH]);
      /* t_dsa_elapsed += omp_get_wtime() - t_start; */

      /* t_start = omp_get_wtime(); */
      res += seq_dot(&blk_elems, p_t_tmp1+((i%QUEUE_DEPTH)*blk_elems), p_t_tmp2+((i%QUEUE_DEPTH)*blk_elems));
      /* t_cpu_elapsed += omp_get_wtime() - t_start; */

      async_copy(&blk_elems, p_t_src1+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp1+((i%QUEUE_DEPTH)*blk_elems), p_t_dml_jobs_t1[i%QUEUE_DEPTH]);
      async_copy(&blk_elems, p_t_src2+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp2+((i%QUEUE_DEPTH)*blk_elems), p_t_dml_jobs_t2[i%QUEUE_DEPTH]);
    }

    for (int j=i; j<(i+QUEUE_DEPTH); j++) {
      dml_wait_job(p_t_dml_jobs_t1[j%QUEUE_DEPTH]);
      dml_wait_job(p_t_dml_jobs_t2[j%QUEUE_DEPTH]);
      res += seq_dot(&blk_elems, p_t_tmp1+((j%QUEUE_DEPTH)*blk_elems), p_t_tmp2+((j%QUEUE_DEPTH)*blk_elems));
    }

    if (tail) {
      res += seq_dot(&tail, p_t_src1+(num_blks*blk_elems),  p_t_src2+(num_blks*blk_elems));
    }

    res_thrs[ithr] = res;
    /* printf ("TID-%d: DSA = %f us, CPU = %f us\n", ithr, t_dsa_elapsed*1.e6, t_cpu_elapsed*1.e6); fflush(0); */
  }

  for (int i=0; i<MAX_THREADS; i++) {
    tmp += res_thrs[i];
  }

  return tmp;
}

double dsa_omp_dot_v2(long long *p_n, double *p_src1, double *p_src2,
                   double *p_tmp1, double *p_tmp2,
                   dml_job_t **p_dml_jobs_t1, dml_job_t **p_dml_jobs_t2)
{
  long long n = *p_n;
  double tmp = 0.;
  double res_thrs[MAX_THREADS] = {0};

#pragma omp parallel
  {
    int nthrs = omp_get_num_threads();
    int ithr  = omp_get_thread_num();
    double res = 0.;

    long long chunk = n/nthrs;
    long long tail  = n - (chunk*nthrs);
    long long start = ithr * chunk;
    if ((tail) && (ithr == nthrs-1)) {
      chunk += tail;
    }

    double *p_t_src1 = p_src1 + start;
    double *p_t_src2 = p_src2 + start;

    double *p_t_tmp1 = p_tmp1 + (ithr * ((BLK_SIZE_IN_BYTES*QUEUE_DEPTH)/sizeof(double)));

    dml_job_t **p_t_dml_jobs_t1 = p_dml_jobs_t1 + (ithr*QUEUE_DEPTH);

    long long blk_elems = BLK_SIZE_IN_BYTES/sizeof(double);
    long long num_blks  = chunk/blk_elems;
    tail = chunk - (blk_elems*num_blks);

    for (int i=0; i<QUEUE_DEPTH; i++) {
      async_copy(&blk_elems, p_t_src1+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp1+(i*blk_elems), p_t_dml_jobs_t1[i]);
    }

    long long tmp_elems = QUEUE_DEPTH * blk_elems;
    res += seq_dot(&tmp_elems, p_t_src1, p_t_src2);

    int i;
    /* double t_start, t_dsa_elapsed=0., t_cpu_elapsed=0.; */
    for (i=QUEUE_DEPTH; i<(num_blks-QUEUE_DEPTH); i++) {
      /* t_start = omp_get_wtime(); */
      dml_wait_job(p_t_dml_jobs_t1[i%QUEUE_DEPTH]);
      /* t_dsa_elapsed += omp_get_wtime() - t_start; */

      /* t_start = omp_get_wtime(); */
      res += seq_dot(&blk_elems, p_t_tmp1+((i%QUEUE_DEPTH)*blk_elems), p_t_src2+(i*blk_elems));
      /* t_cpu_elapsed += omp_get_wtime() - t_start; */

      async_copy(&blk_elems, p_t_src1+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp1+((i%QUEUE_DEPTH)*blk_elems), p_t_dml_jobs_t1[i%QUEUE_DEPTH]);
    }

    for (int j=i; j<(i+QUEUE_DEPTH); j++) {
      dml_wait_job(p_t_dml_jobs_t1[j%QUEUE_DEPTH]);
      res += seq_dot(&blk_elems, p_t_tmp1+((j%QUEUE_DEPTH)*blk_elems), p_t_src2+(j*blk_elems));
    }

    if (tail) {
      res += seq_dot(&tail, p_t_src1+(num_blks*blk_elems),  p_t_src2+(num_blks*blk_elems));
    }

    res_thrs[ithr] = res;
    /* printf ("TID-%d: DSA = %f us, CPU = %f us\n", ithr, t_dsa_elapsed*1.e6, t_cpu_elapsed*1.e6); fflush(0); */
  }

  for (int i=0; i<MAX_THREADS; i++) {
    tmp += res_thrs[i];
  }

  return tmp;
}

double dsa_omp_dot_numa(long long *p_n, double *p_src1, double *p_src2,
                        double *p_tmp1, double *p_tmp2,
                        dml_job_t **p_dml_jobs_t1, dml_job_t **p_dml_jobs_t2,
                        int *p_src1_numa_ids, int *p_src2_numa_ids)
{
  long long n = *p_n;
  double tmp = 0.;
  double res_thrs[MAX_THREADS] = {0};

#pragma omp parallel
  {
    int nthrs = omp_get_num_threads();
    int ithr  = omp_get_thread_num();
    double res = 0.;

    long long chunk = n/nthrs;
    long long tail  = n - (chunk*nthrs);
    long long start = ithr * chunk;
    if ((tail) && (ithr == nthrs-1)) {
      chunk += tail;
    }

    double *p_t_src1 = p_src1 + start;
    double *p_t_src2 = p_src2 + start;

    double *p_t_tmp1 = p_tmp1 + (ithr * ((BLK_SIZE_IN_BYTES*QUEUE_DEPTH)/sizeof(double)));
    double *p_t_tmp2 = p_tmp2 + (ithr * ((BLK_SIZE_IN_BYTES*QUEUE_DEPTH)/sizeof(double)));

    /* double p_t_tmp1[((BLK_SIZE_IN_BYTES*QUEUE_DEPTH)/8)] = {0}; */
    /* double p_t_tmp2[((BLK_SIZE_IN_BYTES*QUEUE_DEPTH)/8)] = {0}; */

    int numa_id = ithr;

    dml_job_t **p_t_dml_jobs_t1 = p_dml_jobs_t1 + (ithr*QUEUE_DEPTH);
    dml_job_t **p_t_dml_jobs_t2 = p_dml_jobs_t2 + (ithr*QUEUE_DEPTH);

    long long blk_elems = BLK_SIZE_IN_BYTES/sizeof(double);
    long long num_blks  = chunk/blk_elems;
    tail = chunk - (blk_elems*num_blks);

    for (int i=0; i<QUEUE_DEPTH; i++) {
      async_copy_numa(&blk_elems, p_t_src1+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp1+(i*blk_elems), numa_id, p_t_dml_jobs_t1[i]);
      async_copy_numa(&blk_elems, p_t_src2+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp2+(i*blk_elems), numa_id, p_t_dml_jobs_t2[i]);
    }

    long long tmp_elems = QUEUE_DEPTH * blk_elems;
    res += cpu_omp_dot_nthrs(&tmp_elems, p_t_src1, p_t_src2, 4);

    int i;
    double t_start, t_dsa_elapsed=0., t_cpu_elapsed=0.;
    for (i=QUEUE_DEPTH; i<(num_blks-QUEUE_DEPTH); i++) {
      /* t_start = omp_get_wtime(); */
      dml_wait_job(p_t_dml_jobs_t1[i%QUEUE_DEPTH]);
      dml_wait_job(p_t_dml_jobs_t2[i%QUEUE_DEPTH]);
      /* t_dsa_elapsed += omp_get_wtime() - t_start; */

      /* t_start = omp_get_wtime(); */
      res += cpu_omp_dot_nthrs(&blk_elems, p_t_tmp1+((i%QUEUE_DEPTH)*blk_elems), p_t_tmp2+((i%QUEUE_DEPTH)*blk_elems), 4);
      /* t_cpu_elapsed += omp_get_wtime() - t_start; */

      async_copy_numa(&blk_elems, p_t_src1+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp1+((i%QUEUE_DEPTH)*blk_elems), numa_id, p_t_dml_jobs_t1[i%QUEUE_DEPTH]);
      async_copy_numa(&blk_elems, p_t_src2+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp2+((i%QUEUE_DEPTH)*blk_elems), numa_id, p_t_dml_jobs_t2[i%QUEUE_DEPTH]);
    }

    for (int j=i; j<(i+QUEUE_DEPTH); j++) {
      dml_wait_job(p_t_dml_jobs_t1[j%QUEUE_DEPTH]);
      dml_wait_job(p_t_dml_jobs_t2[j%QUEUE_DEPTH]);
      res += seq_dot(&blk_elems, p_t_tmp1+((j%QUEUE_DEPTH)*blk_elems), p_t_tmp2+((j%QUEUE_DEPTH)*blk_elems));
    }

    if (tail) {
      res += seq_dot(&tail, p_t_src1+(num_blks*blk_elems),  p_t_src2+(num_blks*blk_elems));
    }

    res_thrs[ithr] = res;
    /* printf ("TID-%d: DSA = %f us, CPU = %f us\n", ithr, t_dsa_elapsed*1.e6, t_cpu_elapsed*1.e6); fflush(0); */
  }

  for (int i=0; i<MAX_THREADS; i++) {
    tmp += res_thrs[i];
  }

  return tmp;
}


double dsa_omp_dot_ht(long long *p_n, double *p_src1, double *p_src2,
                   double *p_tmp1, double *p_tmp2,
                   dml_job_t **p_dml_jobs_t1, dml_job_t **p_dml_jobs_t2)
{
  long long n = *p_n;
  double tmp = 0.;
  double res_thrs[MAX_THREADS] = {0};

#pragma omp parallel
  {
    int nthrs = omp_get_num_threads();
    int ithr  = omp_get_thread_num();
    double res = 0.;

    long long chunk = n/nthrs;
    long long tail  = n - (chunk*nthrs);
    long long start = ithr * chunk;
    if ((tail) && (ithr == nthrs-1)) {
      chunk += tail;
    }

    double *p_t_src1 = p_src1 + start;
    double *p_t_src2 = p_src2 + start;

    if (ithr%2 == 0) {
      double *p_t_tmp1 = p_tmp1 + (ithr * ((BLK_SIZE_IN_BYTES*QUEUE_DEPTH)/sizeof(double)));

      dml_job_t **p_t_dml_jobs_t1 = p_dml_jobs_t1 + (ithr*QUEUE_DEPTH);

      long long blk_elems = BLK_SIZE_IN_BYTES/sizeof(double);
      long long num_blks  = chunk/blk_elems;
      tail = chunk - (blk_elems*num_blks);

      for (int i=0; i<QUEUE_DEPTH; i++) {
        async_copy(&blk_elems, p_t_src1+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp1+(i*blk_elems), p_t_dml_jobs_t1[i]);
      }

      long long tmp_elems = QUEUE_DEPTH * blk_elems;
      res += seq_dot(&tmp_elems, p_t_src1, p_t_src2);

      int i;
      double t_start, t_dsa_elapsed=0., t_cpu_elapsed=0.;
      for (i=QUEUE_DEPTH; i<(num_blks-QUEUE_DEPTH); i++) {
        /* t_start = omp_get_wtime(); */
        dml_wait_job(p_t_dml_jobs_t1[i%QUEUE_DEPTH]);
        /* t_dsa_elapsed += omp_get_wtime() - t_start; */

        /* t_start = omp_get_wtime(); */
        res += seq_dot(&blk_elems, p_t_tmp1+((i%QUEUE_DEPTH)*blk_elems), p_t_src2+(i*blk_elems));
        /* t_cpu_elapsed += omp_get_wtime() - t_start; */

        async_copy(&blk_elems, p_t_src1+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp1+((i%QUEUE_DEPTH)*blk_elems), p_t_dml_jobs_t1[i%QUEUE_DEPTH]);
      }

      for (int j=i; j<(i+QUEUE_DEPTH); j++) {
        dml_wait_job(p_t_dml_jobs_t1[j%QUEUE_DEPTH]);
        res += seq_dot(&blk_elems, p_t_tmp1+((j%QUEUE_DEPTH)*blk_elems), p_t_src2+(j*blk_elems));
      }

      if (tail) {
        res += seq_dot(&tail, p_t_src1+(num_blks*blk_elems),  p_t_src2+(num_blks*blk_elems));
      }

      res_thrs[ithr] = res;
      /* printf ("TID-%d: DSA = %f us, CPU = %f us\n", ithr, t_dsa_elapsed*1.e6, t_cpu_elapsed*1.e6); fflush(0); */
    } else {
        res_thrs[ithr] = seq_dot(&chunk, p_t_src1, p_t_src2);
    }
  }

  for (int i=0; i<MAX_THREADS; i++) {
    tmp += res_thrs[i];
  }

  return tmp;
}


#if 0
double dsa_omp_dot_obsolete(long long *p_n, double *p_src1, double *p_src2,
                   double *p_tmp1, double *p_tmp2,
                   dml_job_t **p_dml_jobs_t1, dml_job_t **p_dml_jobs_t2)
{
  long long n = *p_n;
  double tmp = 0.;
  double res_thrs[MAX_THREADS] = {0};

#pragma omp parallel
  {
    int nthrs = omp_get_num_threads();
    int ithr  = omp_get_thread_num();
    double res = 0.;

    long long chunk = BLK_SIZE_IN_BYTES/sizeof(double);
    long long start = ithr * chunk;
    double *p_t_src1 = p_src1 + start;
    double *p_t_src2 = p_src2 + start;

    long long blk_elems = nthrs * chunk;
    long long num_blks  = n/blk_elems;
    long long tail      = n - (blk_elems*num_blks);

    if (ithr == 0) {
      for (int i=0; i<QUEUE_DEPTH; i++) {
        async_copy(&blk_elems, p_t_src1+((i+1)*blk_elems), p_tmp1+(i*blk_elems), p_dml_jobs_t1[i]);
      }
    }

    res += seq_dot(&chunk, p_t_src1, p_t_src2);

    int i;
    for (i=QUEUE_DEPTH; i<(num_blks-QUEUE_DEPTH); i++) {
      if (ithr == 0) {
        dml_wait_job(p_dml_jobs_t1[i%QUEUE_DEPTH]);
      }

#pragma omp barrier

      res += seq_dot(&chunk, p_tmp1+((i%QUEUE_DEPTH)*blk_elems)+(ithr*chunk), p_t_src2+((i-1)*blk_elems)+(0*chunk));

      if (ithr == 0) {
        async_copy(&blk_elems, p_t_src1+((i+1)*blk_elems), p_tmp1+((i%QUEUE_DEPTH)*blk_elems), p_dml_jobs_t1[i%QUEUE_DEPTH]);
      }
    }


    for (int j=i; j<(i+QUEUE_DEPTH); j++) {
      if (ithr == 0) {
        dml_wait_job(p_dml_jobs_t1[j%QUEUE_DEPTH]);
      }
#pragma omp barrier
      res += seq_dot(&chunk, p_tmp1+((j%QUEUE_DEPTH)*blk_elems)+(ithr*chunk), p_t_src2+((j-1)*blk_elems)+(0*chunk));
    }


    if (tail) {
#pragma omp for
      for(long long i=blk_elems*num_blks; i<n; i++) {
        res += p_src1[i] * p_src2[i];
      }
    }

    res_thrs[ithr] = res;
  }

  for (int i=0; i<MAX_THREADS; i++) {
    tmp += res_thrs[i];
  }

  return tmp;
}
#endif

#endif

int main (int argc, char **argv)
{
  double *p_src1 = NULL, *p_src2 = NULL, *p_dst = NULL, *p_dst_ref = NULL, *p_t_elapsed = NULL;
  double t_start, size_mB, size_gB, res, ref_res;
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
  p_t_elapsed = (double *)_mm_malloc(sizeof(double)*NTRIALS, ALIGN);

  srand48(42);

#pragma omp parallel for
  for (long long i=0; i<num_elems; i++) {
    p_src1[i] = i%124; //rand()%1024;
    p_src2[i] = i%124; //rand()%1024;
  }


  long page_size = sysconf(_SC_PAGESIZE);
  long num_pages = num_bytes/page_size;
  printf ("Page size = %ld bytes, Number of pages in each buffer = %ld\n",
          page_size, num_pages);
  int *p_src1_numa_ids = (int *)_mm_malloc(sizeof(int)*num_pages, ALIGN);
  int *p_src2_numa_ids = (int *)_mm_malloc(sizeof(int)*num_pages, ALIGN);

  for (long long i=0; i<num_pages; i++) {
    get_mempolicy(&p_src1_numa_ids[i], NULL, MAX_NUMA_NODES, &p_src1[i*512], MPOL_F_NODE|MPOL_F_ADDR);
    get_mempolicy(&p_src2_numa_ids[i], NULL, MAX_NUMA_NODES, &p_src2[i*512], MPOL_F_NODE|MPOL_F_ADDR);
    /* printf ("page[%ld]: p_src1 = %p, numa_id = %d\n", i, &p_src1[i*512], p_src1_numa_ids[i]); */
    /* printf ("page[%ld]: p_src2 = %p, numa_id = %d\n", i, &p_src2[i*512], p_src2_numa_ids[i]); */
    /* fflush(0); */
  }

  int src1_page_count_in_numa[MAX_NUMA_NODES] = {0};
  int src2_page_count_in_numa[MAX_NUMA_NODES] = {0};
  for (long long i=0; i<num_pages; i++) {
    src1_page_count_in_numa[p_src1_numa_ids[i]]++;
    src2_page_count_in_numa[p_src2_numa_ids[i]]++;
  }

  printf ("N0\tN1\tN2\tN3\tN4\tN5\tN6\tN7\n");
  /* printf ("src1:"); */
  for (int i=0; i<MAX_NUMA_NODES; i++) {
    printf("%d\t", src1_page_count_in_numa[i]);
  }
  printf ("\n");
  /* printf ("src2:"); */
  for (int i=0; i<MAX_NUMA_NODES; i++) {
    printf("%d\t", src2_page_count_in_numa[i]);
  }
  printf ("\n");

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
  for (long long i=0; i<((MAX_THREADS*BLK_SIZE_IN_BYTES*QUEUE_DEPTH)/sizeof(double)); i++) {
    p_tmp1[i] = 5.;
    p_tmp2[i] = 5.;
  }
#endif

  ref_res = seq_dot(&num_elems, p_src1, p_src2);

  size_mB = num_bytes/1.e6;
  size_gB = size_mB/1.e3;

  for (int i=0; i<NTRIALS; i++) {
#if defined (DEMOTE_BUFFERS)
    demote_buffers(&num_elems, p_src1, p_src2);
#endif

    t_start = dsecnd();

#if defined (USE_CPU)
    res = cpu_omp_dot(&num_elems, p_src1, p_src2);
#elif defined (USE_DSA)
    /* res = dsa_omp_dot(&num_elems, p_src1, p_src2, p_tmp1, p_tmp2, p_dml_jobs_t1, p_dml_jobs_t2); */
    res = dsa_omp_dot_v2(&num_elems, p_src1, p_src2, p_tmp1, NULL, p_dml_jobs_t1, NULL);
    /* res = dsa_omp_dot_numa(&num_elems, p_src1, p_src2, p_tmp1, p_tmp2, p_dml_jobs_t1, p_dml_jobs_t2, */
                           /* p_src1_numa_ids, p_src2_numa_ids); */
#endif

    p_t_elapsed[i] = dsecnd() - t_start;
    printf ("Iter[%d]: N = %ld, Size(MB) = %.2f, Time(us) = %.2f, Bandwidth(GB/s) = %.2f\n",
            i, num_elems, size_mB, p_t_elapsed[i]*1.e6, (2.*size_gB)/p_t_elapsed[i]); fflush(0);

    if (fabs(res - ref_res) > 1.e-04) {
      printf ("validation failed. Expected = %lf, Observed = %lf\n", ref_res, res);
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

  printf ("N = %ld, Size(MB) = %.2f, Avg Bandwidth(GB/s) = %.2f, Best Time(us) = %.2f, Best Bandwidth(GB/s) = %.2f\n",
          num_elems, size_mB, (2.*size_gB)/(t_avg), t_best*1.e6, (2.*size_gB)/(t_best));

bailout:
  _mm_free(p_src1);
  _mm_free(p_src2);
  _mm_free(p_t_elapsed);
  _mm_free(p_src1_numa_ids);
  _mm_free(p_src2_numa_ids);
#ifdef USE_DSA
  _mm_free(p_tmp1);
  _mm_free(p_tmp2);
  _mm_free(p_dml_jobs_t1);
  _mm_free(p_dml_jobs_t2);
#endif

  return 0;
}
