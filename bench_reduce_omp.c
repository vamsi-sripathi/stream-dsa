#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "float.h"
#include "time.h"
#include "math.h"
#include "mkl.h"
#include "omp.h"
#include "immintrin.h"
#include <unistd.h>

#define NTRIALS            (100)
#define ALIGN              (4*1024) 
#define QUEUE_DEPTH        (4)
#define MAX_THREADS        (56)
#define BLK_SIZE_IN_BYTES  (1024*1024)

#if !defined(USE_CPU) && !defined(USE_DSA)
#error USE_CPU of USE_DSA should be defined
#endif

#undef DEMOTE_BUFFERS

void demote_buffers(long long *p_n, double *p_src)
{
  long long n = *p_n;
  for (long long i=0; i<n; i+=8) {
    /* _mm_cldemote(p_src); */
    _mm_clflushopt(p_src); 
    p_src += 8;
  }
}
 
double cpu_reduce(long long *p_n, double *p_src)
{
  double tmp = 0.;

  for (long long i=0; i<*p_n; i++) {
    tmp += p_src[i];
  }

  return tmp;
}

double cpu_omp_reduce(long long *p_n, double *p_src)
{
  long long n = *p_n;
  double tmp = 0.;

#pragma omp parallel for reduction(+:tmp)
  for (long long i=0; i<n; i++) {
    tmp += p_src[i];
  }

  return tmp;
}


#if defined(USE_DSA)
#include "dml/dml.h"

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
    printf ("\tdml_execute_job failed, status = %u\n", status);
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
    printf ("\tdml_submit_job failed, status = %u\n", status);
  }
}

double dsa_omp_reduce(long long *p_n, double *p_src, double *p_tmp, dml_job_t **p_dml_jobs)
{
  long long n = *p_n;
  double tmp = 0.0;
  double res_thrs[MAX_THREADS] = {0};

#pragma omp parallel
  {
    int nthrs  = omp_get_num_threads();
    int ithr   = omp_get_thread_num();

    double res      = 0.0;
    long long chunk = n/nthrs;
    long long tail  = n - (chunk*nthrs);
    long long start = ithr * chunk;
    if ((tail) && (ithr == nthrs-1)) {
      chunk += tail;
    }

    double *p_t_src = p_src + start;
    double *p_t_tmp = p_tmp + (ithr * ((BLK_SIZE_IN_BYTES*QUEUE_DEPTH)/sizeof(double)));
    dml_job_t **p_t_dml_jobs = p_dml_jobs + (ithr*QUEUE_DEPTH);

    long long blk_elems = BLK_SIZE_IN_BYTES/sizeof(double);
    long long num_blks  = chunk/blk_elems;
    tail = chunk - (blk_elems*num_blks);

    for (int i=0; i<QUEUE_DEPTH; i++) {
      async_copy(&blk_elems, p_t_src+((QUEUE_DEPTH+i)*blk_elems), p_t_tmp+(i*blk_elems), p_t_dml_jobs[i]);
    }

    long long tmp_elems = QUEUE_DEPTH * blk_elems;
    res += cpu_reduce(&tmp_elems, p_t_src);

    double t_start, t_dsa_elapsed=0., t_cpu_elapsed=0.;
    for (int j=QUEUE_DEPTH; j<(num_blks-QUEUE_DEPTH); j++) {
      /* t_start = omp_get_wtime(); */
      dml_wait_job(p_t_dml_jobs[j%QUEUE_DEPTH]);
      /* t_dsa_elapsed += omp_get_wtime() - t_start; */

      /* t_start = omp_get_wtime(); */
      res += cpu_reduce(&blk_elems, p_t_tmp+((j%QUEUE_DEPTH)*blk_elems));
      /* t_cpu_elapsed += omp_get_wtime() - t_start; */

      async_copy(&blk_elems, p_t_src+((j+QUEUE_DEPTH)*blk_elems), p_t_tmp+((j%QUEUE_DEPTH)*blk_elems), p_t_dml_jobs[j%QUEUE_DEPTH]);
    }

    for (int i=0; i<QUEUE_DEPTH; i++) {
      dml_wait_job(p_t_dml_jobs[i]);
      res += cpu_reduce(&blk_elems, p_t_tmp+(i*blk_elems));
    }

    if (tail) {
      res += cpu_reduce(&tail, p_t_src+(num_blks*blk_elems));
    }

    res_thrs[ithr] = res;
    /* printf ("TID-%d: DSA = %f us, CPU = %f us\n", ithr, t_dsa_elapsed*1.e6, t_cpu_elapsed*1.e6); fflush(0); */
  }

  for (int i=0; i<MAX_THREADS; i++) {
    tmp += res_thrs[i];
  }

  return tmp;
}


double dsa_omp_reduce_v2(long long *p_n, double *p_src, double *p_tmp, dml_job_t **p_dml_jobs)
{
  long long n = *p_n;
  double res = 0.0;
  double res_thrs[MAX_THREADS] = {0};

#define CHUNK_SIZE_IN_BYTES (4*1024*1024)

#pragma omp parallel
  {
    int nthrs  = omp_get_num_threads();
    int ithr   = omp_get_thread_num();

    long long chunk_elems = CHUNK_SIZE_IN_BYTES/sizeof(double);
    long long blk_elems   = chunk_elems * QUEUE_DEPTH;
    long long num_blks    = n/blk_elems;
    long long tail        = n - (num_blks*blk_elems);

    if (ithr == 0) {
      for (int i=0; i<QUEUE_DEPTH; i++) {
        async_copy(&chunk_elems, p_src+((QUEUE_DEPTH+i)*chunk_elems), p_tmp+(i*chunk_elems), p_dml_jobs[i]);
      }
    }

#pragma omp for reduction(+:res)
    for (long long i=0; i<blk_elems; i++) {
      res += p_src[i];
    }

    if (ithr == 0) {
      for (int i=0; i<QUEUE_DEPTH; i++) {
        dml_wait_job(p_dml_jobs[i]);
      }
    }

    for (long long i=1; i<num_blks-1; i++) {
      if (ithr == 0) {
        for (int j=0; j<QUEUE_DEPTH; j++) {
          /* async_copy(&chunk_elems, p_src+(((i+1)*blk_elems)+(j*chunk_elems)), p_tmp+((i%2)*blk_elems)+(j*chunk_elems), p_dml_jobs[j]); */
        }
      }

/* #pragma omp barrier */
#pragma omp for reduction(+:res)
      for (long long j=0; j<blk_elems; j++) {
        res += *(p_tmp+(((i-1)%2)*blk_elems)+j);
      }

      if (ithr == 0) {
        for (int j=0; j<QUEUE_DEPTH; j++) {
          /* dml_wait_job(p_dml_jobs[j]); */
        }
      }
    }

#pragma omp barrier
#pragma omp for reduction(+:res)
    for (long long j=0; j<blk_elems; j++) {
      res += *(p_tmp+(((num_blks-2)%2)*blk_elems)+j);
    }

    if (tail) {
#pragma omp for reduction(+:res)
      for (long long j=num_blks*blk_elems; j<n; j++) {
        res += p_src[j];
      }
    }

  }
  return res;
}

double dsa_omp_reduce_v3(long long *p_n, double *p_src, double *p_tmp, dml_job_t **p_dml_jobs)
{
  long long n = *p_n;
  double res = 0.0;
  double res_thrs[MAX_THREADS] = {0};

#define CHUNK_SIZE_IN_BYTES (16*1024*1024)

#pragma omp parallel
  {
    int nthrs  = omp_get_num_threads();
    int ithr   = omp_get_thread_num();

    long long chunk_elems = CHUNK_SIZE_IN_BYTES/sizeof(double);
    long long blk_elems   = chunk_elems * QUEUE_DEPTH;
    long long num_blks    = n/blk_elems;
    long long tail        = n - (num_blks*blk_elems);

    if (ithr%4 == 0) {
      for (int i=0; i<QUEUE_DEPTH; i++) {
        async_copy(&chunk_elems, p_src+((QUEUE_DEPTH+i)*chunk_elems), p_tmp+(i*chunk_elems), p_dml_jobs[i]);
      }
    }

#pragma omp for reduction(+:res)
    for (long long i=0; i<blk_elems; i++) {
      res += p_src[i];
    }

    if (ithr == 0) {
      for (int i=0; i<QUEUE_DEPTH; i++) {
        dml_wait_job(p_dml_jobs[i]);
      }
    }

    for (long long i=1; i<num_blks-1; i++) {
      if (ithr == 0) {
        for (int j=0; j<QUEUE_DEPTH; j++) {
          /* async_copy(&chunk_elems, p_src+(((i+1)*blk_elems)+(j*chunk_elems)), p_tmp+((i%2)*blk_elems)+(j*chunk_elems), p_dml_jobs[j]); */
        }
      }

/* #pragma omp barrier */
#pragma omp for reduction(+:res)
      for (long long j=0; j<blk_elems; j++) {
        res += *(p_tmp+(((i-1)%2)*blk_elems)+j);
      }

      if (ithr == 0) {
        for (int j=0; j<QUEUE_DEPTH; j++) {
          /* dml_wait_job(p_dml_jobs[j]); */
        }
      }
    }

#pragma omp barrier
#pragma omp for reduction(+:res)
    for (long long j=0; j<blk_elems; j++) {
      res += *(p_tmp+(((num_blks-2)%2)*blk_elems)+j);
    }

    if (tail) {
#pragma omp for reduction(+:res)
      for (long long j=num_blks*blk_elems; j<n; j++) {
        res += p_src[j];
      }
    }

  }
  return res;
}



double dsa_seq_reduce(long long *p_n, double *p_src, double *p_tmp,  dml_job_t **p_dml_jobs)
{
  long long num_elems   = *p_n;
  long long blk_elems   = BLK_SIZE_IN_BYTES/sizeof(double);
  long long num_blks    = num_elems/blk_elems;
  long long tail        = num_elems - (blk_elems * num_blks);

  double res = 0.;

  for (int i=0; i<QUEUE_DEPTH; i++) {
    async_copy(&blk_elems, p_src+((QUEUE_DEPTH+i)*blk_elems), p_tmp+(i*blk_elems), p_dml_jobs[i]);
  }

  long long tmp_elems = QUEUE_DEPTH * blk_elems;
  res += cpu_reduce(&tmp_elems, p_src);

  for (int j=QUEUE_DEPTH; j<(num_blks-QUEUE_DEPTH); j++) {
    dml_wait_job(p_dml_jobs[j%QUEUE_DEPTH]);

    res += cpu_reduce(&blk_elems, p_tmp+((j%QUEUE_DEPTH)*blk_elems));

    async_copy(&blk_elems, p_src+((j+QUEUE_DEPTH)*blk_elems), p_tmp+((j%QUEUE_DEPTH)*blk_elems), p_dml_jobs[j%QUEUE_DEPTH]);
  }

  for (int i=0; i<QUEUE_DEPTH; i++) {
    dml_wait_job(p_dml_jobs[i]);
    res += cpu_reduce(&blk_elems, p_tmp+(i*blk_elems));
  }

  if (tail) {
    res += cpu_reduce(&tail, p_src+(num_blks*blk_elems));
  }


  return res;

}
#endif

int main (int argc, char **argv)
{
  double *p_src = NULL, *p_tmp = NULL, *p_t_elapsed = NULL;
  double res, t_start;
  double size_mB, size_gB;
  long long num_bytes, num_blks, blk_elems, num_elems, tail;

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
  
  num_elems   = num_bytes/sizeof(double);
  p_src       = (double *)_mm_malloc(num_bytes, ALIGN);
  p_tmp       = (double *)_mm_malloc(MAX_THREADS*BLK_SIZE_IN_BYTES*QUEUE_DEPTH, ALIGN);
  p_t_elapsed = (double *)_mm_malloc(sizeof(double)*NTRIALS, ALIGN);

  srand(42);
//#pragma omp parallel for
  for (long long i=0; i<num_elems; i++) {
    /* p_src[i] = drand48(); */
    /* p_src[i] = 3.5; */
    p_src[i] = rand()%1024;
  }

#if defined (USE_DSA)
  dml_job_t** p_dml_jobs = NULL;
  p_dml_jobs = (dml_job_t **)_mm_malloc(sizeof(dml_job_t *)*MAX_THREADS*QUEUE_DEPTH, ALIGN);

//#pragma omp parallel for
  for (int i=0; i<MAX_THREADS*QUEUE_DEPTH; i++) {
    p_dml_jobs[i] = init_dml();
  }

//#pragma omp parallel for
  for (long long i=0; i<(MAX_THREADS*BLK_SIZE_IN_BYTES*QUEUE_DEPTH)/sizeof(double); i++) {
    p_tmp[i] = 5.;
  }

#endif

  double ref_res = cpu_reduce(&num_elems, p_src);

  size_mB = num_bytes/1.e6;
  size_gB = size_mB/1.e3;

  for (int i=0; i<NTRIALS; i++) {
#if defined (DEMOTE_BUFFERS)
    demote_buffers(&num_elems, p_src);
#endif
    res = 0.;
    t_start = dsecnd();

#if defined (USE_CPU)
    res = cpu_omp_reduce(&num_elems, p_src);
#elif defined (USE_DSA)
    res = dsa_omp_reduce(&num_elems, p_src, p_tmp, p_dml_jobs);
#endif

    if (fabs(res - ref_res) > 1.e-04) {
      printf ("validation failed. Expected = %lf, Observed = %lf\n", ref_res, res);
      goto bailout;
    }

    p_t_elapsed[i] = dsecnd() - t_start;
    /* printf ("Iter[%d]: N = %ld, Size(MB) = %.2f, Time (us) = %.2f, Bandwidth(GB/s) = %.4f\n", */
    /*         i, num_elems, size_mB, p_t_elapsed[i]*1.e6, (size_gB)/(p_t_elapsed[i])); */
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

  printf ("N = %ld, Size(MB) = %.2f, Best Time(us) = %.2f, Best Bandwidth(GB/s) = %.4f\n", num_elems, size_mB, t_best*1.e6, (size_gB)/(t_best));

bailout:
  _mm_free(p_src);
  _mm_free(p_tmp);
  _mm_free(p_t_elapsed);
#ifdef USE_DSA
  _mm_free(p_dml_jobs);
#endif

  return 0;
}
